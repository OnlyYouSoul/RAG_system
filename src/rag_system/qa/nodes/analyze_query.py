from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_system.qa.llm import get_chat_model
from rag_system.qa.state import QAState, QueryFilters

# LLM 不可用时选择的意图
_FALLBACK_INTENT = "retrieval_qa"

# 允许过滤的字段白名单（编译 Milvus 表达式时按此顺序）
_FILTERABLE_FIELDS = ("kb_id", "department", "doc_type", "language", "document_id")

# 意图类别，注入到 system_prompt 的 {{categories}}
_INTENT_CATEGORIES = {
    "retrieval_qa": "需要检索知识库正文才能回答（询问知识、事实、文档内容、操作方法等）",
    "metadata_qa": (
        "针对文档元数据本身的统计/罗列/筛选类问题，无需看正文内容，"
        "例如「policy 分类下有哪几篇文档」「7月10号入库的文档有哪些」"
        "「infra 部门共有多少篇文档」「按入库时间列出最近的文档」"
    ),
    "chitchat": "寒暄、问候、闲聊、情绪表达，或明显与知识库无关，无需检索",
}

# 当前支持的文档类型，注入到 filterable_metadata 的 {{doc_categaries}}
_DOC_CATEGORIES = ("pdf", "docx", "pptx", "xlsx", "txt", "md")

# doc_type 归一化
_DOC_TYPE_ALIASES = {
    "pdf": "pdf",
    "word": "docx",
    "doc": "docx",
    "docx": "docx",
    "ppt": "pptx",
    "pptx": "pptx",
    "excel": "xlsx",
    "xls": "xlsx",
    "xlsx": "xlsx",
    "text": "txt",
    "txt": "txt",
    "markdown": "md",
    "md": "md",
}


class QueryAnalysis(BaseModel):

    intent: str = Field(
        description="意图类别，只能是 'retrieval_qa'（需检索知识库正文）、"
        "'metadata_qa'（针对文档元数据的统计/罗列/筛选，无需正文）"
        " 或 'chitchat'（闲聊、与知识库无关）",
    )
    confidence: float = Field(
        description="意图判定置信度，0~1 之间的小数",
        ge=0.0,
        le=1.0,
    )
    department: Optional[str] = Field(
        default=None, description="部门，如 infra / hr / finance；查询未提及则为 null"
    )
    doc_type: Optional[str] = Field(
        default=None,
        description="文档类型，如 pdf / docx / pptx / xlsx / txt / md；查询未提及则为 null",
    )
    language: Optional[str] = Field(
        default=None, description="语言，如 zh / en；查询未提及则为 null"
    )


# system_prompt 模板，{{categories}} 在运行时替换为实际类别说明
_SYSTEM_PROMPT_TEMPLATE = (
    "你是一个智能助手，负责对用户的问题进行意图分类，并从问题中抽取可用于过滤"
    "知识库的结构化条件。\n"
    "目前的意图类别如下：{categories}\n"
    "过滤字段的取值说明见 filterable_metadata。只抽取查询中明确提到的信息，"
    "未提及的字段一律留空(null)，不要臆测或编造。\n"
    "只输出结构化结果，不要额外解释。"
)

# 传给 LLM 的 human 消息：完整结构化 JSON payload
_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{system_prompt}"),
        ("human", "请依据以下 JSON 请求进行分析：\n{payload}"),
    ]
)


def _render_system_prompt() -> str:
    categories = "；".join(f"{k}（{v}）" for k, v in _INTENT_CATEGORIES.items())
    return _SYSTEM_PROMPT_TEMPLATE.format(categories=categories)


def _build_payload(state: QAState) -> dict[str, Any]:
    """把当前请求组织成任务约定的结构化 JSON。"""
    doc_categories = "、".join(_DOC_CATEGORIES)
    return {
        "system_prompt": _render_system_prompt(),
        "query": state["query"],
        "chat_history": state.get("chat_history", []) or [],
        "user_context": state.get("user_context", {}) or {},
        "metadata_schema": {
            "filtered_fields": {
                "doc_type": "文档类型",
                "department": "部门",
                "title": "标题",
                "source_file": "文件名",
                "ingested_at": "入库时间",
            }
        },
        "filterable_metadata": {
            "doc_type": f"文档类型，当前文档类型有{doc_categories}",
            "language": "语言",
            "ingested_at": "入库时间",
        },
        "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _escape(value: str) -> str:
    return value.replace('"', '\\"')


def _compile_expr(filters: QueryFilters) -> str:
    clauses = []
    for field in _FILTERABLE_FIELDS:
        value = filters.get(field)
        if value:
            clauses.append(f'{field} == "{_escape(str(value))}"')
    return " and ".join(clauses)


def _extracted_filters(result: QueryAnalysis) -> QueryFilters:
    """从 LLM 结果里取出过滤字段并做归一化。"""
    extracted: QueryFilters = {}
    if result.department:
        extracted["department"] = result.department.strip()
    if result.doc_type:
        normalized = _DOC_TYPE_ALIASES.get(result.doc_type.strip().lower())
        if normalized:
            extracted["doc_type"] = normalized
    if result.language:
        extracted["language"] = result.language.strip().lower()
    return extracted


def analyze_query(state: QAState) -> QAState:
    """意图分类 + 元数据过滤构建，一次 LLM 调用完成。"""
    if state.get("error"):
        return {}

    payload = _build_payload(state)
    request_filters: QueryFilters = state.get("request_filters", {}) or {}

    intent = _FALLBACK_INTENT
    confidence = 0.0
    extracted: QueryFilters = {}
    error: Optional[str] = None

    try:
        model = get_chat_model().with_structured_output(QueryAnalysis)
        chain = _prompt | model
        result: QueryAnalysis = chain.invoke(
            {
                "system_prompt": payload["system_prompt"],
                "payload": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        )
        intent = result.intent if result.intent in _INTENT_CATEGORIES else _FALLBACK_INTENT
        confidence = result.confidence
        extracted = _extracted_filters(result)
    except Exception as exc:  # LLM 不可用时回退
        error = f"查询分析失败，回退为 {_FALLBACK_INTENT}：{exc}"

    out: QAState = {
        "intent": intent,
        "intent_confidence": confidence,
        "analysis_payload": payload,
    }
    if error:
        out["error"] = error

    # 闲聊、元数据问答都不走 Milvus 检索，无需编译过滤表达式
    # （metadata_qa 由 generate_sql 直接写 SQL 过滤）
    if intent in ("chitchat", "metadata_qa"):
        out["filters"] = {}
        out["milvus_expr"] = ""
        return out

    # 归并：显式传入的 request_filters 覆盖 LLM 抽取结果
    merged: QueryFilters = {}
    for field in _FILTERABLE_FIELDS:
        if request_filters.get(field):
            merged[field] = request_filters[field]
        elif extracted.get(field):
            merged[field] = extracted[field]

    out["filters"] = merged
    out["milvus_expr"] = _compile_expr(merged)
    return out
