"""生成回答：根据意图与上下文充分性，走三种话术。

1. chitchat：无检索上下文，直接自然回复。
2. retrieval_qa + 充分：严格基于压缩后的上下文作答，带来源引用 [n]。
3. retrieval_qa + 不足：基于弱上下文尽力回答，并明确提示资料可能不足，
   不得编造；无任何命中时直接说明无法回答。

产出 state["answer"]；检索问答分支额外产出 state["citations"]（来源清单）。
LLM 不可用时回退为提示信息，不抛异常。
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from rag_system.qa.llm import get_chat_model
from rag_system.qa.state import QAState

# 用于拼接带编号的上下文，便于 LLM 引用 [n]
_SOURCE_TEMPLATE = "[{n}] {text}"

# metadata_qa：序列化后的结果行喂给 LLM 的总字符上限（兜底，防 prompt 膨胀）
_METADATA_ROWS_MAX_CHARS = int(os.getenv("QA_METADATA_ROWS_MAX_CHARS", "6000"))

# metadata_qa：把 SQL 查询结果转成自然语言
_METADATA_SYSTEM = (
    "你是一个知识库元数据助手。下面给出的是针对文档元数据的数据库查询结果"
    "（JSON 行数组）。请用简洁、自然的中文回答用户问题。\n"
    "要求：\n"
    "1. 只依据查询结果作答，不要编造结果之外的内容。\n"
    "2. 结果为空时如实说明「没有符合条件的文档」。\n"
    "3. 涉及罗列时可用简短列表；涉及计数直接给出数字。"
)

_METADATA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _METADATA_SYSTEM),
        (
            "human",
            "用户问题：{query}\n\n查询结果（共 {row_count} 行）：\n{rows}\n\n请据此作答。",
        ),
    ]
)

_CHITCHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个友好的智能助手。用自然、简洁的中文回应用户的寒暄或闲聊。"),
        ("human", "{query}"),
    ]
)

_QA_SYSTEM = (
    "你是一个严谨的知识库问答助手。请严格依据提供的【上下文】回答用户问题。\n"
    "要求：\n"
    "1. 只使用上下文中的信息，不要编造上下文之外的内容。\n"
    "2. 在引用具体信息处标注来源编号，如 [1]、[2]。\n"
    "3. 若上下文不足以完整回答，明确指出哪些信息缺失，不要臆测。\n"
    "4. 用简洁清晰的中文作答。"
)

_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _QA_SYSTEM),
        (
            "human",
            "用户问题：{query}\n\n【上下文】\n{context}\n\n"
            "{insufficient_note}请依据上述上下文作答，并在引用处标注来源编号。",
        ),
    ]
)


def _hit_text(hit: dict) -> str:
    return str(hit.get("compressed_text") or hit.get("text") or "").strip()


def _build_context(hits: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """拼接带编号的上下文，并生成引用清单。"""
    blocks: list[str] = []
    citations: list[dict[str, Any]] = []
    for n, hit in enumerate(hits, start=1):
        text = _hit_text(hit)
        if not text:
            continue
        blocks.append(_SOURCE_TEMPLATE.format(n=n, text=text))
        citations.append(
            {
                "n": n,
                "document_id": hit.get("document_id"),
                "title": hit.get("title"),
                "chunk_id": hit.get("chunk_id"),
                "chunk_index": hit.get("chunk_index"),
            }
        )
    return "\n\n".join(blocks), citations


def _generate_chitchat(query: str) -> QAState:
    try:
        chain = _CHITCHAT_PROMPT | get_chat_model()
        resp = chain.invoke({"query": query})
        return {"answer": resp.content, "citations": []}
    except Exception as exc:
        return {"answer": f"（回复生成失败：{exc}）", "citations": []}


def _generate_metadata(state: QAState) -> QAState:
    """metadata_qa：基于 SQL 查询结果生成自然语言回答。

    execute_sql 若最终仍失败（sql_error 有值、重试用尽），这里兜底提示。
    """
    query = state["query"]

    if state.get("sql_error"):
        return {
            "answer": "抱歉，我没能把这个元数据问题转换成可执行的查询，请换个说法再试。",
            "citations": [],
        }

    rows = state.get("sql_rows", []) or []
    row_count = state.get("sql_row_count", len(rows)) or len(rows)
    if not rows:
        return {"answer": "没有找到符合条件的文档。", "citations": []}

    # rows 已在 execute_sql 做过行数与单格截断，这里再对序列化后的总长度兜底，
    # 防止极端情况下 payload 仍过大撑爆生成 prompt。
    rows_json = json.dumps(rows, ensure_ascii=False, default=str)
    if len(rows_json) > _METADATA_ROWS_MAX_CHARS:
        rows_json = rows_json[:_METADATA_ROWS_MAX_CHARS] + "…（结果过多，已截断）"

    try:
        chain = _METADATA_PROMPT | get_chat_model()
        resp = chain.invoke(
            {
                "query": query,
                "row_count": row_count,
                "rows": rows_json,
            }
        )
        return {"answer": resp.content, "citations": []}
    except Exception as exc:
        return {"answer": f"（回答生成失败：{exc}）", "citations": []}


def generate_answer(state: QAState) -> QAState:
    if state.get("error"):
        return {}

    query = state["query"]

    # 1. 闲聊
    if state.get("intent") == "chitchat":
        return _generate_chitchat(query)

    # 1b. 元数据问答：基于 SQL 结果作答
    if state.get("intent") == "metadata_qa":
        return _generate_metadata(state)

    # 2/3. 检索问答：优先用压缩后的命中，回退到原始命中
    hits = state.get("compressed_hits") or state.get("hits") or []
    context, citations = _build_context(hits)

    # 完全无命中：直接说明无法回答，不调用 LLM
    if not context:
        return {
            "answer": "抱歉，知识库中没有检索到与该问题相关的资料，暂时无法回答。",
            "citations": [],
        }

    enough = state.get("has_enough_context", False)
    insufficient_note = (
        ""
        if enough
        else "注意：以下上下文可能不足以完整回答该问题，请只依据已有信息作答，"
        "对缺失部分如实说明，不要编造。\n\n"
    )

    try:
        chain = _QA_PROMPT | get_chat_model()
        resp = chain.invoke(
            {
                "query": query,
                "context": context,
                "insufficient_note": insufficient_note,
            }
        )
        return {"answer": resp.content, "citations": citations}
    except Exception as exc:
        return {
            "answer": f"（回答生成失败：{exc}）",
            "citations": citations,
        }
