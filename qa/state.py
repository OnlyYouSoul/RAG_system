from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict

# 意图类别
Intent = Literal["retrieval_qa", "chitchat"]


class QueryFilters(TypedDict, total=False):

    kb_id: str
    department: str
    doc_type: str
    language: str
    document_id: str


class UserContext(TypedDict, total=False):
    """调用方带入的用户身份与权限信息。"""

    user_id: str
    department: str
    allowed_kb_ids: list[str]


class ChatTurn(TypedDict, total=False):
    role: str        # user / assistant
    content: str


class QAState(TypedDict, total=False):
    """贯穿整张 QA 图的状态。

    total=False：节点只需返回自己负责的字段，其余保持不变。
    """

    query: str
    raw_query: str                  # 未处理的原始输入
    top_k: int                      # 检索条数
    request_filters: QueryFilters
    chat_history: list[ChatTurn]    # 历史对话，可选
    user_context: UserContext       # 用户身份与权限，可选

    intent: Intent
    intent_confidence: float

    # analyze_query（意图分类 + 过滤构建合并节点）产出
    filters: QueryFilters
    milvus_expr: str                # 编译成 Milvus boolean expression 的过滤串
    analysis_payload: dict[str, Any]  # 发送给 LLM 的结构化 JSON（便于调试/审计）

    # rewrite_query 产出：结合历史改写成的独立检索问题
    search_query: str

    query_vector: list[float]
    hits: list[dict[str, Any]]

    # enough_context 产出
    has_enough_context: bool
    context_reason: str             # 判断依据，便于调试/审计

    # compress_context 产出
    compressed_hits: list[dict[str, Any]]  # 抽取压缩后的命中（保留 compressed_text）
    context: str                    # 拼接好、可直接喂给生成节点的上下文

    # generate_answer 产出
    answer: str
    citations: list[dict[str, Any]]  # 引用来源（document_id / title / chunk_id）

    error: Optional[str]
