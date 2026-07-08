"""检索节点：查询向量化 + Milvus 向量检索（带元数据过滤）。

依赖 analyze_query 产出的 milvus_expr 做标量过滤，top_k 来自 receive_query。
检索结果写入 state["hits"]，每条含 distance（COSINE 相似度，越大越近）与实体字段。
"""

from __future__ import annotations

from utils.embedding import embed_query
from utils.milvus import search

from qa.state import QAState

# 返回给下游的实体字段
_OUTPUT_FIELDS = [
    "text",
    "chunk_id",
    "document_id",
    "kb_id",
    "title",
    "doc_type",
    "department",
    "chunk_index",
]


def retrieve(state: QAState) -> QAState:
    if state.get("error"):
        return {}

    # 优先用改写后的独立检索问题（rewrite_query 产出），否则用原查询
    query = state.get("search_query") or state["query"]
    top_k = state.get("top_k", 5)
    milvus_expr = state.get("milvus_expr", "") or ""

    try:
        query_vector = embed_query(query)
        hits = search(
            query_vector=query_vector,
            top_k=top_k,
            filter=milvus_expr,
            output_fields=_OUTPUT_FIELDS,
        )
    except Exception as exc:  # 向量化或检索失败
        return {
            "query_vector": [],
            "hits": [],
            "error": f"检索失败：{exc}",
        }

    return {
        "query_vector": query_vector,
        "hits": hits,
    }
