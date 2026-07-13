"""检索节点：混合检索（dense 向量 + BM25 全文），服务端 RRF 融合。

依赖 analyze_query 产出的 milvus_expr 做标量过滤，top_k 来自 receive_query。
- dense：查询向量化后做 COSINE 近邻。
- BM25：用查询文本做全文检索（服务端分词），补足关键词/专有名词召回。
两路由 Milvus 用 RRF 融合。结果写入 state["hits"]，每条含 distance
（此处为 RRF 融合分，越大越好）与实体字段。
"""

from __future__ import annotations

from utils.embedding import embed_query
from utils.milvus import hybrid_search

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
        hits = hybrid_search(
            query_vector=query_vector,
            query_text=query,
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
