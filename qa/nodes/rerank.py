"""重排序节点：用交叉编码器对检索命中按与查询的相关性重新排序。

向量检索（bi-encoder）召回快但精度有限，rerank（cross-encoder）对
query 与每个候选片段联合打分，排序更准。重排后给每条命中写入
rerank_score，并按其降序，同时截断到 top_k。

未配置 RERANK_* 或调用失败时原样透传，不阻断流程。
"""

from __future__ import annotations

from utils.rerank import is_configured, rerank as _rerank_api

from qa.state import QAState


def rerank(state: QAState) -> QAState:
    if state.get("error"):
        return {}

    hits = state.get("hits", []) or []
    if not hits:
        return {}

    query = state.get("search_query") or state["query"]
    top_k = state.get("top_k", len(hits))

    # 未配置重排服务：保持向量检索顺序，仅按 top_k 截断
    if not is_configured():
        return {"hits": hits[:top_k]}

    documents = [str(h.get("text", "")) for h in hits]

    try:
        ranked = _rerank_api(query, documents, top_n=top_k)
    except Exception:  # 重排失败不阻断，退化为原顺序
        return {"hits": hits[:top_k]}

    if not ranked:
        return {"hits": hits[:top_k]}

    reordered = []
    for item in ranked:
        idx = item["index"]
        if 0 <= idx < len(hits):
            hit = dict(hits[idx])
            hit["rerank_score"] = item["score"]
            reordered.append(hit)

    return {"hits": reordered[:top_k]}
