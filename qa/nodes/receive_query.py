from __future__ import annotations

from qa.state import QAState

_DEFAULT_TOP_K = 5
_MAX_TOP_K = 50


def receive_query(state: QAState) -> QAState:
    raw = state.get("query") or state.get("raw_query") or ""
    query = raw.strip()

    if not query:
        return {"error": "空查询：query 不能为空"}

    # 归一化 top_k
    top_k = state.get("top_k", _DEFAULT_TOP_K)
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        top_k = _DEFAULT_TOP_K
    top_k = max(1, min(top_k, _MAX_TOP_K))

    return {
        "raw_query": raw,
        "query": query,
        "top_k": top_k,
        # 显式传入的过滤条件透传给下游归并
        "request_filters": state.get("request_filters", {}) or {},
        "error": None,
    }
