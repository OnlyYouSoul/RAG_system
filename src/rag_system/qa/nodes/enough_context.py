"""上下文充分性判断：检索结果是否足以支撑回答。

评分优先级：
1. rerank_score（交叉编码器相关性，0~1，最可靠）——rerank 生效时用它，
   阈值 _RERANK_THRESHOLD。
2. distance（无 rerank 时的兜底）——现在 retrieve 走混合检索，distance 是
   RRF 融合分（量纲小，非 0~1 相似度），只用一个很低的地板值 _RRF_FLOOR
   判断“有实质命中”，避免误判为不足。

至少 _MIN_STRONG_HITS 条达到阈值才算“上下文充分”。阈值均可用环境变量调节。
"""

from __future__ import annotations

import os

from rag_system.qa.state import QAState

# 交叉编码器相关性阈值（rerank_score，0~1，越大越相关）
_RERANK_THRESHOLD = float(os.getenv("QA_CONTEXT_RERANK_THRESHOLD", "0.2"))
# 无 rerank 时对 RRF 融合分的地板值（远小于 COSINE，仅表示“有实质命中”）
_RRF_FLOOR = float(os.getenv("QA_CONTEXT_RRF_FLOOR", "0.01"))
# 达到阈值的命中至少要有几条
_MIN_STRONG_HITS = int(os.getenv("QA_CONTEXT_MIN_HITS", "1"))


def _score_and_threshold(hits: list[dict]) -> tuple[list[float], float, str]:
    """按是否已 rerank 选择打分口径与阈值。"""
    if any("rerank_score" in h for h in hits):
        scores = [float(h.get("rerank_score", 0.0)) for h in hits]
        return scores, _RERANK_THRESHOLD, "rerank_score"
    scores = [float(h.get("distance", 0.0)) for h in hits]
    return scores, _RRF_FLOOR, "RRF"


def enough_context(state: QAState) -> QAState:
    if state.get("error"):
        return {}

    # 闲聊无需检索，视为不需要额外上下文
    if state.get("intent") == "chitchat":
        return {
            "has_enough_context": False,
            "context_reason": "闲聊意图，无需检索上下文",
        }

    hits = state.get("hits", []) or []
    if not hits:
        return {
            "has_enough_context": False,
            "context_reason": "检索无命中",
        }

    scores, threshold, kind = _score_and_threshold(hits)
    strong = [s for s in scores if s >= threshold]
    enough = len(strong) >= _MIN_STRONG_HITS
    best = max(scores, default=0.0)

    if enough:
        reason = f"命中 {len(strong)} 条 {kind}≥{threshold}（最高 {best:.3f}），上下文充分"
    else:
        reason = (
            f"最高 {kind} {best:.3f} < 阈值 {threshold}，"
            f"或强命中不足 {_MIN_STRONG_HITS} 条，上下文不足"
        )

    return {
        "has_enough_context": enough,
        "context_reason": reason,
    }
