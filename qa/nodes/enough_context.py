"""上下文充分性判断：检索结果是否足以支撑回答。"""

from __future__ import annotations

import os

from qa.state import QAState

# 命中相似度阈值（COSINE，0~1，越大越相似）
_SIM_THRESHOLD = float(os.getenv("QA_CONTEXT_SIM_THRESHOLD", "0.35"))
# 达到阈值的命中至少要有几条
_MIN_STRONG_HITS = int(os.getenv("QA_CONTEXT_MIN_HITS", "1"))


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

    strong = [h for h in hits if float(h.get("distance", 0.0)) >= _SIM_THRESHOLD]
    enough = len(strong) >= _MIN_STRONG_HITS

    if enough:
        best = max(float(h.get("distance", 0.0)) for h in hits)
        reason = (
            f"命中 {len(strong)} 条相似度≥{_SIM_THRESHOLD}"
            f"（最高 {best:.3f}），上下文充分"
        )
    else:
        best = max((float(h.get("distance", 0.0)) for h in hits), default=0.0)
        reason = (
            f"最高相似度 {best:.3f} < 阈值 {_SIM_THRESHOLD}，"
            f"或强命中不足 {_MIN_STRONG_HITS} 条，上下文不足"
        )

    return {
        "has_enough_context": enough,
        "context_reason": reason,
    }
