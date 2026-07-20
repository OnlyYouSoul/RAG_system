"""查询改写：结合聊天历史把口语化/指代性问题改写成独立检索问题。

无历史时直接沿用原查询；LLM 不可用时回退为原查询，不阻断流程。
产出写入 state["search_query"]，供 retrieve 优先使用。
"""

from __future__ import annotations

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_system.qa.llm import get_chat_model
from rag_system.qa.state import QAState

# 只取最近若干轮历史，避免 prompt 过长
_MAX_HISTORY_TURNS = 6


class RewrittenQuery(BaseModel):
    """改写结果。"""

    query: str = Field(description="改写后的独立检索问题，须自洽、不依赖上下文指代")


_SYSTEM_PROMPT = (
    "你负责把用户在多轮对话中的最新问题改写成一个自洽、独立的检索问题。\n"
    "要求：\n"
    "1. 消解指代（它/这个/那个/上面说的等），补全成明确的名词。\n"
    "2. 只依据历史补全必要信息，不要臆造历史中不存在的内容。\n"
    "3. 保持原问题的意图与语言，不要额外解释或作答。\n"
    "若原问题本身已经独立完整，原样返回即可。"
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "历史对话：\n{history}\n\n最新问题：{query}"),
    ]
)


def _format_history(history: list) -> str:
    lines = []
    for turn in history[-_MAX_HISTORY_TURNS:]:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rewrite_query(state: QAState) -> QAState:
    if state.get("error"):
        return {}

    query = state["query"]
    history = state.get("chat_history", []) or []

    # 无历史无需改写，直接沿用原查询
    history_text = _format_history(history)
    if not history_text:
        return {"search_query": query}

    try:
        model = get_chat_model().with_structured_output(RewrittenQuery)
        chain = _prompt | model
        result: RewrittenQuery = chain.invoke(
            {"history": history_text, "query": query}
        )
        rewritten = (result.query or "").strip()
        return {"search_query": rewritten or query}
    except Exception:  # 改写失败不阻断，回退原查询
        return {"search_query": query}
