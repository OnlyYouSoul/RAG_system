"""组装 QA 查询图（LangGraph）。

分支：
- 校验失败（空查询等）：直接结束。
- 闲聊(chitchat)：跳过检索，直连 generate_answer。
- 检索问答(retrieval_qa)：rewrite_query -> retrieve -> rerank -> enough_context。
  上下文充分 -> compress_context -> generate_answer；
  不足 -> generate_answer（基于弱上下文兜底，提示资料不足）。
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from qa.state import QAState
from qa.nodes import (
    receive_query,
    analyze_query,
    rewrite_query,
    retrieve,
    rerank,
    enough_context,
    compress_context,
    generate_answer,
)


def _route_after_receive(state: QAState) -> str:
    """receive_query 校验失败则直接结束。"""
    if state.get("error"):
        return "end"
    return "continue"


def _route_after_analyze(state: QAState) -> str:
    """闲聊直连生成；出错直接结束；否则进入检索流水线。"""
    if state.get("error"):
        return "end"
    if state.get("intent") == "chitchat":
        return "chitchat"
    return "retrieve"


def _route_after_enough(state: QAState) -> str:
    if state.get("error"):
        return "insufficient"
    return "enough" if state.get("has_enough_context") else "insufficient"


def build_qa_graph():
    graph = StateGraph(QAState)

    graph.add_node("receive_query", receive_query)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rerank", rerank)
    graph.add_node("enough_context", enough_context)
    graph.add_node("compress_context", compress_context)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "receive_query")

    # 空查询等校验失败：跳过后续，直接结束
    graph.add_conditional_edges(
        "receive_query",
        _route_after_receive,
        {"continue": "analyze_query", "end": END},
    )

    # 闲聊直连生成；检索问答进入检索流水线
    graph.add_conditional_edges(
        "analyze_query",
        _route_after_analyze,
        {"retrieve": "rewrite_query", "chitchat": "generate_answer", "end": END},
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "enough_context")

    # 上下文充分 -> 压缩上下文再生成；不足 -> 直接兜底生成
    graph.add_conditional_edges(
        "enough_context",
        _route_after_enough,
        {"enough": "compress_context", "insufficient": "generate_answer"},
    )

    graph.add_edge("compress_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
