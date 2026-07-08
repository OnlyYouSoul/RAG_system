"""组装 QA 查询图（LangGraph）。

当前流水线：
    receive_query -> analyze_query
        -> [rewrite_query -> retrieve -> rerank -> enough_context] -> END

- analyze_query 合并了意图分类与元数据过滤构建，一次 LLM 调用产出意图与过滤条件。
- rewrite_query 结合聊天历史把指代性问题改写成独立检索问题。
- retrieve 做查询向量化 + Milvus 检索。
- rerank 用交叉编码器对命中重排（未配置时透传）。
- enough_context 判断命中是否足以支撑回答。

分支：
- 校验失败（空查询等）：直接结束。
- 闲聊(chitchat)：跳过检索，直连结束（未来接 generate_answer）。
- 检索问答(retrieval_qa)：rewrite_query -> retrieve -> rerank -> enough_context。
  上下文充分 / 不足两条分支目前都到 END，后续分别接
  generate_answer 与澄清/兜底节点。
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
)


def _route_after_receive(state: QAState) -> str:
    """receive_query 校验失败则直接结束。"""
    if state.get("error"):
        return "end"
    return "continue"


def _route_after_analyze(state: QAState) -> str:
    """闲聊跳过检索；出错直接结束；否则进入检索流水线。"""
    if state.get("error"):
        return "end"
    if state.get("intent") == "chitchat":
        return "chitchat"
    return "retrieve"


def _route_after_enough(state: QAState) -> str:
    """上下文充分与否的分支（下游生成节点接入前，两支都到 END）。"""
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

    graph.add_edge(START, "receive_query")

    # 空查询等校验失败：跳过后续，直接结束
    graph.add_conditional_edges(
        "receive_query",
        _route_after_receive,
        {"continue": "analyze_query", "end": END},
    )

    # 闲聊跳过检索
    graph.add_conditional_edges(
        "analyze_query",
        _route_after_analyze,
        {"retrieve": "rewrite_query", "chitchat": END, "end": END},
    )

    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "enough_context")

    # 上下文充分 -> 生成回答（待接入）；不足 -> 澄清/兜底（待接入）
    graph.add_conditional_edges(
        "enough_context",
        _route_after_enough,
        {"enough": END, "insufficient": END},
    )

    return graph.compile()
