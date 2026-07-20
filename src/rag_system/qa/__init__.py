from rag_system.qa.state import QAState

__all__ = ["QAState", "build_qa_graph"]


def build_qa_graph():
    # 延迟导入，避免在只用到 state 时就加载 langgraph / 各节点
    from rag_system.qa.graph import build_qa_graph as _build

    return _build()
