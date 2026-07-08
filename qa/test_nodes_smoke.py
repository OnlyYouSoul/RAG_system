"""QA 图节点的冒烟测试（不依赖真实 LLM / embedding / Milvus）。

用 RunnableLambda 顶替结构化输出模型，patch 掉 embed_query / search，
验证节点数据流与图连接。
运行：python -m qa.test_nodes_smoke
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

# 直接拿到子模块（qa.nodes 里同名函数会遮蔽属性，用 import_module 取模块本身）
_aq_mod = importlib.import_module("qa.nodes.analyze_query")
_rw_mod = importlib.import_module("qa.nodes.rewrite_query")
_rt_mod = importlib.import_module("qa.nodes.retrieve")
_rr_mod = importlib.import_module("qa.nodes.rerank")
QueryAnalysis = _aq_mod.QueryAnalysis
RewrittenQuery = _rw_mod.RewrittenQuery


def _fake_analysis(intent: str, **filters):
    """按 schema 造一个 QueryAnalysis 结果。"""

    class _FakeModel:
        def with_structured_output(self, schema):
            assert schema is QueryAnalysis, f"未预期的 schema: {schema}"
            return RunnableLambda(
                lambda _: QueryAnalysis(intent=intent, confidence=0.92, **filters)
            )

    return _FakeModel()


def _fake_rewrite(rewritten: str):
    """顶替 rewrite_query 的结构化输出模型。"""

    class _FakeModel:
        def with_structured_output(self, schema):
            assert schema is RewrittenQuery, f"未预期的 schema: {schema}"
            return RunnableLambda(lambda _: RewrittenQuery(query=rewritten))

    return _FakeModel()


def _fake_hits(*distances):
    """按给定相似度分数造检索命中。"""
    return [
        {"distance": d, "text": f"chunk-{i}", "chunk_id": i, "title": "T"}
        for i, d in enumerate(distances)
    ]


def main() -> None:
    # 检索问答：显式 kb_id 应覆盖并叠加 LLM 抽取；命中强 -> 上下文充分
    # 带历史 -> rewrite_query 走 LLM；rerank 已配置 -> 逆序重排验证生效
    with patch.object(
        _aq_mod,
        "get_chat_model",
        return_value=_fake_analysis("retrieval_qa", department="infra", doc_type="PDF"),
    ), patch.object(
        _rw_mod, "get_chat_model", return_value=_fake_rewrite("编译环境怎么配置？")
    ), patch.object(_rt_mod, "embed_query", return_value=[0.1] * 4), patch.object(
        _rt_mod, "search", return_value=_fake_hits(0.82, 0.61, 0.20)
    ) as search_mock, patch.object(
        _rr_mod, "is_configured", return_value=True
    ), patch.object(
        # 逆序重排：把最后一条顶到最前，验证 hits 真的被重排 + 带上 rerank_score
        _rr_mod,
        "_rerank_api",
        side_effect=lambda q, docs, top_n=None: [
            {"index": i, "score": float(i)} for i in reversed(range(len(docs)))
        ],
    ):
        from qa import build_qa_graph

        app = build_qa_graph()

        out = app.invoke(
            {
                "query": "  infra部门的编译环境PDF怎么配置？  ",
                "request_filters": {"kb_id": "kb_demo"},
                "user_context": {"user_id": "u1", "department": "infra"},
                "chat_history": [{"role": "user", "content": "上一轮问题"}],
            }
        )
        assert out["query"] == "infra部门的编译环境PDF怎么配置？", out["query"]
        assert out["intent"] == "retrieval_qa", out["intent"]
        assert out["filters"] == {
            "kb_id": "kb_demo",
            "department": "infra",
            "doc_type": "pdf",
        }, out["filters"]
        assert 'department == "infra"' in out["milvus_expr"]
        assert 'kb_id == "kb_demo"' in out["milvus_expr"]
        assert out["top_k"] == 5

        # 结构化 payload 应带上任务约定的字段
        payload = out["analysis_payload"]
        assert payload["query"] == out["query"], payload
        assert payload["user_context"] == {"user_id": "u1", "department": "infra"}, payload
        assert payload["chat_history"] == [{"role": "user", "content": "上一轮问题"}], payload
        assert "filterable_metadata" in payload and "metadata_schema" in payload, payload
        assert "current_time" in payload, payload

        # rewrite_query：有历史 -> 用改写后的独立问题；retrieve 用它去 embed
        assert out["search_query"] == "编译环境怎么配置？", out.get("search_query")

        # retrieve：应带着编译好的过滤表达式与 top_k 调用 search
        _, kwargs = search_mock.call_args
        assert kwargs["filter"] == out["milvus_expr"], kwargs
        assert kwargs["top_k"] == 5, kwargs
        assert len(out["hits"]) == 3, out["hits"]

        # rerank：逆序重排生效 -> 原 chunk-2 排到最前，且带 rerank_score
        assert out["hits"][0]["chunk_id"] == 2, out["hits"]
        assert "rerank_score" in out["hits"][0], out["hits"][0]

        # enough_context：有强命中 -> 充分
        assert out["has_enough_context"] is True, out
        print("[ok] 检索问答:", out["intent"], "| 改写:", out["search_query"])

    # 弱命中 + 无历史（rewrite 直接透传）+ rerank 未配置（透传）-> 上下文不足
    with patch.object(
        _aq_mod, "get_chat_model", return_value=_fake_analysis("retrieval_qa")
    ), patch.object(_rt_mod, "embed_query", return_value=[0.1] * 4), patch.object(
        _rt_mod, "search", return_value=_fake_hits(0.10, 0.05)
    ), patch.object(_rr_mod, "is_configured", return_value=False):
        from qa import build_qa_graph

        app = build_qa_graph()
        out_weak = app.invoke({"query": "无关的冷门问题"})
        assert out_weak["intent"] == "retrieval_qa", out_weak
        # 无历史：search_query 直接等于原查询
        assert out_weak["search_query"] == "无关的冷门问题", out_weak
        # rerank 未配置：命中原样透传（无 rerank_score）
        assert "rerank_score" not in out_weak["hits"][0], out_weak["hits"][0]
        assert out_weak["has_enough_context"] is False, out_weak
        print("[ok] 弱命中不足 + rerank 透传:", out_weak["context_reason"])

    # 闲聊：无需过滤条件
    with patch.object(
        _aq_mod, "get_chat_model", return_value=_fake_analysis("chitchat")
    ):
        from qa import build_qa_graph

        app = build_qa_graph()
        out_chat = app.invoke({"query": "你好呀"})
        assert out_chat["intent"] == "chitchat", out_chat
        assert out_chat["filters"] == {}, out_chat
        assert out_chat["milvus_expr"] == "", out_chat
        print("[ok] 闲聊短路:", out_chat["intent"])

        # 空查询：校验失败，直接结束，不进入分析
        out2 = app.invoke({"query": "   "})
        assert out2.get("error"), out2
        assert "intent" not in out2, out2
        print("[ok] 空查询短路:", out2["error"])

        # top_k 上限裁剪
        out3 = app.invoke({"query": "你好呀", "top_k": 999})
        assert out3["top_k"] == 50, out3["top_k"]
        print("[ok] top_k 裁剪:", out3["top_k"])

    print("\n全部冒烟测试通过 ✅")


if __name__ == "__main__":
    main()
