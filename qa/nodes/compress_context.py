"""上下文压缩：对 rerank 后的每个 chunk 做 LLM 抽取式压缩。

只保留与问题相关的原文句子/片段（逐字抽取，不改写、不臆造），丢弃无关内容，
显著缩短喂给生成节点的上下文、降低噪声与 token 成本。

实现：每个 chunk 独立走一次抽取，用 Runnable.batch 并发执行——既保证
每 chunk 的抽取质量，又避免串行等待与「大 prompt 后段 chunk 质量打折」。
抽取结果为空的 chunk 直接丢弃；LLM 不可用时回退为原文，不阻断流程。
"""

from __future__ import annotations

import os

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from qa.llm import get_chat_model
from qa.state import QAState

# 并发上限
_MAX_CONCURRENCY = 5
# 拼接上下文时片段之间的分隔
_JOINER = "\n\n---\n\n"

# 短文本护栏：原文短于此长度直接跳过压缩（原样保留）。
# 抽取式压缩对短片段（如标题、单句）是负优化——LLM 往往整段照抄，
# 却额外引入延迟、token 成本与偶发的分隔符/改写噪声，得不偿失。
_MIN_COMPRESS_LENGTH = int(os.getenv("QA_COMPRESS_MIN_LENGTH", "200"))


class CompressedChunk(BaseModel):
    """单个 chunk 的抽取压缩结果。"""

    relevant_text: str = Field(
        description="从原文中逐字抽取的、与问题相关的句子或片段；"
        "拼接保持原文顺序。若整段都与问题无关，返回空字符串。"
    )


_SYSTEM_PROMPT = (
    "你负责从给定文档片段中，逐字抽取与用户问题相关的句子或片段。\n"
    "要求：\n"
    "1. 只做抽取，不要改写、概括或补充，原文怎么写就怎么摘。\n"
    "2. 保持原文顺序，去掉与问题无关的句子。\n"
    "3. 若整段都与问题无关，返回空字符串。"
)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "用户问题：{query}\n\n文档片段：\n{chunk}"),
    ]
)
   

def compress_context(state: QAState) -> QAState:
    if state.get("error"):
        return {}

    hits = state.get("hits", []) or []
    if not hits:
        return {"compressed_hits": [], "context": ""}

    query = state.get("search_query") or state["query"]

    # 短文本护栏：短片段跳过 LLM 压缩，原样保留；只把长片段送去抽取。
    # 保留原始顺序，压缩完再按 hits 顺序合并。
    long_indices = [
        i for i, h in enumerate(hits)
        if len(str(h.get("text", ""))) >= _MIN_COMPRESS_LENGTH
    ]

    # LLM 不可用：回退为原文，不阻断
    try:
        results_by_index: dict[int, str] = {}
        if long_indices:
            model = get_chat_model().with_structured_output(CompressedChunk)
            chain = _prompt | model
            inputs = [
                {"query": query, "chunk": str(hits[i].get("text", ""))}
                for i in long_indices
            ]
            batch = chain.batch(
                inputs, config={"max_concurrency": _MAX_CONCURRENCY}
            )
            for i, result in zip(long_indices, batch):
                results_by_index[i] = (
                    getattr(result, "relevant_text", "") or ""
                ).strip()
    except Exception:
        compressed_hits = [dict(h) for h in hits]
        for h in compressed_hits:
            h["compressed_text"] = str(h.get("text", ""))
        context = _JOINER.join(h["compressed_text"] for h in compressed_hits)
        return {"compressed_hits": compressed_hits, "context": context}

    compressed_hits: list[dict] = []
    for i, hit in enumerate(hits):
        if i in results_by_index:
            text = results_by_index[i]
            if not text:  # 整段无关，丢弃
                continue
        else:
            # 短片段护栏：原样保留，不判无关
            text = str(hit.get("text", ""))
        new_hit = dict(hit)
        new_hit["compressed_text"] = text
        compressed_hits.append(new_hit)

    # 全被判为无关时，退回原文（避免把上下文压没）
    if not compressed_hits:
        compressed_hits = [dict(h) for h in hits]
        for h in compressed_hits:
            h["compressed_text"] = str(h.get("text", ""))

    context = _JOINER.join(h["compressed_text"] for h in compressed_hits)
    return {"compressed_hits": compressed_hits, "context": context}
