"""端到端验证：混合检索（dense + BM25）真实链路。

从已解析的 chunks JSON 入库（自动建带 BM25 的新 schema），
再对比 纯 dense vs 混合检索 的召回，确认 BM25 那一路生效。
运行：python tests/verify_hybrid.py
"""

from __future__ import annotations

import json

from rag_system import config
from rag_system.ingest import embedding
from rag_system.store import milvus
from rag_system.ingest.chunking import Chunk

config.load_env()

_CHUNKS_JSON = str(config.OUTPUT_DIR / "compile_env" / "compile_env_chunks.json")


def _load_chunks() -> list[Chunk]:
    recs = json.load(open(_CHUNKS_JSON, encoding="utf-8"))
    return [Chunk(chunk_id=r["chunk_id"], text=r["text"], metadata=r["metadata"]) for r in recs]


def _ingest_if_needed() -> None:
    client = milvus.get_client()
    if client.has_collection(milvus.MILVUS_COLLECTION):
        stats = client.get_collection_stats(milvus.MILVUS_COLLECTION)
        if int(stats.get("row_count", 0)) > 0:
            print(f"[skip] collection 已有 {stats['row_count']} 条，跳过入库")
            return
    chunks = _load_chunks()
    print(f"[embed] 向量化 {len(chunks)} 条 ...")
    vectors = embedding.embed_texts([c.text for c in chunks])
    milvus.insert_chunks(chunks, vectors)
    client.flush(milvus.MILVUS_COLLECTION)
    print(f"[milvus] 已写入，row_count={client.get_collection_stats(milvus.MILVUS_COLLECTION)['row_count']}")


def _short(hit: dict) -> str:
    return f"#{hit.get('chunk_id')} {str(hit.get('text',''))[:40].replace(chr(10),' ')}"


def main() -> None:
    milvus.ensure_collection()
    _ingest_if_needed()

    query = "编译环境怎么配置"
    qv = embedding.embed_query(query)

    print(f"\n=== query: {query} ===")

    dense = milvus.search(qv, top_k=5, output_fields=["text", "chunk_id"])
    print("\n[纯 dense]")
    for h in dense:
        print(f"  {h['distance']:.4f}  {_short(h)}")

    hybrid = milvus.hybrid_search(qv, query, top_k=5, output_fields=["text", "chunk_id"])
    print("\n[混合 dense+BM25 (RRF)]")
    for h in hybrid:
        print(f"  {h['distance']:.4f}  {_short(h)}")

    dense_ids = {h["chunk_id"] for h in dense}
    hybrid_ids = {h["chunk_id"] for h in hybrid}
    only_hybrid = hybrid_ids - dense_ids
    print(f"\n混合检索新增（BM25 贡献）的 chunk_id: {sorted(only_hybrid) or '无（两路高度重合）'}")

    # 上下文压缩（真实 LLM 抽取式）
    from rag_system.qa.nodes.compress_context import compress_context

    state = {"query": query, "search_query": query, "hits": hybrid}
    out = compress_context(state)
    raw_len = sum(len(str(h.get("text", ""))) for h in hybrid)
    comp_len = len(out["context"])
    print("\n[上下文压缩 LLM 抽取式]")
    print(f"  压缩前原文总长: {raw_len} 字  ->  压缩后: {comp_len} 字  "
          f"（保留 {comp_len / raw_len:.0%}，{len(out['compressed_hits'])}/{len(hybrid)} 段有相关内容）")
    for h in out["compressed_hits"][:3]:
        print(f"  #{h.get('chunk_id')}: {h['compressed_text'][:80].replace(chr(10),' ')}")

    # 全链路：真实图跑一遍检索问答，看最终生成的答案 + 引用
    from rag_system.qa import build_qa_graph

    app = build_qa_graph()
    print("\n[全链路 QA]")
    final = app.invoke({"query": "编译环境需要用什么操作系统和编译工具？"})
    print("  intent:", final.get("intent"), "| 充分:", final.get("has_enough_context"))
    print("  answer:", str(final.get("answer", "")).replace("\n", " "))
    cites = final.get("citations", [])
    print("  citations:", [(c["n"], c.get("title"), c.get("chunk_id")) for c in cites])

    print("\n验证完成 ✅")


if __name__ == "__main__":
    main()
