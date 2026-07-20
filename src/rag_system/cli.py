"""RAG_System 统一入口。

子命令：
- ingest    PDF 解析 + Markdown 改写 + chunking(+ 可选写入 Milvus)
- query     基于已入库知识跑 QA 图（单次问答或交互式多轮）
- recreate  按最新 schema drop 并重建 Milvus collection（会清空旧数据）

示例：
    rag ingest --pdf data/pdfs/EOS.pdf --kb-id demo --to-milvus
    rag ingest --pdf data/pdfs/EOS.pdf --kb-id demo --to-milvus --recreate
    rag query "编译环境怎么配置" --kb-id demo
    rag query                       # 进入交互式多轮问答
    rag recreate
"""

import argparse
import json
import os
from pathlib import Path

from rag_system import config
from rag_system.ingest.pipeline import (
    find_markdown_file,
    run_mineru,
    process_markdown_images,
    init_minio_client,
    ensure_bucket,
)
from rag_system.ingest.chunking import chunk_text_by_paragraph
from rag_system.ingest.metadata import build_document_metadata, chunk_doc_fields
from rag_system.ingest import embedding
from rag_system.store import milvus, sqlite_store, postgres_store


def _run_ingest(args: argparse.Namespace) -> None:
    """PDF -> Markdown 改写 -> chunking -> (可选) 写入 Milvus。"""
    pdf_path = Path(args.pdf).resolve()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解析 PDF
    if not args.skip_parse:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")
        run_mineru(pdf_path, output_dir, backend=args.backend, lang=args.lang)

    # 2. 找到 MinerU 导出的原始 Markdown
    raw_md_file = find_markdown_file(output_dir)
    print(f"[Markdown] MinerU 原始 Markdown：{raw_md_file}")

    # 3. 改写 Markdown：图片上传 MinIO + 视觉描述，得到 *_minio_vision.md
    bucket = os.getenv("MINIO_BUCKET", "mineru-images")
    minio_client = init_minio_client()
    ensure_bucket(minio_client, bucket)

    md_file = process_markdown_images(
        md_file=raw_md_file,
        minio_client=minio_client,
        bucket=bucket,
        object_prefix=pdf_path.stem,
        enable_vision=not args.no_vision,
    )

    # 文档级 metadata
    lang_map = {"ch": "zh"}
    doc_metadata = build_document_metadata(
        pdf_path,
        kb_id=args.kb_id,
        department=args.department,
        language=lang_map.get(args.lang, args.lang),
        doc_type="pdf",
        title=args.title,
    )

    # chunking
    chunks = chunk_text_by_paragraph(
        str(md_file),
        chunk_size=args.chunk_size,
        overlap_size=args.overlap,
        doc_metadata=chunk_doc_fields(doc_metadata),
    )
    print(f"[Chunk] 共生成 {len(chunks)} 个 chunk")

    doc_metadata["total_chunks"] = len(chunks)

    # 保存 chunking 结果为 JSON 文件
    records = [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    doc_dir = output_dir / pdf_path.stem
    doc_dir.mkdir(parents=True, exist_ok=True)

    doc_out = doc_dir / f"{pdf_path.stem}_document.json"
    doc_out.write_text(json.dumps(doc_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Document] 文档档案 -> {doc_out}")

    chunks_out = doc_dir / f"{pdf_path.stem}_chunks.json"
    chunks_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done -> {chunks_out}")

    # 3.5 (可选) 保存文档级 + chunk 级元数据到关系库（覆盖式）
    # Postgres 是正式方案，SQLite 为临时/离线备选。
    if args.to_postgres:
        result = postgres_store.save_ingest_result(doc_metadata, chunks)
        print(
            f"[Postgres] 已写入 {result['chunks_written']} 个 chunk "
            f"(document_id={result['document_id']})"
        )
    if args.to_sqlite:
        result = sqlite_store.save_ingest_result(doc_metadata, chunks)
        print(
            f"[SQLite] 已写入 {result['chunks_written']} 个 chunk "
            f"(document_id={result['document_id']}) -> {result['db_path']}"
        )

    # 4. 向量化 + 写入 Milvus
    if args.to_milvus:
        if args.recreate:
            milvus.recreate_collection()
            print(f"[Milvus] 已按最新 schema 重建 collection：{milvus.MILVUS_COLLECTION}")
        vectors = embedding.embed_texts([chunk.text for chunk in chunks])
        print(f"[Embedding] 向量化完成：{len(vectors)} 条 x {embedding.EMBEDDING_DIM} 维")
        milvus.insert_chunks(chunks, vectors)
        milvus.get_client().flush(milvus.MILVUS_COLLECTION)
        print(f"[Milvus] 已写入 collection：{milvus.MILVUS_COLLECTION}")


def _build_request_filters(args: argparse.Namespace) -> dict:
    """从命令行参数收集显式过滤条件（None 的不带）。"""
    filters = {}
    for key in ("kb_id", "department", "doc_type", "document_id"):
        value = getattr(args, key, None)
        if value:
            filters[key] = value
    return filters


def _print_answer(out: dict) -> None:
    """打印 QA 图输出：答案 + 引用清单。"""
    if out.get("error"):
        print(f"[error] {out['error']}")
        return
    print(f"\n{out.get('answer', '')}")
    citations = out.get("citations", []) or []
    if citations:
        print("\n引用：")
        for c in citations:
            print(f"  [{c.get('n')}] {c.get('title')} (chunk_id={c.get('chunk_id')})")


def _run_query(args: argparse.Namespace) -> None:
    """跑 QA 图。带 question 则单次问答；否则进入交互式多轮。"""
    from rag_system.qa import build_qa_graph

    app = build_qa_graph()
    request_filters = _build_request_filters(args)

    base_state: dict = {"top_k": args.top_k}
    if request_filters:
        base_state["request_filters"] = request_filters

    # 单次问答
    if args.question:
        state = {**base_state, "query": args.question}
        _print_answer(app.invoke(state))
        return

    # 交互式多轮：把每轮的 query/answer 回写 chat_history，供改写/分析消费
    print("交互式问答（输入 exit / quit 退出）")
    chat_history: list[dict] = []
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        state = {**base_state, "query": question, "chat_history": list(chat_history)}
        out = app.invoke(state)
        _print_answer(out)

        answer = out.get("answer")
        if answer:
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})


def _run_recreate(args: argparse.Namespace) -> None:
    """按最新 schema drop 并重建 collection（清空旧数据）。"""
    milvus.recreate_collection()
    print(f"[Milvus] 已按最新 schema 重建 collection：{milvus.MILVUS_COLLECTION}")


def _add_ingest_parser(subparsers) -> None:
    p = subparsers.add_parser("ingest", help="PDF 解析 + Markdown 改写 + chunking(+ 写入 Milvus)")
    p.add_argument("--pdf", default=str(config.PDF_DIR / "EOS.pdf"))
    p.add_argument("--out", default=str(config.OUTPUT_DIR))
    p.add_argument("--backend", default="pipeline")
    p.add_argument("--lang", default="ch")
    p.add_argument("--chunk-size", type=int, default=800)
    p.add_argument("--overlap", type=int, default=100)
    p.add_argument("--kb-id", required=True)
    p.add_argument("--department", default="unknown")
    p.add_argument("--title", default=None)
    p.add_argument("--skip-parse", action="store_true", help="跳过 PDF 解析，复用已有 Markdown")
    p.add_argument("--no-vision", action="store_true", help="只上传图片并替换链接，不调用视觉模型")
    p.add_argument("--to-milvus", action="store_true")
    p.add_argument(
        "--to-postgres",
        action="store_true",
        help="把文档级与 chunk 级元数据保存到 Postgres（同 document_id 覆盖，正式方案）",
    )
    p.add_argument(
        "--to-sqlite",
        action="store_true",
        help="把文档级与 chunk 级元数据保存到 data/sqlite/rag_metadata.db（临时/离线备选，同 document_id 覆盖）",
    )
    p.add_argument(
        "--recreate",
        action="store_true",
        help="写入前 drop 并按最新 schema 重建 collection（schema 变更后重入库用，会清空旧数据）",
    )
    p.set_defaults(func=_run_ingest)


def _add_query_parser(subparsers) -> None:
    p = subparsers.add_parser("query", help="基于已入库知识问答（单次或交互式多轮）")
    p.add_argument("question", nargs="?", default=None, help="问题；省略则进入交互式多轮问答")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--kb-id", default=None, help="限定知识库")
    p.add_argument("--department", default=None, help="限定部门")
    p.add_argument("--doc-type", dest="doc_type", default=None, help="限定文档类型")
    p.add_argument("--document-id", dest="document_id", default=None, help="限定文档")
    p.set_defaults(func=_run_query)


def _add_recreate_parser(subparsers) -> None:
    p = subparsers.add_parser("recreate", help="按最新 schema drop 并重建 Milvus collection（清空旧数据）")
    p.set_defaults(func=_run_recreate)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG_System 统一入口：ingest（入库）/ query（问答）/ recreate（重建 schema）"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_ingest_parser(subparsers)
    _add_query_parser(subparsers)
    _add_recreate_parser(subparsers)

    args = parser.parse_args()

    config.load_env()

    args.func(args)


if __name__ == "__main__":
    main()
