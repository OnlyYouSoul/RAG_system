import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from rewrite import (
    PROJECT_DIR,
    find_markdown_file,
    run_mineru,
    process_markdown_images,
    init_minio_client,
    ensure_bucket,
)
from utils.chunking import chunk_text_by_paragraph
from utils.metadata import build_document_metadata, chunk_doc_fields
from utils import embedding, milvus


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF 解析 + Markdown 改写 + chunking")
    parser.add_argument("--pdf", default=str(PROJECT_DIR / "EOS.pdf"))
    parser.add_argument("--out", default=str(PROJECT_DIR / "mineru_output"))
    parser.add_argument("--backend", default="pipeline")
    parser.add_argument("--lang", default="ch")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--kb-id", required=True)
    parser.add_argument("--department", default="unknown")
    parser.add_argument("--title", default=None)
    parser.add_argument("--skip-parse", action="store_true", help="跳过 PDF 解析，复用已有 Markdown")
    parser.add_argument("--no-vision", action="store_true", help="只上传图片并替换链接，不调用视觉模型")
    parser.add_argument("--to-milvus", action="store_true")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="写入前 drop 并按最新 schema 重建 collection（schema 变更后重入库用，会清空旧数据）",
    )
    args = parser.parse_args()

    load_dotenv(PROJECT_DIR / ".env")

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

    # 4. 向量化 + 写入 Milvus
    if args.to_milvus:
        if args.recreate:
            milvus.recreate_collection()
            print(f"[Milvus] 已重建 collection：{milvus.MILVUS_COLLECTION}")
        vectors = embedding.embed_texts([chunk.text for chunk in chunks])
        print(f"[Embedding] 向量化完成：{len(vectors)} 条 x {embedding.EMBEDDING_DIM} 维")
        milvus.insert_chunks(chunks, vectors)
        milvus.get_client().flush(milvus.MILVUS_COLLECTION)
        print(f"[Milvus] 已写入 collection：{milvus.MILVUS_COLLECTION}")


if __name__ == "__main__":
    main()