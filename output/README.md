# output/

入库流水线（`rag ingest` / `python start.py ingest`）的产物目录。**内容不入 git**（见 `.gitignore`），仅保留本说明与 `.gitkeep`。

每个文档一个子目录，以 PDF 文件名（stem）命名：

```
output/
└── <doc>/
    ├── auto/                       # MinerU 原始解析产物
    │   ├── <doc>.md                # 原始 Markdown
    │   ├── <doc>_minio_vision.md   # 改写后：图片换成 MinIO URL + 视觉描述
    │   ├── <doc>_content_list.json
    │   ├── <doc>_middle.json
    │   ├── <doc>_model.json
    │   ├── <doc>_layout.pdf / _span.pdf / _origin.pdf
    │   └── images/                 # 抽取的图片
    ├── <doc>_document.json         # 文档级元数据（document_id / kb_id / total_chunks …）
    └── <doc>_chunks.json           # 切块结果（chunk_id / text / metadata）
```

- `*_chunks.json` 是向量化与入库 Milvus 的输入。
- `*_document.json` 对应关系库（Postgres/SQLite）的 `documents` 表一行。
