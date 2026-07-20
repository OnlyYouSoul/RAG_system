# RAG_System

一个面向文档的 RAG（检索增强生成）系统：把 PDF 解析成 Markdown，抽取并托管图片、生成视觉描述，切分成 chunk、向量化写入 Milvus（入库流水线），并基于已入库知识跑 LangGraph 问答图（检索问答 / 元数据问答 / 闲聊）。

## 流程概览

```
入库：PDF ──MinerU──> Markdown ──> 图片上传对象存储 + 视觉描述 ──> 切块 ──> 向量化 ──> Milvus
                                                          └──> 文档/chunk 元数据 ──> Postgres / SQLite
问答：query ──> analyze(意图/过滤) ──> rewrite ──> 混合检索 ──> rerank ──> 压缩 ──> 生成(带引用)
                        └── metadata_qa ──> 生成 SQL ──> 执行 ──> 生成
```

1. **解析**：调用 [MinerU](../MinerU) 源码把 PDF 解析为 Markdown（`pipeline` 后端，CPU 可用）。
2. **改写**（`ingest/pipeline.py`）：把本地图片上传到对象存储（MinIO / AIStor，S3 兼容），替换成公开 URL，并可选调用视觉模型为每张图生成中文说明。
3. **切块**（`ingest/chunking.py`）：按段落切分，超长段落再按句子切，相邻 chunk 带 overlap。
4. **元数据**（`ingest/metadata.py`）：基于文件内容 SHA256 生成 `document_id`，附加 `kb_id`、`title`、`department` 等。
5. **向量化**（`ingest/embedding.py`）：OpenAI 兼容接口调嵌入模型（默认 `BAAI/bge-m3`，1024 维）。
6. **入库**（`store/milvus.py`、`store/postgres_store.py`、`store/sqlite_store.py`）：向量写 Milvus（`HNSW` + `COSINE`，含 BM25 混合检索），元数据写关系库。
7. **问答**（`qa/`）：LangGraph 图，见 [docs/QA_PIPELINE_STATUS.md](./docs/QA_PIPELINE_STATUS.md)。

## 目录结构

```
RAG_System/
├── src/rag_system/          # 后端逻辑（src-layout，安装为 rag_system 包）
│   ├── cli.py               # 统一入口：ingest / query / recreate
│   ├── config.py            # 路径常量（PROJECT_ROOT / OUTPUT_DIR / …）与 .env 加载
│   ├── ingest/              # 入库期：pipeline(解析改写) / chunking / embedding / metadata
│   ├── store/               # 存储层：milvus / postgres_store / sqlite_store / pg_query / redis_client
│   ├── rerank.py            # 检索期交叉编码器精排客户端
│   └── qa/                  # LangGraph 问答图：graph / state / llm / nodes/
├── tests/                   # test_nodes_smoke.py（冒烟）/ verify_hybrid.py（真实链路）
├── data/                    # 输入数据（不入库）：pdfs/、sqlite/rag_metadata.db
├── output/                  # 入库产物（不入库，见 output/README.md）
├── docs/                    # ARCHITECTURE / DEPLOY / QA_PIPELINE_STATUS
├── start.py                 # 兼容 shim（等价 `rag ...`）
├── pyproject.toml           # 依赖 + 构建（uv / hatchling，Python 3.12）
├── docker-compose.yml       # 整套栈编排
└── .env                     # 运行配置（不入库）
```

架构与模块职责详见 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)。

## 环境变量（`.env`）

| 变量 | 说明 |
| --- | --- |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 对象存储地址与凭据 |
| `MINIO_BUCKET` | 图片 bucket |
| `MINIO_PUBLIC_BASE_URL` | 图片对外可访问的基础 URL |
| `VISION_BASE_URL` / `VISION_API_KEY` / `VISION_MODEL` | 视觉模型（图片描述）|
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 嵌入模型 |
| `CHAT_BASE_URL` / `CHAT_API_KEY` / `CHAT_MODEL` | 问答/分类/改写/压缩用 chat 模型 |
| `RERANK_BASE_URL` / `RERANK_API_KEY` / `RERANK_MODEL` | 交叉编码器精排（未配置则透传）|
| `MILVUS_URI` / `MILVUS_TOKEN` / `MILVUS_DB` / `MILVUS_COLLECTION` | Milvus 连接 |
| `POSTGRES_DSN` 或 `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | 关系库连接 |
| `REDIS_URL` | Redis 连接 |

## 本地运行

```bash
uv sync                         # 安装依赖（含 ../MinerU 可编辑依赖）与本项目包

# 入库
uv run rag ingest \
    --pdf data/pdfs/compile_env.pdf \
    --kb-id kb_demo \
    --department infra \
    --to-milvus --to-postgres

# 问答（单次 / 交互式）
uv run rag query "编译环境怎么配置" --kb-id kb_demo
uv run rag query                       # 进入交互式多轮

# 重建 Milvus schema（会清空旧数据）
uv run rag recreate
```

`uv run rag` 与 `uv run python start.py` 等价。常用参数：`--skip-parse` 复用已解析 Markdown、`--no-vision` 跳过图片描述、`--chunk-size` / `--overlap` 控制切块、`--recreate` 重建 schema。

## 测试

```bash
uv run python tests/test_nodes_smoke.py   # 冒烟测试，不依赖外部服务
uv run python tests/verify_hybrid.py      # 真实链路验证（需 Milvus + embedding/chat/rerank key）
```

## Docker 部署

见 [docs/DEPLOY.md](./docs/DEPLOY.md)。整套栈由 `docker-compose.yml` 编排，包含 `milvus-standalone`、`aistor-server`（对象存储）、`etcd`、`redis`、`postgres` 以及本项目 `app`。
