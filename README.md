# RAG_System

一个面向文档的 RAG（检索增强生成）**数据入库流水线**：把 PDF 解析成 Markdown，抽取并托管图片、生成视觉描述，切分成 chunk，向量化后写入 Milvus，供后续检索使用。

## 流程概览

```
PDF ──MinerU──> Markdown ──> 图片上传对象存储 + 视觉描述 ──> 切块(chunk) ──> 向量化 ──> Milvus
```

1. **解析**：调用 [MinerU](../MinerU) 源码把 PDF 解析为 Markdown（`pipeline` 后端，CPU 可用）。
2. **改写**（`rewrite.py`）：把 Markdown 里的本地图片上传到对象存储（MinIO / AIStor，S3 兼容），替换成公开 URL，并可选调用视觉模型为每张图生成一句中文说明。
3. **切块**（`utils/chunking.py`）：按段落切分，超长段落再按句子切，相邻 chunk 之间带 overlap。
4. **元数据**（`utils/metadata.py`）：基于文件内容 SHA256 生成 `document_id`，附加 `kb_id`、`title`、`department` 等字段。
5. **向量化**（`utils/embedding.py`）：通过 OpenAI 兼容接口调用嵌入模型（默认 `BAAI/bge-m3`，1024 维）。
6. **入库**（`utils/milvus.py`）：写入 Milvus collection（`HNSW` + `COSINE`，过滤字段带标量索引）。

`src/redis_client.py` 是异步任务队列 / 文件锁的骨架，用于后续把入库做成异步任务，目前尚未接入服务。

## 目录结构

```
RAG_System/
├── start.py            # 入口：解析 + 改写 + 切块 + 向量化 + 入库
├── rewrite.py          # MinerU 解析 + 图片上传对象存储 + 视觉描述
├── utils/
│   ├── chunking.py     # 段落 / 句子切分，overlap
│   ├── embedding.py    # OpenAI 兼容嵌入
│   ├── metadata.py     # 文档级元数据、文件 hash
│   └── milvus.py       # Milvus schema / insert / search
├── src/redis_client.py # Redis 任务队列 / 锁（骨架）
├── pyproject.toml      # 依赖（uv 管理，Python 3.12）
└── .env                # 运行配置（不入库）
```

## 环境变量（`.env`）

| 变量 | 说明 |
| --- | --- |
| `MINIO_ENDPOINT` / `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 对象存储地址与凭据 |
| `MINIO_BUCKET` | 图片 bucket |
| `MINIO_PUBLIC_BASE_URL` | 图片对外可访问的基础 URL |
| `VISION_BASE_URL` / `VISION_API_KEY` / `VISION_MODEL` | 视觉模型（图片描述）|
| `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` / `EMBEDDING_DIM` | 嵌入模型 |
| `MILVUS_URI` / `MILVUS_TOKEN` / `MILVUS_DB` / `MILVUS_COLLECTION` | Milvus 连接 |
| `REDIS_URL` | Redis 连接 |

## 本地运行

```bash
uv sync                 # 安装依赖（含 ../MinerU 可编辑依赖）
uv run python start.py \
    --pdf compile_env.pdf \
    --kb-id kb_demo \
    --department infra \
    --to-milvus
```

常用参数：`--skip-parse` 复用已解析的 Markdown、`--no-vision` 跳过图片描述、`--chunk-size` / `--overlap` 控制切块。

## Docker 部署

见 [DEPLOY.md](./docs/DEPLOY.md)。整套栈由 `../docker-compose.yml` 编排，包含 `milvus-standalone`、`aistor-server`（对象存储）、`etcd`、`redis` 以及本项目 `app`。
