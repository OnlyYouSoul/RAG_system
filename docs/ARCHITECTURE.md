# 架构说明

> 后端逻辑统一收敛在 `src/rag_system/` 包（src-layout）。本文档记录模块职责、
> 数据流与依赖方向。问答图各节点的细节见 [QA_PIPELINE_STATUS.md](./QA_PIPELINE_STATUS.md)。

## 分层与模块职责

| 层 | 模块 | 职责 |
|---|---|---|
| 入口 | `cli.py` | argparse 子命令 `ingest` / `query` / `recreate`；组装入库流水线、驱动 QA 图 |
| 配置 | `config.py` | `PROJECT_ROOT` / `DATA_DIR` / `OUTPUT_DIR` / `SQLITE_DB_PATH` / `MINERU_SOURCE_DIR` 等路径常量与 `load_env()`，收敛原先散落各处的 `PROJECT_DIR` |
| 入库 `ingest/` | `pipeline.py` | MinerU 解析 PDF + 图片上传对象存储 + 视觉描述改写 |
| | `chunking.py` | 段落 / 句子切分，overlap |
| | `embedding.py` | OpenAI 兼容嵌入（bge-m3，1024 维），`embed_texts` / `embed_query` |
| | `metadata.py` | 文档级元数据、文件内容 hash → `document_id` |
| 存储 `store/` | `milvus.py` | Milvus schema（dense + BM25 sparse + 标量索引）、`search` / `hybrid_search` / `insert_chunks` |
| | `postgres_store.py` | 文档级/chunk 级元数据落 Postgres（正式方案，同 `document_id` 覆盖） |
| | `sqlite_store.py` | 同上，SQLite 离线备选，库落 `data/sqlite/rag_metadata.db` |
| | `pg_query.py` | 只读 SELECT 校验与执行（供 metadata_qa 分支的 SQL 执行） |
| | `redis_client.py` | 异步任务队列 / 文件锁骨架（尚未接入服务） |
| 检索 | `rerank.py` | 交叉编码器精排 HTTP 客户端（未配置则透传） |
| 问答 `qa/` | `graph.py` `state.py` `llm.py` `nodes/` | LangGraph 图组装、状态定义、chat 模型工厂、各节点 |

## 依赖方向

```
cli ──> ingest ──> store        cli ──> qa ──> {ingest.embedding, store.milvus, store.pg_query, rerank}
  └────> qa                     所有模块 ──> config（仅路径/环境，无反向依赖）
```

- `config` 是叶子，任何模块都可依赖它，它不依赖别人。
- `qa` 检索节点消费 `ingest.embedding`（向量化查询）、`store.milvus`（混合检索）、`rerank`；metadata_qa 分支消费 `store.pg_query`。
- `ingest` 与 `store` 之间：入库流水线（cli）把 chunk 交给 `store` 落库，`ingest` 本身不依赖 `store`。

## 数据流

```
入库：data/pdfs/<doc>.pdf
   └─ ingest.pipeline ─> output/<doc>/auto/<doc>_minio_vision.md
      └─ ingest.chunking + metadata ─> output/<doc>/<doc>_chunks.json + _document.json
         ├─ ingest.embedding ─> store.milvus（向量 + BM25）
         └─ store.postgres_store / sqlite_store（关系库元数据）

问答：query
   └─ qa.graph：receive ─> analyze ─┬─ retrieval_qa: rewrite ─> retrieve(混合) ─> rerank ─> enough ─> compress ─> generate
                                     ├─ metadata_qa:  generate_sql ─> execute_sql(pg) ─> generate
                                     └─ chitchat:     generate
```

## 产物与数据位置

- 输入 PDF：`data/pdfs/`（不入 git）
- 本地 SQLite 库：`data/sqlite/rag_metadata.db`（不入 git）
- 入库产物（Markdown / chunks / 图片）：`output/<doc>/`（不入 git，结构见 `output/README.md`）
