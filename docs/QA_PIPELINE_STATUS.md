# QA Graph 进度文档

> 基于 LangGraph 的 RAG 问答流水线。本文档记录当前完成情况、各节点职责、
> 待办事项，以及已完成部分可以进一步优化的方向。
>
> 最后更新：2026-07-20

---

## 一、整体流水线

```
receive_query
  ├─ 校验失败（空查询等） ───────────────────────────────────► END
  └─ analyze_query（意图分类 + 元数据过滤构建，一次 LLM 调用）
       ├─ chitchat ─────────────────────────────────────────► generate_answer ─► END
       ├─ metadata_qa（文档元数据的计数/时间/罗列类问题）
       │    └─ generate_sql ─► execute_sql
       │                         ├─ 成功 ────────────────────► generate_answer ─► END
       │                         └─ 失败且未超上限 ─► generate_sql（带错重写，≤2 次）
       └─ retrieval_qa
            └─ rewrite_query ─► retrieve ─► rerank ─► enough_context
                                                          ├─ 充分 ─► compress_context ─► generate_answer ─► END
                                                          └─ 不足 ─────────────────────► generate_answer ─► END
```

核心状态贯穿 `rag_system/qa/state.py` 的 `QAState`（`total=False`，每个节点只返回自己负责的字段）。

---

## 二、已完成的节点

| 节点 | 文件 | 职责 | 关键产出字段 |
|---|---|---|---|
| `receive_query` | `rag_system/qa/nodes/receive_query.py` | 入口校验：query 去空白、空查询短路；`top_k` 归一化裁剪（1~50）；透传 `request_filters` | `query` `raw_query` `top_k` `request_filters` `error` |
| `analyze_query` | `rag_system/qa/nodes/analyze_query.py` | **意图分类 + 元数据过滤构建合并**为一次 LLM 调用。组装结构化 JSON payload（`system_prompt`/`query`/`chat_history`/`user_context`/`metadata_schema`/`filterable_metadata`/`current_time`），产出意图与过滤字段，归并显式 `request_filters`（覆盖 LLM 抽取）并编译成 Milvus 表达式 | `intent` `intent_confidence` `filters` `milvus_expr` `analysis_payload` |
| `rewrite_query` | `rag_system/qa/nodes/rewrite_query.py` | 结合聊天历史把指代性/口语化问题改写成独立检索问题（消解「它/这个」）。无历史直接透传，LLM 失败回退原查询 | `search_query` |
| `retrieve` | `rag_system/qa/nodes/retrieve.py` | **混合检索**：dense 向量（COSINE）+ BM25 全文，Milvus 服务端 RRF 融合。用 `search_query` 优先、`milvus_expr` 过滤 | `query_vector` `hits`（`distance` 为 RRF 融合分） |
| `rerank` | `rag_system/qa/nodes/rerank.py` | 交叉编码器精排（SiliconFlow `/rerank`）。重排 `hits`、写入 `rerank_score`、截断 `top_k`。未配置或失败时原样透传 | `hits`（含 `rerank_score`） |
| `enough_context` | `rag_system/qa/nodes/enough_context.py` | 上下文充分性判断。优先用 `rerank_score`（阈值 0.2），无 rerank 时退回 RRF 地板值（0.01）。闲聊/零命中判为不足 | `has_enough_context` `context_reason` |
| `compress_context` | `rag_system/qa/nodes/compress_context.py` | **LLM 抽取式压缩**：每个 chunk 独立抽取相关原文（不改写、不臆造），`Runnable.batch` 并发（上限 5）。无关段丢弃，全丢时退回原文 | `compressed_hits`（含 `compressed_text`） `context` |
| `generate_sql` | `rag_system/qa/nodes/generate_sql.py` | **metadata_qa 分支**：把元数据问题翻译成单条只读 SELECT。prompt 约束禁止 `SELECT *`、属性类问题只查 documents、计数用 COUNT、按名 ILIKE。经 `validate_select` 校验（单语句/无写关键字/自动补 LIMIT）。带 `sql_error` 时携错重写 | `generated_sql` `sql_error` `sql_attempts` |
| `execute_sql` | `rag_system/qa/nodes/execute_sql.py` | 在只读事务内执行 SQL（`statement_timeout` + 强制 LIMIT + 按 `allowed_kb_ids` 限范）。喂给生成节点前截断单格长文本并限行数（`QA_SQL_MAX_LLM_ROWS`），`sql_row_count` 仍报真实总行数 | `sql_rows` `sql_row_count` `sql_error` |
| `generate_answer` | `rag_system/qa/nodes/generate_answer.py` | 四入口统一生成：①chitchat 自然回复；②metadata_qa 基于 SQL 结果作答（结果序列化后按 `QA_METADATA_ROWS_MAX_CHARS` 兜底截断）；③检索充分→基于压缩上下文作答带引用 `[n]`；④不足→弱上下文兜底并提示资料不足、不编造；完全无命中直接说明无法回答 | `answer` `citations` |

### 支撑组件

| 文件 | 说明 |
|---|---|
| `rag_system/qa/graph.py` | 图组装与路由函数（`_route_after_receive` / `_route_after_analyze` / `_route_after_execute_sql` / `_route_after_enough`） |
| `rag_system/qa/state.py` | `QAState` / `QueryFilters` / `UserContext` / `ChatTurn` |
| `rag_system/qa/llm.py` | `get_chat_model()`（ChatOpenAI，`temperature=0`，`max_tokens` 由 `CHAT_MAX_TOKENS` 配置，env 回退链） |
| `rag_system/store/milvus.py` | schema（含 BM25 `Function` + 中文分析器 + sparse 字段）、`search`（纯 dense）、`hybrid_search`（dense+BM25 RRF）、`insert_chunks` |
| `rag_system/store/pg_query.py` | metadata_qa 用：`validate_select`（只读 SELECT 校验）、`run_readonly_query`（只读事务 + 超时 + kb 限范安全视图） |
| `rag_system/ingest/embedding.py` | `embed_texts` / `embed_query`（bge-m3，dim 1024） |
| `rag_system/rerank.py` | 交叉编码器 rerank 客户端（httpx，未配置返回空） |

### 测试与验证

- `tests/test_nodes_smoke.py` —— 全链路冒烟测试，mock 掉 LLM/embedding/Milvus/rerank，覆盖：检索问答（充分→压缩→生成+引用）、弱命中兜底、闲聊直连、空查询短路、top_k 裁剪。**全绿**。
- `tests/verify_hybrid.py` —— 真实链路验证：从 `output/compile_env/*_chunks.json` 入库，对比纯 dense vs 混合检索的召回差异，跑真实 LLM 压缩 + 全链路 QA。已确认 BM25 在关键词型查询上补召回、生成答案基于上下文且不编造。

---

## 三、真实链路已验证的结论

- **混合检索生效**：关键词型查询（如 `x86_64 Linux 操作系统`、`gcc 版本`）BM25 补召回了 dense top-5 之外、但关键词匹配的 chunk。语义型查询两路重合（符合预期）。
- **生成 grounding 生效**：对上下文未写明的信息如实说「上下文没有具体说明」，未编造；引用清单完整。
- **metadata_qa 结果膨胀已修**：早期「FitDiT 创建时间」等问题因 LLM 生成 `SELECT *`/查 chunks 正文，结果 `json.dumps` 灌进 prompt，单次问答曾达 ~7 万 token。已三层治理：generate_sql prompt 约束（禁 `SELECT *`、属性类只查 documents）、execute_sql 限行数（`QA_SQL_MAX_LLM_ROWS`）、generate_answer 序列化总长兜底截断（`QA_METADATA_ROWS_MAX_CHARS`）。

---

## 四、还要做什么（待办）

### 尚未实现的功能
1. **流式输出（streaming）**：`generate_answer` 目前是一次性 `invoke`，前端体验需要 token 级流式（`stream` / `astream`）。
2. **对外接口层**：目前只有 `build_qa_graph().invoke(...)`，缺 FastAPI/服务端点、请求响应模型、鉴权。
3. **多轮对话闭环**：`chat_history` 已被 `rewrite_query`/`analyze_query` 消费，但没有把本轮 `answer` 回写历史、没有会话存储（Redis 已在 compose 里，可接入）。
4. **不足分支的更优策略**：现在「不足」直接兜底作答。可选：放宽过滤重检一次、向用户澄清、query 扩展后二次检索。
5. **权限过滤落地**：`user_context.allowed_kb_ids` 已在 payload 里，但 `milvus_expr` 尚未强制按用户可见 kb 过滤（**安全相关**，建议优先）。
6. **`kb_id` 归一化缺失**：`analyze_query` 的 `_extracted_filters` 处理了 department/doc_type/language，`kb_id`/`document_id` 只从 `request_filters` 来，符合设计，但需确认调用方一定会传。

### 工程化
7. **单元测试拆分**：目前是单文件冒烟测试，缺各节点的独立单测与边界用例（异常回退路径、空 hits、rerank 失败等）。
8. **可观测性**：加结构化日志/耗时埋点（每个节点的延迟、LLM token 消耗、检索命中数）。
9. **配置集中化**：阈值散落在各节点的 env（`QA_CONTEXT_*`、`RERANK_*`），建议收敛到一个 config 模块。

---

## 五、已完成部分的优化方向

### 检索 / 融合
- **RRF 参数调优**：`hybrid_search` 的 `rrf_k=60`、两路召回深度 `top_k*2` 都是通用默认值，应在真实数据上调。
- **等权 → 加权融合**（✅ 已实现）：`hybrid_search` 新增 `ranker` 开关。`HYBRID_RANKER=weighted` 时改用 `WeightedRanker`，dense/sparse 权重取自 `HYBRID_DENSE_WEIGHT` / `HYBRID_SPARSE_WEIGHT`（默认各 1.0，等价等权）；关键词场景调高 sparse 权重即可让 BM25 更主导。默认仍为 `rrf`。
- **中文分析器**：现用 Milvus 内置 `chinese` 分析器。若领域词多，可评估自定义分析器 / 词典。

### 压缩
- **短文本护栏**（✅ 已实现）：`compress_context` 增加短文本护栏——原文短于 `QA_COMPRESS_MIN_LENGTH`（默认 200 字）的片段跳过 LLM 抽取、原样保留，只对长片段压缩。既消除短标题被「压」变长的负优化，也省掉短片段的 LLM 调用与延迟。
- **压缩成本**：每 chunk 一次 LLM 调用（已并发）。大 `top_k` 时成本可观，可考虑「先按 rerank_score 截断到 N 条再压缩」。

### 充分性判断
- **阈值校准**：`enough_context` 的 `rerank_score>=0.2` / RRF 地板 `0.01` 需按真实分布校准。
- **判据升级**：现为分数阈值。若要更鲁棒，可换成一次 LLM 评判「这些片段能否回答问题」（接口不变）。

### 生成
- **答案后处理**（已知瑕疵）：小模型（Qwen2.5-7B）+ `temperature=0` 偶发重复标点（「，，，」）。可加正则清理，或换更大的 chat 模型。
- **引用校验**：目前不校验答案里的 `[n]` 是否真的出现在 citations 范围内，可加后处理剔除幻觉引用。

### 数据 / Schema
- **Schema 变更需重入库**（✅ 已支持）：`rag_system/store/milvus.py` 引入 schema 版本标记（写入 collection `description` 的 `schema_v{N}`，当前 v2）。`ensure_collection` 检测到已存在 collection 的版本与代码不符时会打警告，提示重入库；新增 `drop_collection()` / `recreate_collection()` 辅助函数。`start.py --recreate` 会先 drop 再按最新 schema 重建后写入（**会清空旧数据**）。**结构变更（如加 BM25 字段）后必须走一次重入库。**
- **入库脚本**：`start.py` 目前面向单 PDF，批量入库 / 增量更新 / 删除文档的路径尚未覆盖。

---

## 六、环境变量速查（QA 相关）

| 变量 | 默认值 | 用途 |
|---|---|---|
| `CHAT_MODEL` | `Qwen/Qwen3.5-397B-A17B` | 生成/分类/改写/压缩/SQL 用 chat 模型 |
| `CHAT_API_KEY` / `CHAT_BASE_URL` | 回退 VISION/EMBEDDING | chat 模型鉴权 |
| `CHAT_MAX_TOKENS` | `2048` | chat 模型输出 token 上限（兜底） |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 向量化模型 |
| `EMBEDDING_DIM` | `1024` | 向量维度（须与 Milvus schema 一致） |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | 交叉编码器重排模型 |
| `RERANK_API_KEY` / `RERANK_BASE_URL` | 回退 EMBEDDING | rerank 鉴权；未配置则 rerank 透传 |
| `QA_CONTEXT_RERANK_THRESHOLD` | `0.2` | 充分性判断（有 rerank 时） |
| `QA_CONTEXT_RRF_FLOOR` | `0.01` | 充分性判断（无 rerank 时的 RRF 地板） |
| `QA_CONTEXT_MIN_HITS` | `1` | 达到阈值的最少命中数 |
| `QA_COMPRESS_MIN_LENGTH` | `200` | 压缩短文本护栏：原文短于此长度跳过 LLM 压缩 |
| `QA_SQL_MAX_LLM_ROWS` | `20` | metadata_qa：喂给 generate_answer 的最多结果行数（防 `SELECT *` 撑爆 prompt；总行数仍如实报） |
| `QA_SQL_MAX_CELL_LEN` | `500` | metadata_qa：单个结果格长文本截断长度 |
| `QA_METADATA_ROWS_MAX_CHARS` | `6000` | metadata_qa：结果行序列化后喂给 LLM 的总字符上限（兜底截断） |
| `HYBRID_RANKER` | `rrf` | 混合检索融合策略：`rrf` 或 `weighted` |
| `HYBRID_DENSE_WEIGHT` / `HYBRID_SPARSE_WEIGHT` | `1.0` / `1.0` | `weighted` 时 dense / BM25 的融合权重 |
| `MILVUS_URI` / `MILVUS_COLLECTION` | `http://127.0.0.1:19530` / `rag_chunks` | Milvus 连接 |

---

## 七、如何运行

```bash
# 冒烟测试（不依赖外部服务）
python tests/test_nodes_smoke.py

# 真实链路验证（需 Milvus + embedding/chat/rerank key）
python tests/verify_hybrid.py

# 代码内调用
from rag_system.qa import build_qa_graph
app = build_qa_graph()
out = app.invoke({"query": "编译环境怎么配置？", "request_filters": {"kb_id": "demo"}})
print(out["answer"], out["citations"])
```
