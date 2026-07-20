"""metadata_qa 分支：执行 generate_sql 产出的只读 SELECT。

- 经 utils.pg_query.run_readonly_query 在只读事务里执行，带 statement_timeout、
  强制 LIMIT，并按 user_context.allowed_kb_ids 限范。
- 成功：写 sql_rows / sql_row_count，清空 sql_error。
- 失败：写 sql_error（保留 generated_sql），交由路由决定重试还是兜底。
"""

from __future__ import annotations

import os

from rag_system.qa.state import QAState
from rag_system.store.pg_query import run_readonly_query

# 单行结果里过大的文本列（如 chunks.text）截断，避免塞爆生成上下文
_MAX_CELL_LEN = int(os.getenv("QA_SQL_MAX_CELL_LEN", "500"))
# 喂给 generate_answer 的最多行数（元数据问答通常只需少量行；防 SELECT * 捞回整表）
_MAX_LLM_ROWS = int(os.getenv("QA_SQL_MAX_LLM_ROWS", "20"))


def _truncate_cell(value):
    if isinstance(value, str) and len(value) > _MAX_CELL_LEN:
        return value[:_MAX_CELL_LEN] + "…"
    return value


def execute_sql(state: QAState) -> QAState:
    if state.get("error"):
        return {}

    # generate_sql 生成失败时会留下 sql_error 且无 generated_sql；
    # 保留该 sql_error 让路由决定重试/兜底，不要覆写成 error 短路整张图。
    sql = state.get("generated_sql")
    if not sql:
        return {"sql_error": state.get("sql_error") or "缺少待执行的 SQL"}

    user_context = state.get("user_context", {}) or {}
    allowed_kb_ids = user_context.get("allowed_kb_ids") or None

    try:
        rows = run_readonly_query(sql, allowed_kb_ids=allowed_kb_ids)
    except Exception as exc:
        # 不写 error：保留 sql_error 让路由决定重试，超出上限再兜底
        return {"sql_error": str(exc)}

    total = len(rows)
    # 逐格截断长文本 + 限制喂给 LLM 的行数，防止 SELECT * 把整表 chunk 正文
    # 灌进 generate_answer 的 prompt（曾导致单次问答 7 万 token）。
    capped = rows[:_MAX_LLM_ROWS]
    capped = [{k: _truncate_cell(v) for k, v in row.items()} for row in capped]
    return {
        "sql_rows": capped,
        "sql_row_count": total,  # 真实行数（供计数类问题直接使用）
        "sql_error": "",
    }
