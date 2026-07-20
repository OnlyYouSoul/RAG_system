"""只读、受限的 LLM SQL 执行层（元数据问答用）。

安全约束（多道防护叠加）：
1. 只读事务：整条查询在 ``default_transaction_read_only = on`` 事务内执行，
   底层杜绝任何写操作，即使校验被绕过也无法改数据。
2. 单条 SELECT 校验：拒绝多语句、非 SELECT/WITH 开头、含 DDL/DML 关键字的语句。
3. 强制 LIMIT：顶层查询没有 LIMIT 时自动补 ``LIMIT _MAX_ROWS``，防大结果集。
4. statement_timeout：单条查询超时上限，防慢查询拖垮库。
5. allowed_kb_ids 限范：调用方带权限时，SQL 只能看到 rag_secure schema 下的
   安全视图（按会话变量 app.allowed_kb_ids 过滤），用户越权也查不到别的知识库。

这些约束共同把「让 LLM 自由写 SQL」的风险压到可接受范围。
"""

from __future__ import annotations

import os
import re

import psycopg
from psycopg import sql as pgsql
from psycopg.rows import dict_row

from rag_system.store.postgres_store import get_dsn

# 结果行上限（顶层无 LIMIT 时自动补）
_MAX_ROWS = int(os.getenv("PG_QUERY_MAX_ROWS", "100"))
# 单条查询超时（毫秒）
_STATEMENT_TIMEOUT_MS = int(os.getenv("PG_QUERY_TIMEOUT_MS", "5000"))

# 只读问答放行的表（供 prompt 展示，也是安全视图的来源表）
ALLOWED_TABLES = ("documents", "chunks")

# 受限查询用的 schema：内含按 kb_id 过滤的安全视图，search_path 指向它。
_SECURE_SCHEMA = "rag_secure"

# 语句级黑名单关键字（整词匹配）。只读事务已兜底，这里做早失败 + 明确报错。
_FORBIDDEN = (
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "grant", "revoke", "copy", "merge", "call", "do", "vacuum", "analyze",
    "reindex", "comment", "lock", "set", "reset", "begin", "commit", "rollback",
    "savepoint", "prepare", "execute", "listen", "notify", "cluster", "refresh",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN) + r")\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\blimit\b", re.IGNORECASE)
_LEADING_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class SQLValidationError(ValueError):
    """生成的 SQL 未通过安全校验。"""


def validate_select(sql: str) -> str:
    """校验并规整单条只读 SELECT。返回清洗后的 SQL；不合规抛 SQLValidationError。"""
    if not sql or not sql.strip():
        raise SQLValidationError("SQL 为空")

    cleaned = sql.strip()
    # 去掉尾部分号（单条语句允许一个结尾分号）
    cleaned = cleaned.rstrip(";").strip()

    # 单语句：正文里不应再出现分号
    if ";" in cleaned:
        raise SQLValidationError("只允许单条 SQL 语句，检测到多语句")

    if not _LEADING_RE.match(cleaned):
        raise SQLValidationError("只允许 SELECT / WITH 查询语句")

    hit = _FORBIDDEN_RE.search(cleaned)
    if hit:
        raise SQLValidationError(f"检测到禁止的关键字：{hit.group(1)}")

    # 顶层没有 LIMIT 时自动补，防大结果集
    if not _LIMIT_RE.search(cleaned):
        cleaned = f"{cleaned}\nLIMIT {_MAX_ROWS}"

    return cleaned


def ensure_secure_views(dsn: str | None = None) -> None:
    """幂等创建 rag_secure schema 及按会话变量过滤 kb_id 的安全视图。

    视图用 current_setting('app.allowed_kb_ids', true)：
    - 未设置/为空 -> 不限制（返回全部）。
    - 设置为逗号分隔的 kb_id 列表 -> 只放行这些 kb 的行。
    security_barrier 防止用户通过巧妙的谓词把过滤条件挤到后面执行而泄露数据。

    注意：创建视图是 DDL，必须在独立的读写（autocommit）连接里做，
    不能放进只读查询事务。
    """
    conn = psycopg.connect(dsn or get_dsn(), autocommit=True)
    try:
        with conn.cursor() as cur:
            cur.execute(
                pgsql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    pgsql.Identifier(_SECURE_SCHEMA)
                )
            )
            cur.execute(
                f"""
                CREATE OR REPLACE VIEW {_SECURE_SCHEMA}.documents
                WITH (security_barrier = true) AS
                SELECT * FROM public.documents
                WHERE current_setting('app.allowed_kb_ids', true) IS NULL
                   OR current_setting('app.allowed_kb_ids', true) = ''
                   OR kb_id = ANY (
                       string_to_array(current_setting('app.allowed_kb_ids', true), ',')
                   )
                """
            )
            # chunks 按其所属 document 的 kb_id 过滤
            cur.execute(
                f"""
                CREATE OR REPLACE VIEW {_SECURE_SCHEMA}.chunks
                WITH (security_barrier = true) AS
                SELECT c.* FROM public.chunks c
                JOIN {_SECURE_SCHEMA}.documents d USING (document_id)
                """
            )
    finally:
        conn.close()


def run_readonly_query(
    query_sql: str,
    allowed_kb_ids: list[str] | None = None,
    dsn: str | None = None,
) -> list[dict]:
    """在只读事务内执行经校验的 SELECT，返回行列表（dict）。

    query_sql 会先经 validate_select 校验/补 LIMIT。allowed_kb_ids 非空时，
    search_path 指向安全视图并设置会话变量，SQL 只能看到放行知识库的数据。
    """
    safe_sql = validate_select(query_sql)

    # 安全视图是 DDL，必须在只读事务外先建好（幂等）
    if allowed_kb_ids:
        ensure_secure_views(dsn)

    conn = psycopg.connect(dsn or get_dsn(), autocommit=False, row_factory=dict_row)
    try:
        with conn.cursor() as cur:
            # 只读必须是事务里的第一条语句
            cur.execute("SET TRANSACTION READ ONLY")
            cur.execute(
                pgsql.SQL("SET LOCAL statement_timeout = {}").format(
                    pgsql.Literal(_STATEMENT_TIMEOUT_MS)
                )
            )

            # 权限限范：search_path 指向安全视图，注入放行的 kb_id
            if allowed_kb_ids:
                cur.execute(
                    pgsql.SQL("SET LOCAL search_path = {}, public").format(
                        pgsql.Identifier(_SECURE_SCHEMA)
                    )
                )
                cur.execute(
                    "SELECT set_config('app.allowed_kb_ids', %s, true)",
                    (",".join(allowed_kb_ids),),
                )

            cur.execute(safe_sql)
            rows = cur.fetchall()
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
