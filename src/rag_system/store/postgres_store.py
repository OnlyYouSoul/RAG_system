"""PostgreSQL 持久化：保存 chunking 结果与元数据（SQLite 方案的正式替代）。

- 连接信息来自环境变量：优先 POSTGRES_DSN / DATABASE_URL，否则由
  POSTGRES_HOST / PORT / DB / USER / PASSWORD 组装（默认对接 docker 里的 postgres:17）。
- 两张表：
    documents  文档级元数据（一行一个文档，主键 document_id）
    chunks     chunk 级元数据（一行一个 chunk，外键 document_id -> documents）
- chunks 表规范化：只存 chunk 自身字段，文档级字段（kb_id/title/…）只在 documents 表，
  查询时按 document_id JOIN。
- 同一文档（document_id 相同，由文件内容 hash 派生）再次入库时覆盖：
  先删该文档的旧行与全部旧 chunk（外键 ON DELETE CASCADE），再插新数据。
"""

import os
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

# documents 表列（与 build_document_metadata 产出的文档级字段对应）
_DOCUMENT_COLUMNS = (
    "document_id",
    "kb_id",
    "title",
    "source_file",
    "source_path",
    "doc_type",
    "file_hash",
    "language",
    "department",
    "ingested_at",
    "total_chunks",
)

# chunks 表存的 chunk 自身字段（文档级字段不冗余，靠 document_id 关联）
_CHUNK_META_COLUMNS = ("chunk_index", "chunk_size", "overlap_size")


def get_dsn() -> str:
    """拼出 Postgres 连接串。优先完整 DSN，否则由分量组装。"""
    dsn = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL")
    if dsn:
        return dsn
    host = os.getenv("POSTGRES_HOST", "127.0.0.1")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "rag")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "123456")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


_CREATE_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    kb_id         TEXT,
    title         TEXT,
    source_file   TEXT,
    source_path   TEXT,
    doc_type      TEXT,
    file_hash     TEXT,
    language      TEXT,
    department    TEXT,
    ingested_at   TIMESTAMPTZ,
    total_chunks  INTEGER
)
"""

# chunk_id 是文档内序号，仅在单个文档内唯一，故主键为 (document_id, chunk_id)。
_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    document_id   TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_id      INTEGER NOT NULL,
    text          TEXT NOT NULL,
    chunk_index   INTEGER,
    chunk_size    INTEGER,
    overlap_size  INTEGER,
    PRIMARY KEY (document_id, chunk_id)
)
"""

_CREATE_CHUNK_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
)


@contextmanager
def _connect(dsn: str | None = None):
    """打开连接，返回后自动 commit/回滚并关闭。"""
    conn = psycopg.connect(dsn or get_dsn(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_DOCUMENTS)
        cur.execute(_CREATE_CHUNKS)
        cur.execute(_CREATE_CHUNK_INDEX)


def init_db(dsn: str | None = None) -> None:
    """建表（幂等）。"""
    with _connect(dsn) as conn:
        _ensure_schema(conn)


def save_ingest_result(
    doc_metadata: dict,
    chunks: list,
    dsn: str | None = None,
) -> dict:
    """把一次 ingest 的文档级元数据与全部 chunk 写入 Postgres（覆盖式）。

    doc_metadata：build_document_metadata 的产出（文档级字段）。
    chunks：chunking.Chunk 列表，每个含 chunk_id / text / metadata。

    同一 document_id 已存在时，先删旧文档行（chunks 随外键级联删除）再插新数据，
    保证库内与最近一次 ingest 一致。整个操作在单事务内完成。
    """
    document_id = doc_metadata.get("document_id")
    if not document_id:
        raise ValueError("doc_metadata 缺少 document_id，无法写入 Postgres")

    doc_row = tuple(doc_metadata.get(col) for col in _DOCUMENT_COLUMNS)

    chunk_rows = []
    for chunk in chunks:
        meta = chunk.metadata or {}
        chunk_rows.append(
            (
                document_id,
                chunk.chunk_id,
                chunk.text,
                *(meta.get(col) for col in _CHUNK_META_COLUMNS),
            )
        )

    doc_placeholders = ", ".join("%s" for _ in _DOCUMENT_COLUMNS)
    doc_cols = ", ".join(_DOCUMENT_COLUMNS)
    chunk_cols = "document_id, chunk_id, text, " + ", ".join(_CHUNK_META_COLUMNS)
    chunk_placeholders = ", ".join("%s" for _ in range(3 + len(_CHUNK_META_COLUMNS)))

    with _connect(dsn) as conn:
        _ensure_schema(conn)
        with conn.cursor() as cur:
            # 覆盖：显式删旧文档行，chunks 经 ON DELETE CASCADE 一并清除
            cur.execute("DELETE FROM documents WHERE document_id = %s", (document_id,))
            cur.execute(
                f"INSERT INTO documents ({doc_cols}) VALUES ({doc_placeholders})",
                doc_row,
            )
            if chunk_rows:
                cur.executemany(
                    f"INSERT INTO chunks ({chunk_cols}) VALUES ({chunk_placeholders})",
                    chunk_rows,
                )

    return {
        "document_id": document_id,
        "chunks_written": len(chunk_rows),
    }
