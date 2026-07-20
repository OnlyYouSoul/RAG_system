"""SQLite 持久化：保存 chunking 结果与元数据。

- 单一共享库，默认落在 <项目根>/data/sqlite/rag_metadata.db。
- 两张表：
    documents  文档级元数据（一行一个文档，主键 document_id）
    chunks     chunk 级元数据（一行一个 chunk，外键 document_id -> documents）
- chunks 表规范化：只存 chunk 自身字段，文档级字段（kb_id/title/…）只在 documents 表，
  查询时按 document_id JOIN。
- 同一文档（document_id 相同，由文件内容 hash 派生）再次入库时覆盖：
  先删该文档的旧行与全部旧 chunk（外键 ON DELETE CASCADE），再插新数据。
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from rag_system import config

SQLITE_DIR = config.SQLITE_DIR
DEFAULT_DB_PATH = config.SQLITE_DB_PATH

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
    ingested_at   TEXT,
    total_chunks  INTEGER
)
"""

# chunk_id 是文档内序号，仅在单个文档内唯一，故主键为 (document_id, chunk_id)。
_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    document_id   TEXT NOT NULL,
    chunk_id      INTEGER NOT NULL,
    text          TEXT NOT NULL,
    chunk_index   INTEGER,
    chunk_size    INTEGER,
    overlap_size  INTEGER,
    PRIMARY KEY (document_id, chunk_id),
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
)
"""

_CREATE_CHUNK_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
)


@contextmanager
def _connect(db_path: Path):
    """打开连接：开启外键约束，返回后自动 commit/回滚并关闭。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB_PATH) -> Path:
    """建表（幂等）。返回实际 db 路径。"""
    with _connect(db_path) as conn:
        conn.execute(_CREATE_DOCUMENTS)
        conn.execute(_CREATE_CHUNKS)
        conn.execute(_CREATE_CHUNK_INDEX)
    return db_path


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_DOCUMENTS)
    conn.execute(_CREATE_CHUNKS)
    conn.execute(_CREATE_CHUNK_INDEX)


def save_ingest_result(
    doc_metadata: dict,
    chunks: list,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    """把一次 ingest 的文档级元数据与全部 chunk 写入 SQLite（覆盖式）。

    doc_metadata：build_document_metadata 的产出（文档级字段）。
    chunks：chunking.Chunk 列表，每个含 chunk_id / text / metadata。

    同一 document_id 已存在时，先删旧文档行（chunks 随外键级联删除）再插新数据，
    保证库内与最近一次 ingest 一致。整个操作在单事务内完成。
    """
    document_id = doc_metadata.get("document_id")
    if not document_id:
        raise ValueError("doc_metadata 缺少 document_id，无法写入 SQLite")

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

    doc_placeholders = ", ".join("?" for _ in _DOCUMENT_COLUMNS)
    doc_cols = ", ".join(_DOCUMENT_COLUMNS)
    chunk_cols = "document_id, chunk_id, text, " + ", ".join(_CHUNK_META_COLUMNS)
    chunk_placeholders = ", ".join("?" for _ in range(3 + len(_CHUNK_META_COLUMNS)))

    with _connect(db_path) as conn:
        _ensure_schema(conn)
        # 覆盖：显式删旧文档行，chunks 经 ON DELETE CASCADE 一并清除
        conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        conn.execute(
            f"INSERT INTO documents ({doc_cols}) VALUES ({doc_placeholders})",
            doc_row,
        )
        conn.executemany(
            f"INSERT INTO chunks ({chunk_cols}) VALUES ({chunk_placeholders})",
            chunk_rows,
        )

    return {
        "db_path": str(db_path),
        "document_id": document_id,
        "chunks_written": len(chunk_rows),
    }
