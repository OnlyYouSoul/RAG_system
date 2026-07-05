import hashlib
from datetime import datetime, timezone
from pathlib import Path


def compute_file_hash(file_path: Path) -> str:
    """计算文件内容的SHA256，用于去重"""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def build_document_metadata(
    source_path,
    *,
    kb_id: str,
    department: str = "unknown",
    language: str = "zh",
    doc_type: str = "pdf",
    title: str | None = None,
) -> dict:
    
    path = Path(source_path).resolve()
    file_hash = compute_file_hash(path)

    return {
        # document_id 由内容 hash 派生
        "document_id": "doc_" + file_hash[:16],
        "kb_id": kb_id,
        "title": title or path.stem,
        "source_file": path.name,
        "source_path": str(path),
        "doc_type": doc_type,
        "file_hash": file_hash,
        "language": language,
        "department": department,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "total_chunks": None,
    }

CHUNK_DOC_FIELDS = ("document_id", "kb_id", "title", "doc_type", "department")

def chunk_doc_fields(doc_metadata: dict) -> dict:
    """从全量文档档案中抽取要嵌进 chunk 的精简字段。"""
    return {k: doc_metadata[k] for k in CHUNK_DOC_FIELDS if k in doc_metadata}