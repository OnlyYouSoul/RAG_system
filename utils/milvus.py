import os
from functools import lru_cache

from pymilvus import DataType, MilvusClient

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "")
MILVUS_DB = os.getenv("MILVUS_DB", "default")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "rag_chunks")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

_TEXT_MAX_LENGTH = 8192

_ID_MAX_LENGTH = 64
_TITLE_MAX_LENGTH = 512
_ENUM_MAX_LENGTH = 64

# 需要建标量索引、用于过滤的字段
_SCALAR_INDEX_FIELDS = ("document_id", "kb_id", "doc_type", "department")

_METRIC_TYPE = "COSINE"
_INDEX_TYPE = "HNSW"


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:

    kwargs = {"uri": MILVUS_URI, "db_name": MILVUS_DB}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**kwargs)


def ensure_collection(
    collection_name: str = MILVUS_COLLECTION,
    dim: int = EMBEDDING_DIM,
) -> str:

    client = get_client()  

    if client.has_collection(collection_name):
        client.load_collection(collection_name)
        return collection_name

    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
        description="RAG chunk 向量库",
    )
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX_LENGTH)
    schema.add_field("chunk_id", DataType.INT64)

    schema.add_field("document_id", DataType.VARCHAR, max_length=_ID_MAX_LENGTH)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=_ID_MAX_LENGTH)
    schema.add_field("title", DataType.VARCHAR, max_length=_TITLE_MAX_LENGTH)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=_ENUM_MAX_LENGTH)
    schema.add_field("department", DataType.VARCHAR, max_length=_ENUM_MAX_LENGTH)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("chunk_size", DataType.INT64)
    schema.add_field("overlap_size", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=_INDEX_TYPE,
        metric_type=_METRIC_TYPE,
        params={"M": 16, "efConstruction": 200},
    )
    # 过滤字段建标量索引，加速 filter
    for field in _SCALAR_INDEX_FIELDS:
        index_params.add_index(field_name=field)

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )
    client.load_collection(collection_name)
    return collection_name


def insert_chunks(
    chunks: list,
    vectors: list[list[float]],
    collection_name: str = MILVUS_COLLECTION,
) -> dict:
    
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks({len(chunks)}) 与 vectors({len(vectors)}) 数量不一致"
        )

    client = get_client()
    ensure_collection(collection_name)

    rows = []
    for chunk, vector in zip(chunks, vectors):
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(
                f"向量维度 {len(vector)} 与配置 EMBEDDING_DIM={EMBEDDING_DIM} 不一致"
            )
        metadata = chunk.metadata or {}
        row = {
            "vector": vector,
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            **{k: v for k, v in metadata.items() if v is not None},
        }
        rows.append(row)

    return client.insert(collection_name=collection_name, data=rows)


def search(
    query_vector: list[float],
    top_k: int = 5,
    filter: str = "",
    collection_name: str = MILVUS_COLLECTION,
    output_fields: list[str] | None = None,
) -> list[dict]:

    client = get_client()
    ensure_collection(collection_name)

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=top_k,
        filter=filter,
        output_fields=output_fields or ["text", "chunk_id", "*"],
        search_params={"metric_type": _METRIC_TYPE, "params": {"ef": 64}},
    )

    hits = results[0] if results else []
    return [
        {"distance": hit["distance"], **hit.get("entity", {})}
        for hit in hits
    ]
