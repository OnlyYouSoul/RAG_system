import os
from functools import lru_cache

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)

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

# 全文检索（BM25）相关
_SPARSE_FIELD = "sparse_vector"          # BM25 Function 产出的稀疏向量字段
_BM25_METRIC = "BM25"
# 中文分析器：Milvus 内置，服务端分词，无需 python 侧 jieba
_TEXT_ANALYZER = {"type": "chinese"}

# Schema 版本：schema 结构（字段/分析器/Function/索引）变更时 +1。
# ensure_collection 会比对已存在 collection 的版本，不一致时警告——
# 因为老 collection 缺 BM25 sparse 字段/分析器，混合检索会失败，必须 drop 重建。
_SCHEMA_VERSION = "2"
_DESC_PREFIX = "RAG chunk 向量库"
_SCHEMA_DESCRIPTION = f"{_DESC_PREFIX} | schema_v{_SCHEMA_VERSION}"

# 混合检索融合策略：
# - HYBRID_RANKER=rrf（默认）    等权 Reciprocal Rank Fusion
# - HYBRID_RANKER=weighted       按分数加权融合（WeightedRanker）
#   dense/sparse 权重分别取自 HYBRID_DENSE_WEIGHT / HYBRID_SPARSE_WEIGHT。
#   关键词/专有名词多的场景可调高 sparse 权重，让 BM25 更主导。
_HYBRID_RANKER = os.getenv("HYBRID_RANKER", "rrf").lower()
_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "1.0"))
_SPARSE_WEIGHT = float(os.getenv("HYBRID_SPARSE_WEIGHT", "1.0"))


@lru_cache(maxsize=1)
def get_client() -> MilvusClient:

    kwargs = {"uri": MILVUS_URI, "db_name": MILVUS_DB}
    if MILVUS_TOKEN:
        kwargs["token"] = MILVUS_TOKEN
    return MilvusClient(**kwargs)


def _collection_schema_version(client: MilvusClient, collection_name: str) -> str | None:
    """从已存在 collection 的 description 里解析 schema 版本。老库无版本标记时返回 None。"""
    try:
        desc = client.describe_collection(collection_name)
    except Exception:
        return None
    text = str(desc.get("description", "")) if isinstance(desc, dict) else ""
    marker = "schema_v"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    return text[idx + len(marker):].strip() or None


def drop_collection(collection_name: str = MILVUS_COLLECTION) -> None:
    """删除 collection（若存在）。schema 变更后重入库前调用。"""
    client = get_client()
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)


def recreate_collection(
    collection_name: str = MILVUS_COLLECTION,
    dim: int = EMBEDDING_DIM,
) -> str:
    """drop 后按最新 schema 重建"""
    drop_collection(collection_name)
    return ensure_collection(collection_name, dim)


def ensure_collection(
    collection_name: str = MILVUS_COLLECTION,
    dim: int = EMBEDDING_DIM,
) -> str:

    client = get_client()

    if client.has_collection(collection_name):
        existing = _collection_schema_version(client, collection_name)
        if existing != _SCHEMA_VERSION:
            print(
                f"[milvus][warn] collection '{collection_name}' 的 schema 版本为 "
                f"{existing or '未标记(旧版)'}，当前代码为 v{_SCHEMA_VERSION}。"
                "字段/分析器/Function 可能已变更，混合检索或写入会失败——"
                "请重入库：recreate_collection() 或 start.py --recreate。"
            )
        client.load_collection(collection_name)
        return collection_name

    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
        description=_SCHEMA_DESCRIPTION,
    )
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)
    # text 开启分析器：作为 BM25 全文检索的输入
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=_TEXT_MAX_LENGTH,
        enable_analyzer=True,
        analyzer_params=_TEXT_ANALYZER,
    )
    # BM25 产出的稀疏向量（由下方 Function 自动生成，无需手动写入）
    schema.add_field(_SPARSE_FIELD, DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("chunk_id", DataType.INT64)

    schema.add_field("document_id", DataType.VARCHAR, max_length=_ID_MAX_LENGTH)
    schema.add_field("kb_id", DataType.VARCHAR, max_length=_ID_MAX_LENGTH)
    schema.add_field("title", DataType.VARCHAR, max_length=_TITLE_MAX_LENGTH)
    schema.add_field("doc_type", DataType.VARCHAR, max_length=_ENUM_MAX_LENGTH)
    schema.add_field("department", DataType.VARCHAR, max_length=_ENUM_MAX_LENGTH)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("chunk_size", DataType.INT64)
    schema.add_field("overlap_size", DataType.INT64)

    # BM25：把 text 文本自动转成 sparse_vector（服务端分词 + 建 BM25 稀疏表示）
    schema.add_function(
        Function(
            name="text_bm25",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=[_SPARSE_FIELD],
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type=_INDEX_TYPE,
        metric_type=_METRIC_TYPE,
        params={"M": 16, "efConstruction": 200},
    )
    # 稀疏向量（BM25）索引
    index_params.add_index(
        field_name=_SPARSE_FIELD,
        index_type="SPARSE_INVERTED_INDEX",
        metric_type=_BM25_METRIC,
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
        anns_field="vector",  # collection 含多个向量字段，须显式指定 dense 字段
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


def hybrid_search(
    query_vector: list[float],
    query_text: str,
    top_k: int = 5,
    filter: str = "",
    collection_name: str = MILVUS_COLLECTION,
    output_fields: list[str] | None = None,
    dense_limit: int | None = None,
    sparse_limit: int | None = None,
    rrf_k: int = 60,
    ranker: str | None = None,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
) -> list[dict]:
    """混合检索：dense 向量 + BM25 全文，服务端融合。

    融合策略由 ``ranker`` 决定（默认取环境变量 HYBRID_RANKER）：
    - ``rrf``      等权 Reciprocal Rank Fusion（只看名次，抗分数尺度差异）。
    - ``weighted`` WeightedRanker，按归一化分数加权。dense/sparse 权重取自
      ``dense_weight``/``sparse_weight``（默认取 HYBRID_DENSE_WEIGHT /
      HYBRID_SPARSE_WEIGHT）。关键词场景可调高 sparse 权重让 BM25 更主导。

    返回每条含 distance（融合分，越大越好）与实体字段。
    """
    client = get_client()
    ensure_collection(collection_name)

    # 两路各自的召回深度
    dense_limit = dense_limit or max(top_k * 2, top_k)
    sparse_limit = sparse_limit or max(top_k * 2, top_k)

    dense_req = AnnSearchRequest(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": _METRIC_TYPE, "params": {"ef": 64}},
        limit=dense_limit,
        expr=filter or None,
    )
    sparse_req = AnnSearchRequest(
        data=[query_text],
        anns_field=_SPARSE_FIELD,
        param={"metric_type": _BM25_METRIC},
        limit=sparse_limit,
        expr=filter or None,
    )

    strategy = (ranker or _HYBRID_RANKER).lower()
    if strategy == "weighted":
        # WeightedRanker 内部对各路分数归一化后加权求和
        fusion = WeightedRanker(
            dense_weight if dense_weight is not None else _DENSE_WEIGHT,
            sparse_weight if sparse_weight is not None else _SPARSE_WEIGHT,
        )
    else:
        fusion = RRFRanker(rrf_k)

    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=[dense_req, sparse_req],
        ranker=fusion,
        limit=top_k,
        output_fields=output_fields or ["text", "chunk_id", "*"],
    )

    hits = results[0] if results else []
    return [
        {"distance": hit["distance"], **hit.get("entity", {})}
        for hit in hits
    ]
