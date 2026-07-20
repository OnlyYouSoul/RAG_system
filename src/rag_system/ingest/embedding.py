import os
from functools import lru_cache

from openai import OpenAI

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

_BATCH_SIZE = 32


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    # 调用时才读 env，避免模块导入早于 load_dotenv 时拿到空值
    return OpenAI(
        api_key=os.getenv("EMBEDDING_API_KEY", ""),
        base_url=os.getenv("EMBEDDING_BASE_URL", EMBEDDING_BASE_URL),
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    
    if not texts:
        return []

    client = get_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        
        for item in sorted(response.data, key=lambda d: d.index):
            if len(item.embedding) != EMBEDDING_DIM:
                raise ValueError(
                    f"向量维度 {len(item.embedding)} 与配置 EMBEDDING_DIM={EMBEDDING_DIM} 不一致，"
                    f"请检查 EMBEDDING_MODEL={EMBEDDING_MODEL}"
                )
            vectors.append(item.embedding)

    return vectors


def embed_query(text: str) -> list[float]:
    """把单条查询转成向量，用于检索。"""
    return embed_texts([text])[0]
