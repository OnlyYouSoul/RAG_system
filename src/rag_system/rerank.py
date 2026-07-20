"""重排序（rerank）客户端。

调用 SiliconFlow / 兼容 OpenAI 风格的 /rerank 接口，用交叉编码器对
候选文档按与 query 的相关性重新打分。未配置 RERANK_* 时返回空表示不可用，
上层据此退化为不重排。
"""

import os
from functools import lru_cache

import httpx

RERANK_BASE_URL = os.getenv(
    "RERANK_BASE_URL",
    os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"),
)
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
_TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "15"))


def _api_key() -> str:
    # 调用时才读 env，避免模块导入早于 load_dotenv
    return os.getenv("RERANK_API_KEY") or os.getenv("EMBEDDING_API_KEY") or ""


def is_configured() -> bool:
    return bool(_api_key())


@lru_cache(maxsize=1)
def _client() -> httpx.Client:
    base = os.getenv("RERANK_BASE_URL", RERANK_BASE_URL).rstrip("/")
    return httpx.Client(base_url=base, timeout=_TIMEOUT)


def rerank(query: str, documents: list[str], top_n: int | None = None) -> list[dict]:
    """对 documents 按与 query 的相关性重排。

    返回 [{"index": 原始下标, "score": 相关性分数}, ...]，按分数从高到低。
    未配置或文档为空时返回空列表，调用方据此退化为不重排。
    """
    if not documents or not is_configured():
        return []

    payload = {
        "model": os.getenv("RERANK_MODEL", RERANK_MODEL),
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = top_n

    resp = _client().post(
        "/rerank",
        json=payload,
        headers={"Authorization": f"Bearer {_api_key()}"},
    )
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    return [
        {"index": item["index"], "score": item.get("relevance_score", item.get("score", 0.0))}
        for item in results
    ]
