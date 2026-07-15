from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    api_key = _first_env("CHAT_API_KEY", "VISION_API_KEY", "EMBEDDING_API_KEY")
    base_url = _first_env(
        "CHAT_BASE_URL",
        "VISION_BASE_URL",
        "EMBEDDING_BASE_URL",
        default="https://api.siliconflow.cn/v1",
    )
    model = _first_env("CHAT_MODEL", default="Qwen/Qwen3.5-397B-A17B")

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )
