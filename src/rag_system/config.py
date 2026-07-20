"""集中管理项目路径常量与环境加载。

原先 PROJECT_DIR 散落在 rewrite.py / sqlite_store.py / start.py，各自
用 Path(__file__).parent 回溯，重构后层级变化易错。这里统一从包位置
回溯到仓库根，供各模块引用。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 本文件位于 <repo>/src/rag_system/config.py，向上三级即仓库根。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MinerU 源码目录：与本仓库同级（pyproject 里的可编辑依赖 ../MinerU）。
MINERU_SOURCE_DIR = PROJECT_ROOT.parent / "MinerU"

# 输入数据与产物目录。
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
SQLITE_DIR = DATA_DIR / "sqlite"
SQLITE_DB_PATH = SQLITE_DIR / "rag_metadata.db"

# MinerU 解析 + chunk 产物统一目录。
OUTPUT_DIR = PROJECT_ROOT / "output"

# .env 位置（仓库根）。
ENV_PATH = PROJECT_ROOT / ".env"


def load_env() -> None:
    """加载仓库根的 .env（若存在）。"""
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH)
    else:
        load_dotenv()


def default_sqlite_db_path() -> Path:
    """SQLite 库路径，允许用 RAG_SQLITE_DB 覆盖。"""
    override = os.getenv("RAG_SQLITE_DB")
    return Path(override) if override else SQLITE_DB_PATH
