# Build context 必须是上级目录 /home/zhang/project，
# 这样才能同时 COPY 到 RAG_System 与它的可编辑依赖 MinerU。
FROM python:3.12-slim

# uv：依赖管理
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# MinerU / pdf 解析所需的系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 保持与本地一致的目录布局：
#   /app/RAG_System   ← 本项目
#   /app/MinerU       ← 可编辑依赖（pyproject 里的 ../MinerU）
WORKDIR /app/RAG_System

ENV UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/RAG_System/.venv \
    PATH="/app/RAG_System/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

# 先放依赖清单，命中构建缓存
COPY MinerU/ /app/MinerU/
COPY RAG_System/pyproject.toml RAG_System/uv.lock RAG_System/.python-version /app/RAG_System/

# 依据 lock 安装依赖（含可编辑的 MinerU）
RUN uv sync --frozen --no-install-project

# 再放项目源码
COPY RAG_System/ /app/RAG_System/

RUN uv sync --frozen

# 默认常驻，方便 `docker compose exec app ...` 触发入库任务
CMD ["sleep", "infinity"]
