# Docker 部署

整套栈由上级目录的 `docker-compose.yml` 编排：

| 服务 | 说明 | 端口 |
| --- | --- | --- |
| `aistor-server` | AIStor 对象存储（S3 兼容），Milvus 与 app 共用 | 9000(API) / 9001(控制台) |
| `etcd` | Milvus 元数据 | — |
| `milvus-standalone` | 向量库 | 19530(gRPC) / 9091(健康) |
| `redis` | 任务队列 / 文件锁 | 6379 |
| `postgres` | 关系库 | 5432 |
| `app` | 本项目入库流水线 | — |

## 前置

- Docker + Docker Compose v2
- 构建 context 是上级目录 `/home/zhang/project`，因为 `app` 依赖同级的 `../MinerU`（可编辑依赖）。
- `.env` 里的外部模型（`VISION_*` / `EMBEDDING_*`）需要可用的 API Key；compose 会自动把 `MINIO_ENDPOINT` / `MILVUS_URI` / `REDIS_URL` 覆盖为容器网络内的服务名。

### AIStor 镜像与 License（重要）

`docker-compose.yml` 默认使用 `quay.io/minio/aistor/minio:latest`（AIStor 企业版）。按[官方容器部署文档](https://docs.min.io/aistor/installation/container/install/)：

1. **登录授权 registry** 才能拉取企业版镜像：`docker login quay.io`（使用 MinIO 提供的授权凭据）。
2. **License 必需**：AIStor 通过 `--license /minio.license` 读取 License **文件**（不是环境变量）。把 License 下载到 `RAG_System/minio.license`，compose 已把它只读挂载进容器（`./RAG_System/minio.license -> /minio.license`）。
   > 没有该文件时 `docker compose up` 会把挂载点当目录创建导致启动失败，务必先放好 License 文件。
3. 数据目录为 `/mnt/data`（对应 `aistor-data` 卷），启动命令 `minio server /mnt/data --console-address ":9001" --license /minio.license`。
4. 默认凭据 `minioadmin:minioadmin`，compose 已用 `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` 覆盖为 `admin` / `88888888`。

**没有 AIStor 授权？** 改用社区版，替换 `aistor-server` 的镜像与命令，并去掉 License 挂载：

```yaml
image: minio/minio:latest
command: server /data --console-address ":9001"
volumes:
  - aistor-data:/data     # 社区版数据目录为 /data
```

## 启动

```bash
cd /home/zhang/project
docker compose up -d --build
docker compose ps          # 等待 milvus / aistor 变为 healthy
```

首次启动会构建 `app` 镜像（安装 MinerU 及依赖），耗时较长。

## 初始化对象存储 bucket

app 首次上传图片时会自动创建 `mineru` bucket，但图片 URL 要能公开访问，需把 bucket 设为可下载（匿名读）。用 AIStor 控制台（http://localhost:9001，admin / 88888888）或 mc：

```bash
docker run --rm --network project_default \
  --entrypoint sh minio/mc -c "\
  mc alias set s3 http://aistor-server:9000 admin 88888888 && \
  mc mb -p s3/mineru && mc anonymous set download s3/mineru"
```

> 网络名以 `docker compose ps` / `docker network ls` 实际输出为准（通常是 `<目录名>_default`）。

## 运行入库任务

`app` 常驻（`sleep infinity`），通过 exec 触发流水线：

```bash
docker compose exec app \
  uv run rag ingest \
    --pdf data/pdfs/compile_env.pdf \
    --kb-id kb_demo \
    --department infra \
    --to-milvus
```

把要处理的 PDF 放到 `RAG_System/data/pdfs/` 下（已挂载 `./data`），或调整 `--pdf` 路径。产物写入挂载出来的 `output/`。

## 常见问题

- **Milvus 一直 unhealthy**：等 `start_period`（90s）过后再看；确认 `aistor-server` 已 healthy 且 `milvus-bucket` 可写。
- **图片 URL 打不开**：bucket 未设匿名读，见上面的 `mc anonymous set download`。
- **拉取 AIStor 失败**：改用社区版 `minio/minio:latest`。
- **MinerU 首次跑很慢**：需要下载模型权重，已挂载 `mineru-models` 卷做缓存，二次运行会快。
