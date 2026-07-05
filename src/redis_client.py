import json
import os
import uuid
import redis.asyncio as redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

redis_client = redis.from_url(
    REDIS_URL,
    decode_responses=True,
)


async def create_parse_task(file_hash: str) -> str:
    task_id = str(uuid.uuid4())

    await redis_client.hset(
        f"task:{task_id}",
        mapping={
            "status": "pending",
            "file_hash": file_hash,
            "progress": "0",
            "message": "waiting",
        },
    )

    await redis_client.expire(f"task:{task_id}", 86400)

    return task_id

async def get_task_status(task_id: str):
    return await redis_client.hgetall(f"task:{task_id}")


async def push_parse_task(task_id: str, file_path: str, file_hash: str):
    payload = {
        "task_id": task_id,
        "file_path": file_path,
        "file_hash": file_hash,
    }

    await redis_client.lpush("queue:pdf_parse", json.dumps(payload))


async def worker_loop():
    while True:
        item = await redis_client.brpop("queue:pdf_parse", timeout=0)
        _, raw = item

        task = json.loads(raw)
        task_id = task["task_id"]

        await redis_client.hset(f"task:{task_id}", mapping={
            "status": "running",
            "progress": "10",
            "message": "parsing pdf",
        })

        try:
            # 这里调用你的 MinerU 解析、chunk、embedding、入库逻辑
            # await parse_and_index_pdf(task["file_path"])

            await redis_client.hset(f"task:{task_id}", mapping={
                "status": "done",
                "progress": "100",
                "message": "completed",
            })

        except Exception as e:
            await redis_client.hset(f"task:{task_id}", mapping={
                "status": "failed",
                "message": str(e),
            })


async def acquire_file_lock(file_hash: str) -> bool:
    return await redis_client.set(
        f"lock:file:{file_hash}",
        "1",
        nx=True,
        ex=3600,
    )