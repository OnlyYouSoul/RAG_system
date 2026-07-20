import argparse
import base64
import mimetypes
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

from minio import Minio
from minio.error import S3Error
from openai import OpenAI


from rag_system import config

PROJECT_DIR = config.PROJECT_ROOT
MINERU_SOURCE_DIR = config.MINERU_SOURCE_DIR

# 把 MinerU 源码加入 import 路径，直接调用解析函数（不再起 CLI 子进程/本地服务）
if str(MINERU_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(MINERU_SOURCE_DIR))

from mineru.cli.common import do_parse, read_fn  # noqa: E402

IMAGE_MD_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def str_to_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def run_mineru(pdf_path: Path, output_dir: Path, backend: str = "pipeline", lang: str = "ch") -> None:
    """直接调用 MinerU 源码把 PDF 解析为 Markdown。"""
    print(f"[MinerU] 源码目录：{MINERU_SOURCE_DIR}")
    print(f"[MinerU] 解析：{pdf_path}")
    do_parse(
        output_dir=str(output_dir),
        pdf_file_names=[pdf_path.stem],
        pdf_bytes_list=[read_fn(pdf_path)],
        p_lang_list=[lang],
        backend=backend,
    )
    print("[MinerU] 解析完成")


def find_markdown_file(output_dir: Path) -> Path:
    """递归寻找 MinerU 输出的 Markdown，取体积最大的作为主文档。"""
    md_files = sorted(output_dir.rglob("*.md"), key=lambda p: p.stat().st_size, reverse=True)
    if not md_files:
        raise FileNotFoundError(f"没有在输出目录中找到 Markdown 文件：{output_dir}")
    return md_files[0]


def init_minio_client() -> Minio:
    return Minio(
        os.getenv("MINIO_ENDPOINT", "127.0.0.1:9000"),
        access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
        secure=str_to_bool(os.getenv("MINIO_SECURE", "false")),
    )


def ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        print(f"[MinIO] 已创建 bucket：{bucket}")
    else:
        print(f"[MinIO] bucket 已存在：{bucket}")


def upload_image(client: Minio, bucket: str, image_path: Path, prefix: str) -> str:
    """上传图片到 MinIO，返回可公开访问的 URL。"""
    content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    object_name = f"{prefix}/{uuid.uuid4().hex}{image_path.suffix.lower() or '.png'}"
    client.fput_object(bucket, object_name, str(image_path), content_type=content_type)

    public_base = os.getenv("MINIO_PUBLIC_BASE_URL", "").rstrip("/")
    if not public_base:
        raise ValueError(
            "请在 .env 中配置 MINIO_PUBLIC_BASE_URL，例如：http://127.0.0.1:9000/mineru-images"
        )
    return f"{public_base}/{object_name}"


def generate_image_description(vision_client: OpenAI, image_path: Path) -> str:
    """调用视觉模型为图片生成一句中文简短说明。"""
    content_type = mimetypes.guess_type(str(image_path))[0] or "application/octet-stream"
    data_url = f"data:{content_type};base64,{base64.b64encode(image_path.read_bytes()).decode()}"

    response = vision_client.chat.completions.create(
        model=os.getenv("VISION_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请用中文为这张文档图片生成一句简短说明。"
                            "要求客观、简洁，不超过35个汉字。"
                            "不要编造图片中不存在的信息。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip().replace("\n", " ")


def resolve_image_path(md_file: Path, raw_link: str) -> Optional[Path]:
    """把 Markdown 中的图片链接解析成本地文件路径。"""
    link = raw_link.strip()
    if link.startswith(("http://", "https://", "data:")):
        return None

    candidate = Path(unquote(link).split("#")[0].split("?")[0])
    search = [candidate] if candidate.is_absolute() else [
        md_file.parent / candidate,
        md_file.parent.parent / candidate,
    ]
    for path in search:
        resolved = path.resolve()
        if resolved.exists():
            return resolved
    return None


def process_markdown_images(
    md_file: Path,
    minio_client: Minio,
    bucket: str,
    object_prefix: str,
    enable_vision: bool = True,
) -> Path:
    """上传图片到 MinIO、替换链接、写回视觉描述，输出新的 md 文件。"""
    markdown = md_file.read_text(encoding="utf-8")
    vision_client = init_vision_client() if enable_vision else None
    cache: dict[Path, tuple[str, str]] = {}  # 本地路径 -> (minio_url, description)

    def replace_match(match: re.Match) -> str:
        alt_text = match.group(1).strip()
        image_path = resolve_image_path(md_file, match.group(2))
        if image_path is None:
            return match.group(0)

        if image_path not in cache:
            print(f"[Image] 处理图片：{image_path}")
            url = upload_image(minio_client, bucket, image_path, object_prefix)
            description = ""
            if vision_client is not None:
                try:
                    description = generate_image_description(vision_client, image_path)
                    print(f"[Vision] 图片描述：{description}")
                except Exception as exc:
                    description = f"图片描述生成失败：{exc}"
            cache[image_path] = (url, description)

        url, description = cache[image_path]
        new_image_md = f"![{alt_text or description or 'image'}]({url})"
        if description:
            return f"{new_image_md}\n\n> 图片说明：{description}\n"
        return new_image_md

    new_markdown = IMAGE_MD_PATTERN.sub(replace_match, markdown)
    output_md = md_file.with_name(md_file.stem + "_minio_vision.md")
    output_md.write_text(new_markdown, encoding="utf-8")
    return output_md


def init_vision_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("VISION_API_KEY"),
        base_url=os.getenv("VISION_BASE_URL", "https://api.openai.com/v1"),
    )


def main() -> None:
    config.load_env()

    parser = argparse.ArgumentParser(description="MinerU 解析 PDF + 图片上传 MinIO + 视觉描述")
    parser.add_argument("--pdf", required=True, help="输入 PDF 文件路径")
    parser.add_argument("--out", default=str(config.OUTPUT_DIR), help="MinerU 输出目录")
    parser.add_argument("--backend", default="pipeline", help="MinerU 后端，CPU 环境建议 pipeline")
    parser.add_argument("--lang", default="ch", help="OCR 语言，中文用 ch")
    parser.add_argument("--prefix", default=None, help="上传到 MinIO 的对象前缀")
    parser.add_argument("--no-vision", action="store_true", help="只上传图片并替换链接，不调用视觉模型")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF 文件不存在：{pdf_path}")

    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bucket = os.getenv("MINIO_BUCKET", "mineru-images")

    run_mineru(pdf_path, output_dir, backend=args.backend, lang=args.lang)

    md_file = find_markdown_file(output_dir)
    print(f"[Markdown] 找到 Markdown 文件：{md_file}")

    minio_client = init_minio_client()
    ensure_bucket(minio_client, bucket)

    final_md = process_markdown_images(
        md_file=md_file,
        minio_client=minio_client,
        bucket=bucket,
        object_prefix=args.prefix or pdf_path.stem,
        enable_vision=not args.no_vision,
    )

    print("\n处理完成")
    print(f"最终 Markdown 文件：{final_md}")


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        raise RuntimeError(f"MinIO 操作失败：{exc}") from exc
