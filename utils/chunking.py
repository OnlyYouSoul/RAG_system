import re
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: int
    text: str
    metadata: dict

def read_md(file_path: str) -> str:
    """读取 Markdown 文件"""
    return Path(file_path).read_text(encoding="utf-8")


def split_markdown_paragraphs(text: str) -> list[str]:
    
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = re.split(r"\n\s*\n+", text)

    return [p.strip() for p in paragraphs if p.strip()]


def split_sentences(text: str) -> list[str]:
    """
    按中英文句号、问号、感叹号等切分句子
    """
    pattern = r"(?<=[。！？!?\.])\s*"
    sentences = re.split(pattern, text)

    return [s.strip() for s in sentences if s.strip()]


def add_overlap(chunks: list[str], overlap_size: int) -> list[str]:
    """
    给相邻 chunk 添加 overlap。
    overlap_size 表示从上一个 chunk 末尾取多少字符拼到当前 chunk 前面。
    """
    if overlap_size <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]

    for i in range(1, len(chunks)):
        overlap_text = chunks[i - 1][-overlap_size:]
        new_chunk = overlap_text + "\n" + chunks[i]
        result.append(new_chunk)

    return result


def split_long_paragraph(
    paragraph: str,
    chunk_size: int
) -> list[str]:
    """
    如果单个段落过长，则按句子继续切分。
    """
    sentences = split_sentences(paragraph)

    chunks = []
    current = ""

    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + len(sentence) <= chunk_size:
            current += sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def chunk_text_by_paragraph(
    file_path: str,
    chunk_size: int = 800,
    overlap_size: int = 100,
    doc_metadata: dict | None = None,
) -> list[Chunk]:
    """按段落切分 Markdown，并把文档级 metadata 合并进每个 chunk。

    doc_metadata：文档级字段（document_id / kb_id / title 等），
    由 metadata.build_document_metadata 构建，整篇文档共享。
    """
    doc_metadata = doc_metadata or {}

    text = read_md(file_path)
    paragraphs = split_markdown_paragraphs(text)

    raw_chunks = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            raw_chunks.append(paragraph)
        else:
            sentence_chunks = split_long_paragraph(paragraph, chunk_size)
            raw_chunks.extend(sentence_chunks)

    overlapped_chunks = add_overlap(raw_chunks, overlap_size)

    chunks = []

    for idx, chunk_text in enumerate(overlapped_chunks):
        chunks.append(
            Chunk(
                chunk_id=idx,
                text=chunk_text,
                metadata={
                    **doc_metadata,
                    "chunk_index": idx,
                    "chunk_size": len(chunk_text),
                    "overlap_size": overlap_size,
                },
            )
        )

    return chunks