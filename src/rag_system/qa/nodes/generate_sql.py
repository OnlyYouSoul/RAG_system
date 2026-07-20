"""metadata_qa 分支：让 LLM 把元数据问题翻译成一条只读 SELECT。

- 只暴露 documents / chunks 两张表的结构给 LLM。
- 产出经 utils.pg_query.validate_select 校验（单条 SELECT、无写关键字、自动补 LIMIT）。
- 若本轮带着上一次的 sql_error（execute_sql 回流），把错误与旧 SQL 一并给 LLM 重写。
- 生成或校验失败时写 error，路由据此兜底。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag_system.qa.llm import get_chat_model
from rag_system.qa.state import QAState
from rag_system.store.pg_query import SQLValidationError, validate_select

# 给 LLM 的表结构说明（与 utils/postgres_store.py 的建表一致）
_SCHEMA_DOC = """
表 documents（文档级元数据，一行一个文档）：
  document_id   TEXT   文档主键
  kb_id         TEXT   知识库 id
  title         TEXT   文档标题
  source_file   TEXT   源文件名
  source_path   TEXT   源文件路径
  doc_type      TEXT   文档类型（pdf/docx/pptx/xlsx/txt/md）
  file_hash     TEXT   文件内容 hash
  language      TEXT   语言（zh/en）
  department    TEXT   所属部门（如 infra/hr/finance/policy）
  ingested_at   TIMESTAMPTZ  入库时间
  total_chunks  INTEGER      该文档切出的 chunk 数

表 chunks（chunk 级元数据，一行一个 chunk，外键 document_id -> documents）：
  document_id   TEXT     所属文档
  chunk_id      INTEGER  文档内序号
  text          TEXT     chunk 正文
  chunk_index   INTEGER  段落序号
  chunk_size    INTEGER  字符数
  overlap_size  INTEGER  与上一 chunk 的重叠字符数
"""

_SYSTEM = (
    "你是一个把自然语言问题翻译成 PostgreSQL 查询的助手。\n"
    "只能查询下面给出的表，只能写单条只读 SELECT 语句（可用 WITH），"
    "严禁任何写操作（INSERT/UPDATE/DELETE/DDL 等）和多语句。\n"
    "生成规则（务必遵守）：\n"
    "1. 只 SELECT 回答问题真正需要的列，严禁 SELECT *。\n"
    "2. 文档属性类问题（创建时间/标题/类型/语言/部门/chunk 数等）只查 documents 表，"
    "不要查 chunks 表，也不要返回正文 text 列。\n"
    "3. 计数用 COUNT，「多少篇文档」用 COUNT(DISTINCT document_id) 或对 documents 计数；"
    "「多少 token/字数」等聚合针对相应数值列用 SUM。\n"
    "4. 时间筛选针对 documents.ingested_at（TIMESTAMPTZ）。\n"
    "5. 按名称找文档时用 title 或 source_file 做 ILIKE 模糊匹配。\n"
    "6. 只有确实要看 chunk 正文内容时才查 chunks.text，并务必带足够小的 LIMIT。\n"
    "只输出 SQL，不要解释。\n"
    f"当前时间：{{now}}\n\n表结构：\n{_SCHEMA_DOC}"
)

_HUMAN = "用户问题：{question}\n\n请写出对应的 SELECT 语句。"

_RETRY_HUMAN = (
    "用户问题：{question}\n\n"
    "上一次生成的 SQL 执行失败：\n{prev_sql}\n\n"
    "数据库报错：{error}\n\n"
    "请修正后重新写出一条正确的 SELECT 语句。"
)


class GeneratedSQL(BaseModel):
    sql: str = Field(description="一条只读 SELECT 语句，不带解释、不带分号结尾也可")


def _build_chain(is_retry: bool):
    human = _RETRY_HUMAN if is_retry else _HUMAN
    prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", human)])
    return prompt | get_chat_model().with_structured_output(GeneratedSQL)


def generate_sql(state: QAState) -> QAState:
    if state.get("error"):
        return {}

    question = state.get("search_query") or state["query"]
    prev_error = state.get("sql_error")
    attempts = int(state.get("sql_attempts", 0) or 0)
    is_retry = bool(prev_error)

    params = {"now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "question": question}
    if is_retry:
        params["prev_sql"] = state.get("generated_sql", "")
        params["error"] = prev_error

    # 生成/校验失败走 sql_error（而非顶层 error），交由路由决定重试还是兜底，
    # 不能短路整张图——否则 generate_answer 无法给出兜底话术。
    try:
        result: GeneratedSQL = _build_chain(is_retry).invoke(params)
    except Exception as exc:
        return {"sql_error": f"SQL 生成失败：{exc}", "sql_attempts": attempts + 1}

    try:
        safe_sql = validate_select(result.sql)
    except SQLValidationError as exc:
        return {
            "sql_error": f"SQL 未通过安全校验：{exc}",
            "generated_sql": result.sql,  # 保留原样供重试参考
            "sql_attempts": attempts + 1,
        }

    # 进入执行前清掉上一轮错误，标记本次尝试
    return {
        "generated_sql": safe_sql,
        "sql_error": "",
        "sql_attempts": attempts + 1,
    }
