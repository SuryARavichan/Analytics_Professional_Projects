"""
text_to_sql.py — the core QueryMind pipeline: natural-language question ->
LLM-generated SQL -> safety-checked execution -> LLM-generated narrative
insight.

Three deliberate design choices worth calling out in an interview:
  1. Schema-aware prompting (see schema.py) — the model sees real table
     structure and sample rows, not just column names.
  2. A hard safety guardrail BEFORE execution — the model's SQL is parsed
     and rejected if it contains anything other than a single read-only
     SELECT. This is non-negotiable for any "let an LLM touch a database"
     system, and the query also runs against a read-only SQLite connection
     as defense in depth.
  3. Two-stage LLM use — one call writes SQL, a second call (given the
     *result* of that SQL) writes the plain-English insight. Splitting
     these means the insight is always grounded in the real query result,
     not a hallucinated guess at what the data probably says.
"""
from __future__ import annotations
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "shared"))
from llm_client import LLMClient, LLMResponse  # noqa: E402

FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REPLACE|TRUNCATE)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are PathfinderAI, a SQL analyst copilot for a product-led-growth SaaS company's "
    "SQLite database (Flowdesk, a team-collaboration app). Given the schema below and a "
    "business question in plain English, respond with ONLY a single, valid, read-only "
    "SQLite SELECT statement that answers it. No prose, no markdown code fences, no "
    "explanation -- SQL only. Never write INSERT/UPDATE/DELETE/DROP or any statement "
    "that isn't a SELECT."
)


def extract_sql(raw: str) -> str:
    """Strips markdown fences / stray prose a model might still add despite instructions."""
    text = raw.strip()
    text = re.sub(r"^```sql\s*|```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    # If the model added a leading explanation line, keep from the first SELECT/WITH onward.
    match = re.search(r"(SELECT|WITH)\b", text, re.IGNORECASE)
    return text[match.start():].strip() if match else text


def is_safe_select(sql: str) -> tuple[bool, str]:
    if FORBIDDEN.search(sql):
        return False, "Rejected: query contains a non-read-only statement."
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        return False, "Rejected: multiple statements are not allowed."
    if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
        return False, "Rejected: query must start with SELECT or WITH."
    return True, ""


@dataclass
class QueryResult:
    question: str
    sql: str
    is_safe: bool
    safety_note: str
    columns: list
    rows: list
    error: str | None
    llm_meta: dict


def ask_database(question: str, db_path: str, schema_desc: str, client: LLMClient,
                  prompt_id: str | None = None) -> QueryResult:
    prompt = f"SCHEMA:\n{schema_desc}\n\nQUESTION: {question}\n\nSQL:"
    resp: LLMResponse = client.complete(prompt=prompt, system=SYSTEM_PROMPT, prompt_id=prompt_id)
    sql = extract_sql(resp.text)

    safe, note = is_safe_select(sql)
    columns, rows, error = [], [], None

    if safe:
        try:
            # read-only connection: defense in depth even though the safety
            # regex already blocks write statements
            uri = f"file:{Path(db_path).resolve()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True)
            cur = conn.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            columns = [d[0] for d in cur.description] if cur.description else []
            conn.close()
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
    else:
        error = note

    return QueryResult(
        question=question, sql=sql, is_safe=safe, safety_note=note,
        columns=columns, rows=rows, error=error,
        llm_meta={"backend": resp.backend, "model": resp.model, "cached": resp.cached,
                   "latency_ms": round(resp.latency_ms, 1)},
    )


INSIGHT_SYSTEM_PROMPT = (
    "You are a data analyst writing a one-to-two sentence executive insight from a query "
    "result. Be specific with numbers. Do not restate the question; state the finding and, "
    "if relevant, why it matters. No hedging language."
)


def generate_insight(question: str, columns: list, rows: list, client: LLMClient,
                      prompt_id: str | None = None) -> LLMResponse:
    preview = ", ".join(columns) + "\n" + "\n".join(str(r) for r in rows[:15])
    prompt = f"QUESTION: {question}\n\nQUERY RESULT ({len(rows)} rows):\n{preview}"
    return client.complete(prompt=prompt, system=INSIGHT_SYSTEM_PROMPT, prompt_id=prompt_id)
