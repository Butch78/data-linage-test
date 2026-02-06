from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import LegalDeps, legal_agent
from .vectorstore import seed_collection

DATABASE = Path("legal_agent.db")


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DATABASE))
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_db()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS queries (
            query_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_query TEXT NOT NULL,
            result_json TEXT NOT NULL,
            messages_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );
        """
    )
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    await seed_collection()
    yield


app = FastAPI(title="Legal Agent PoC", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None


@app.post("/query")
async def post_query(req: QueryRequest) -> dict[str, Any]:
    session_id = req.session_id or str(uuid.uuid4())
    query_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_db()

    # Ensure session exists
    existing = conn.execute(
        "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not existing:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
            (session_id, now),
        )
        conn.commit()

    # Load previous message history for multi-turn context
    prev_rows = conn.execute(
        "SELECT messages_json FROM queries WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    ).fetchall()

    message_history = None
    if prev_rows:
        # Use messages from the most recent query as the conversation history
        last_messages_json = prev_rows[-1]["messages_json"]
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        message_history = ModelMessagesTypeAdapter.validate_json(last_messages_json)

    deps = LegalDeps(session_id=session_id, user_query=req.query)

    result = await legal_agent.run(
        req.query,
        deps=deps,
        message_history=message_history,
    )

    # Serialize new messages
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    new_messages_json = ModelMessagesTypeAdapter.dump_json(result.new_messages())

    # Persist query and result
    conn.execute(
        "INSERT INTO queries (query_id, session_id, user_query, result_json, messages_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            query_id,
            session_id,
            req.query,
            result.output.model_dump_json(),
            new_messages_json.decode(),
            now,
        ),
    )
    conn.commit()
    conn.close()

    return {
        "session_id": session_id,
        "query_id": query_id,
        "result": result.output.model_dump(),
    }


@app.get("/lineage/{session_id}")
async def get_lineage(session_id: str) -> dict[str, Any]:
    conn = _get_db()

    rows = conn.execute(
        "SELECT query_id, user_query, result_json, created_at "
        "FROM queries WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "session_id": session_id,
            "total_queries": 0,
            "unique_sources": 0,
            "queries": [],
            "all_sources": [],
        }

    queries = []
    seen_sources: dict[str, dict] = {}

    for row in rows:
        result_data = json.loads(row["result_json"])
        sources = result_data.get("sources_cited", [])

        for source in sources:
            doc_id = source["document_id"]
            if doc_id not in seen_sources:
                seen_sources[doc_id] = source

        queries.append(
            {
                "query_id": row["query_id"],
                "user_query": row["user_query"],
                "answer": result_data["answer"],
                "confidence": result_data["confidence"],
                "reasoning": result_data["reasoning"],
                "sources_cited": sources,
                "created_at": row["created_at"],
            }
        )

    return {
        "session_id": session_id,
        "total_queries": len(queries),
        "unique_sources": len(seen_sources),
        "queries": queries,
        "all_sources": list(seen_sources.values()),
    }


@app.get("/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    conn = _get_db()
    rows = conn.execute(
        "SELECT s.session_id, s.created_at, COUNT(q.query_id) as query_count "
        "FROM sessions s LEFT JOIN queries q ON s.session_id = q.session_id "
        "GROUP BY s.session_id ORDER BY s.created_at DESC"
    ).fetchall()
    conn.close()

    return [
        {
            "session_id": row["session_id"],
            "created_at": row["created_at"],
            "query_count": row["query_count"],
        }
        for row in rows
    ]


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return """<!DOCTYPE html>
<html>
<head><title>Legal Agent PoC</title></head>
<body>
<h1>Legal Agent PoC &mdash; Swiss Tenancy Law</h1>
<h2>Endpoints</h2>
<ul>
  <li><code>POST /query</code> &mdash; Submit a legal question</li>
  <li><code>GET /lineage/{session_id}</code> &mdash; View data lineage for a session</li>
  <li><code>GET /sessions</code> &mdash; List all sessions</li>
</ul>
<h2>Example</h2>
<pre>
curl -s -X POST http://localhost:8000/query \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Can a landlord terminate a lease after a tenant reports building defects?"}' \\
  | python -m json.tool
</pre>
</body>
</html>"""
