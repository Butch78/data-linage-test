from __future__ import annotations

from dataclasses import dataclass

import logfire
from pydantic_ai import Agent, RunContext

from .models import LegalResult
from .vectorstore import search_documents

logfire.configure(send_to_logfire="if-token-present")
logfire.instrument_pydantic_ai()


@dataclass
class LegalDeps:
    session_id: str
    user_query: str


legal_agent = Agent(
    "openai:gpt-4o",
    result_type=LegalResult,
    deps_type=LegalDeps,
    system_prompt=(
        "You are a Swiss legal research assistant specialising in tenancy law "
        "(Mietrecht). Your role is to provide accurate, well-sourced legal analysis "
        "based on Swiss federal statutes and court decisions.\n\n"
        "Rules:\n"
        "1. Always cite documents returned by your search tools. Never fabricate "
        "citations or refer to documents you have not retrieved.\n"
        "2. Rate your confidence as 'high' if multiple authoritative sources agree, "
        "'medium' if sources are limited or partially relevant, and 'low' if the "
        "question falls outside the available materials.\n"
        "3. Follow this research order: first search case law for relevant court "
        "decisions, then search statutes for applicable legal provisions, then "
        "synthesise your findings into a coherent legal analysis.\n"
        "4. In your reasoning field, explain the legal reasoning chain: which "
        "provisions apply, how courts have interpreted them, and how they bear on "
        "the user's question.\n"
        "5. When citing sources, include the exact document_id, title, and section "
        "from the search results."
    ),
)


@legal_agent.tool
async def search_case_law(ctx: RunContext[LegalDeps], query: str) -> list[dict]:
    """Search for relevant court decisions in Swiss tenancy law.

    Args:
        query: The legal question or topic to search for in case law.
    """
    results = await search_documents(query, limit=3)
    # Filter to court decisions only
    return [
        r
        for r in results
        if "BGer" in r["title"] or "Mietgericht" in r["title"]
    ]


@legal_agent.tool
async def search_statutes(ctx: RunContext[LegalDeps], query: str) -> list[dict]:
    """Search for relevant Swiss statutory provisions in tenancy law.

    Args:
        query: The legal question or topic to search for in statutes.
    """
    results = await search_documents(query, limit=3)
    # Filter to statutes only
    return [r for r in results if "OR Art" in r["title"]]
