from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class LegalSource(BaseModel):
    """A cited source document returned by vector search."""

    document_id: str
    title: str
    section: str
    retrieved_text: str
    relevance_score: float
    retrieved_at: datetime


class LegalResult(BaseModel):
    """Structured agent output — the agent's result_type."""

    answer: str
    sources_cited: list[LegalSource]
    confidence: str  # "high", "medium", or "low"
    reasoning: str


class DocumentChunk(BaseModel):
    """A chunk stored in Qdrant."""

    document_id: str
    title: str
    section: str
    text: str
    metadata: dict
