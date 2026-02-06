from __future__ import annotations

import os

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION_NAME = "legal_documents"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

SEED_DOCUMENTS = [
    {
        "document_id": "or-art-271",
        "title": "OR Art. 271 — Protection Against Termination",
        "section": "Art. 271 OR (Code of Obligations)",
        "text": (
            "A termination of a residential or commercial lease may be contested "
            "if it contravenes the principle of good faith. In particular, a "
            "termination is contestable if given because the tenant asserts claims "
            "arising from the lease in good faith, because the tenant is affiliated "
            "with an organisation that represents the interests of tenants, or "
            "during proceedings related to the lease. The burden of proof for "
            "retaliatory motive rests with the tenant."
        ),
    },
    {
        "document_id": "or-art-271a",
        "title": "OR Art. 271a — Annulment of Termination",
        "section": "Art. 271a OR (Code of Obligations)",
        "text": (
            "A termination by the landlord is annullable if given during or within "
            "three years after the conclusion of conciliation or court proceedings "
            "related to the tenancy, unless the proceedings were initiated in a "
            "frivolous manner. A termination is also annullable if given in "
            "retaliation for the tenant exercising rights under the lease, such as "
            "demanding repairs or reporting defects to the authorities. The "
            "three-year protection period begins from the date of final judgment."
        ),
    },
    {
        "document_id": "or-art-259a",
        "title": "OR Art. 259a — Tenant Remedies for Defects",
        "section": "Art. 259a OR (Code of Obligations)",
        "text": (
            "If a defect arises during the tenancy that the tenant is not obliged "
            "to remedy and that cannot be attributed to the tenant, the tenant may "
            "demand that the landlord remedy the defect, reduce the rent in "
            "proportion to the defect, or claim damages. For minor defects, the "
            "tenant must notify the landlord and allow reasonable time for repair. "
            "For serious defects affecting habitability, the tenant may deposit "
            "rent with the authorities and seek immediate remediation."
        ),
    },
    {
        "document_id": "bger-4a-123-2024",
        "title": "BGer 4A_123/2024 — Federal Supreme Court",
        "section": "Judgment of 15 March 2024",
        "text": (
            "The Federal Supreme Court held that a termination notice served "
            "within six months of the tenant filing a complaint about mould and "
            "structural dampness was retaliatory within the meaning of Art. 271a "
            "OR. The landlord failed to demonstrate a legitimate economic interest "
            "independent of the complaint. The court emphasised that temporal "
            "proximity between a tenant's assertion of rights and a subsequent "
            "termination creates a strong presumption of retaliation that the "
            "landlord must rebut with concrete evidence."
        ),
    },
    {
        "document_id": "bger-4a-456-2023",
        "title": "BGer 4A_456/2023 — Federal Supreme Court",
        "section": "Judgment of 22 November 2023",
        "text": (
            "In this case the Federal Supreme Court examined whether a planned "
            "comprehensive renovation constituted a valid ground for termination "
            "under Art. 271 OR. The court ruled that renovation alone does not "
            "justify termination unless the landlord proves that continued "
            "occupancy would render the renovation substantially more expensive "
            "or technically impossible. Mere inconvenience to the construction "
            "schedule is insufficient. The tenant's long occupancy of 18 years "
            "was a factor weighing against termination."
        ),
    },
    {
        "document_id": "zh-mietgericht-2024-31",
        "title": "ZH Mietgericht 2024/31 — Zurich Rental Court",
        "section": "Decision of 8 July 2024",
        "text": (
            "The Zurich Rental Court granted the tenant's petition to annul a "
            "termination notice issued after the tenant reported persistent water "
            "ingress and requested a rent reduction. The court found that the "
            "landlord's stated reason — personal use by a family member — was "
            "pretextual, as no concrete plans for personal use were presented. "
            "The court awarded the tenant an extension of the lease by two years "
            "and ordered the landlord to remedy the reported defects within 90 "
            "days."
        ),
    },
]


def get_qdrant_client() -> QdrantClient:
    """Return a QdrantClient pointed at the QDRANT_URL environment variable."""
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


async def get_embedding(text: str) -> list[float]:
    """Get an embedding vector from OpenAI's text-embedding-3-small model."""
    client = openai.AsyncOpenAI()
    response = await client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


async def seed_collection() -> None:
    """Delete and recreate the collection, embed all seed documents, and upsert."""
    client = get_qdrant_client()

    # Recreate collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    # Embed and upsert all seed documents
    points = []
    for i, doc in enumerate(SEED_DOCUMENTS):
        embedding = await get_embedding(doc["text"])
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "document_id": doc["document_id"],
                    "title": doc["title"],
                    "section": doc["section"],
                    "text": doc["text"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Seeded {len(points)} documents into Qdrant")


async def search_documents(query: str, limit: int = 3) -> list[dict]:
    """Embed the query and search Qdrant for similar documents."""
    client = get_qdrant_client()
    query_embedding = await get_embedding(query)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    )

    documents = []
    for point in results.points:
        documents.append(
            {
                "document_id": point.payload["document_id"],
                "title": point.payload["title"],
                "section": point.payload["section"],
                "text": point.payload["text"],
                "score": point.score,
            }
        )

    return documents
