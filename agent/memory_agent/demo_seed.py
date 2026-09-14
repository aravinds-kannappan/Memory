from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from memory_agent.capture import Capturer
from memory_agent.search.embeddings import embed_text
from memory_agent.store import MemoryStore, utcnow


DEMO_PAGES = [
    {
        "url": "https://demo.browsermemory.local/suppliers/northwind-parts",
        "title": "Northwind Parts — Wholesale fasteners",
        "domain": "demo.browsermemory.local",
        "text": (
            "Northwind Parts supplies industrial fasteners for mid-size manufacturers. "
            "Minimum order: 500 units on standard hex bolts. "
            "Lead time averages 8 business days. Free freight over $2,000. "
            "Contact: orders@northwind-parts.example."
        ),
        "hours_ago": 14,
    },
    {
        "url": "https://demo.browsermemory.local/suppliers/harbor-alloy",
        "title": "Harbor Alloy — Metal components catalog",
        "domain": "demo.browsermemory.local",
        "text": (
            "Harbor Alloy lists CNC-machined brackets and plates. "
            "Minimum order: 50 units. Rush production available. "
            "MOQ waived for sample kits under 10 pieces."
        ),
        "hours_ago": 13,
    },
    {
        "url": "https://demo.browsermemory.local/suppliers/summit-packaging",
        "title": "Summit Packaging — Corrugated & mailers",
        "domain": "demo.browsermemory.local",
        "text": (
            "Summit Packaging offers recycled mailers and custom inserts. "
            "Minimum order: 1,000 units for printed boxes. "
            "Delivery to West Coast warehouses in 5 days."
        ),
        "hours_ago": 12,
    },
    {
        "url": "https://demo.browsermemory.local/notes/comparison-grid",
        "title": "Supplier comparison notes",
        "domain": "demo.browsermemory.local",
        "text": (
            "Comparing Northwind, Harbor Alloy, and Summit. "
            "Need to check delivery times tomorrow. "
            "Northwind stands out for the 500-unit minimum on fasteners."
        ),
        "hours_ago": 11,
    },
]


def seed_demo_memory(store: MemoryStore) -> dict[str, Any]:
    """Seed the flagship demo: find the supplier with MOQ 500, then resume."""
    page_ids = []
    now = datetime.now(timezone.utc)
    for page in DEMO_PAGES:
        when = (now - timedelta(hours=page["hours_ago"])).isoformat()
        page_id = store.upsert_page(page["url"], title=page["title"], domain=page["domain"])
        emb = embed_text(page["text"])
        snap_id, _ = store.save_snapshot(
            page_id,
            text=page["text"],
            title=page["title"],
            domain=page["domain"],
            embedding=emb,
            extraction_ok=True,
            captured_at=when,
        )
        store.record_visit(page_id, when=when, source="demo", snapshot_id=snap_id)
        page_ids.append(page_id)

    ws = store.create_workspace(
        name="Supplier comparison",
        summary="Comparing Northwind Parts, Harbor Alloy, and Summit Packaging across four tabs.",
        next_step="Tomorrow, compare delivery times.",
        page_ids=page_ids,
    )
    return {
        "pages": len(page_ids),
        "workspace_id": ws["id"],
        "workspace_name": ws["name"],
        "flagship_query": "What was that supplier with the minimum order of 500?",
    }
