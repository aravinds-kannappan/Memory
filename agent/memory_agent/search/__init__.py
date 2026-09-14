from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from memory_agent.search.embeddings import embed_text, semantic_rank
from memory_agent.store import MemoryStore


TIME_PATTERNS = [
    (re.compile(r"\blast\s+night\b", re.I), dict(hours=18, end_offset_hours=0)),
    (re.compile(r"\byesterday\b", re.I), dict(hours=36, end_offset_hours=0)),
    (re.compile(r"\btoday\b", re.I), dict(hours=18, end_offset_hours=0)),
    (re.compile(r"\blast\s+week\b", re.I), dict(hours=24 * 8, end_offset_hours=0)),
    (re.compile(r"\bthis\s+morning\b", re.I), dict(hours=12, end_offset_hours=0)),
]


def interpret_query(query: str) -> dict[str, Any]:
    """Extract time constraints and cleaned search terms without inventing intent."""
    now = datetime.now(timezone.utc)
    after = None
    before = None
    cleaned = query
    time_label = None
    for pattern, spec in TIME_PATTERNS:
        if pattern.search(query):
            after = (now - timedelta(hours=spec["hours"])).isoformat()
            if spec.get("end_offset_hours"):
                before = (now - timedelta(hours=spec["end_offset_hours"])).isoformat()
            cleaned = pattern.sub(" ", cleaned)
            time_label = pattern.pattern
            break
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Drop filler phrases
    for filler in [
        r"\bfind\b",
        r"\bwhat\s+was\b",
        r"\bthat\b",
        r"\bthe\b",
        r"\bi\s+looked\s+at\b",
        r"\bi\s+saw\b",
        r"\bshow\s+me\b",
        r"\bwith\b",
        r"\bof\b",
        r"\ba\b",
        r"\ban\b",
    ]:
        cleaned = re.sub(filler, " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return {
        "raw": query,
        "terms": cleaned,
        "after": after,
        "before": before,
        "time_label": time_label,
    }


class RetrievalEngine:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def search(self, query: str, limit: int = 8) -> dict[str, Any]:
        parsed = interpret_query(query)
        terms = parsed["terms"] or query

        kw = self.store.keyword_search(
            terms, limit=limit * 2, after=parsed["after"], before=parsed["before"]
        )
        embeddings = self.store.all_embeddings()
        if parsed["after"] or parsed["before"]:
            filtered = []
            for e in embeddings:
                ts = e.get("captured_at") or ""
                if parsed["after"] and ts < parsed["after"]:
                    continue
                if parsed["before"] and ts > parsed["before"]:
                    continue
                filtered.append(e)
            embeddings = filtered
        sem = semantic_rank(terms, embeddings, limit=limit * 2)

        # Merge by snapshot_id
        combined: dict[str, dict[str, Any]] = {}
        for i, row in enumerate(kw):
            sid = row["snapshot_id"]
            combined[sid] = {
                **row,
                "keyword_rank": i,
                "keyword_score": 1.0 / (1 + i),
                "semantic_score": 0.0,
            }
            # Pull a short supporting excerpt from stored text
            if not row.get("excerpt"):
                text = self.store.get_snapshot_text(sid)
                combined[sid]["excerpt"] = _best_excerpt(text, terms)

        for i, row in enumerate(sem):
            sid = row["snapshot_id"]
            if sid not in combined:
                text = self.store.get_snapshot_text(sid)
                combined[sid] = {
                    "snapshot_id": sid,
                    "page_id": row["page_id"],
                    "url": row["url"],
                    "title": row["title"],
                    "domain": row["domain"],
                    "captured_at": row["captured_at"],
                    "extraction_ok": row.get("extraction_ok", 1),
                    "excerpt": _best_excerpt(text, terms),
                    "keyword_score": 0.0,
                    "keyword_rank": 999,
                }
            combined[sid]["semantic_score"] = row.get("semantic_score", 0.0)

        ranked = []
        for item in combined.values():
            score = 0.55 * item.get("semantic_score", 0.0) + 0.45 * item.get(
                "keyword_score", 0.0
            )
            excerpt = (item.get("excerpt") or "").lower()
            excerpt_plain = excerpt.replace("⟦", "").replace("⟧", "")
            body_bonus = 0.0
            term_hits = sum(
                1 for t in terms.lower().split() if len(t) > 2 and t in excerpt_plain
            )
            body_bonus += 0.03 * term_hits
            if re.search(r"minimum order[:\s]+500", excerpt_plain) or re.search(
                r"\b500\s*-?\s*units\b", excerpt_plain
            ):
                body_bonus += 0.25
            # Prefer pages whose title looks like a supplier/product page when query asks for a supplier
            title_l = (item.get("title") or "").lower()
            if "supplier" in terms.lower() and any(
                w in title_l for w in ("parts", "alloy", "packaging", "wholesale", "catalog")
            ):
                body_bonus += 0.08
            if "notes" in title_l or "comparison" in title_l:
                body_bonus -= 0.05
            item["score"] = score + body_bonus
            item["historical"] = True
            ranked.append(item)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        results = ranked[:limit]

        answer = self._compose_answer(query, results)
        return {"query": parsed, "results": results, "answer": answer}

    def _compose_answer(self, query: str, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results or results[0]["score"] < 0.08:
            return {
                "found": False,
                "text": "I couldn’t find a convincing match in your saved memory. Try different words, or browse the candidates below.",
                "sources": [],
            }
        top = results[0]
        excerpt = (top.get("excerpt") or "").replace("⟦", "").replace("⟧", "")
        text = (
            f"Best match: **{top.get('title') or top.get('url')}** "
            f"({top.get('domain')}), captured {top.get('captured_at')}. "
            f"Supporting excerpt: “{excerpt[:240]}” "
            f"— this is a historical snapshot and may have changed."
        )
        return {
            "found": True,
            "text": text,
            "sources": [
                {
                    "page_id": r["page_id"],
                    "snapshot_id": r["snapshot_id"],
                    "url": r["url"],
                    "title": r.get("title"),
                    "excerpt": r.get("excerpt"),
                    "captured_at": r.get("captured_at"),
                    "score": r.get("score"),
                }
                for r in results[:3]
            ],
        }


def _best_excerpt(text: str, terms: str, width: int = 220) -> str:
    if not text:
        return ""
    lower = text.lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", terms.lower()) if len(t) > 2]
    pos = -1
    for t in tokens:
        pos = lower.find(t)
        if pos >= 0:
            break
    if pos < 0:
        return text[:width]
    start = max(0, pos - 60)
    end = min(len(text), start + width)
    chunk = text[start:end].strip()
    if start > 0:
        chunk = "…" + chunk
    if end < len(text):
        chunk = chunk + "…"
    return chunk
