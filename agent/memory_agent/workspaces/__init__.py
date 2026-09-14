from __future__ import annotations

import re
from collections import Counter
from typing import Any
from urllib.parse import urlparse

from memory_agent.capture.cdp import ChromeCDP
from memory_agent.store import MemoryStore


def suggest_workspace(
    tabs: list[dict[str, Any]], note: str = ""
) -> dict[str, str]:
    """Propose a workspace name + summary from tab titles/domains — no invented commitments."""
    if not tabs:
        return {
            "name": "Saved place",
            "summary": "Empty workspace.",
        }
    domains = [t.get("domain") or urlparse(t.get("url", "")).netloc for t in tabs]
    domain_counts = Counter(d for d in domains if d)
    top_domain = domain_counts.most_common(1)[0][0] if domain_counts else "browsing"
    titles = [t.get("title") or "" for t in tabs]
    keywords = Counter()
    for title in titles:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", title):
            if tok.lower() in {"the", "and", "for", "with", "from", "http", "https"}:
                continue
            keywords[tok.lower()] += 1
    top_words = [w for w, _ in keywords.most_common(3)]
    if top_words:
        name = " ".join(w.capitalize() for w in top_words[:2])
        if len(name) < 4:
            name = f"{top_domain} research"
    else:
        name = f"{top_domain} session"

    summary = (
        f"{len(tabs)} related tabs across {len(domain_counts)} site"
        f"{'s' if len(domain_counts) != 1 else ''}"
        f" (top: {top_domain})."
    )
    if note.strip():
        # User intent stays separate — only reflected as next_step, not as observed fact
        pass
    return {"name": name[:80], "summary": summary}


class WorkspaceService:
    def __init__(self, store: MemoryStore, cdp: ChromeCDP | None = None) -> None:
        self.store = store
        self.cdp = cdp or ChromeCDP()

    def propose_from_open_tabs(self, next_step: str = "") -> dict[str, Any]:
        tabs = self.cdp.list_tabs() if self.cdp.available() else []
        # Fall back to recent pages if CDP offline
        if not tabs:
            recent = self.store.recent_pages(limit=12)
            tabs = [
                {
                    "id": None,
                    "title": p["title"],
                    "url": p["url"],
                    "domain": p["domain"],
                    "page_id": p["id"],
                }
                for p in recent
            ]
            source = "recent_memory"
        else:
            source = "open_tabs"
            for t in tabs:
                page = self.store.upsert_page(
                    t["url"], title=t.get("title") or "", domain=t.get("domain") or ""
                )
                t["page_id"] = page

        suggestion = suggest_workspace(tabs, note=next_step)
        return {
            "source": source,
            "suggestion": suggestion,
            "next_step": next_step,
            "tabs": tabs,
            "cdp_available": self.cdp.available(),
        }

    def save(
        self,
        name: str,
        summary: str,
        next_step: str,
        page_ids: list[str],
        keep_open_page_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        ws = self.store.create_workspace(
            name=name,
            summary=summary,
            next_step=next_step,
            page_ids=page_ids,
            keep_open=set(keep_open_page_ids or []),
        )
        return ws

    def close_approved_tabs(
        self, tab_ids: list[str], workspace_id: str | None = None
    ) -> dict[str, Any]:
        if not self.cdp.available():
            return {
                "ok": False,
                "reason": "chrome_cdp_unavailable",
                "hint": "Start Chrome with remote debugging to allow tab close.",
                "closed": [],
            }
        closed = []
        errors = []
        for tid in tab_ids:
            result = self.cdp.close_tab(tid)
            if result.get("ok"):
                closed.append(tid)
            else:
                errors.append({"id": tid, "error": result.get("error")})
        if workspace_id and closed:
            self.store.mark_workspace_closed(workspace_id)
        return {"ok": True, "closed": closed, "errors": errors}

    def resume(self, workspace_id: str) -> dict[str, Any]:
        ws = self.store.get_workspace(workspace_id)
        if not ws:
            return {"ok": False, "reason": "not_found"}

        open_tabs = self.cdp.list_tabs() if self.cdp.available() else []
        open_by_url = {t["url"]: t for t in open_tabs}
        reused = []
        opened = []
        missing = []

        for item in ws.get("items", []):
            url = item["url"]
            if url in open_by_url:
                reused.append({"url": url, "tab_id": open_by_url[url]["id"]})
                continue
            if self.cdp.available():
                result = self.cdp.open_url(url)
                if result.get("ok"):
                    opened.append({"url": url, "tab": result.get("tab")})
                else:
                    missing.append({"url": url, "error": result.get("error")})
            else:
                # Without CDP, open via system browser one-by-one is aggressive;
                # return URLs for the UI / webbrowser helper.
                missing.append({"url": url, "error": "cdp_unavailable"})

        return {
            "ok": True,
            "workspace": ws,
            "reused": reused,
            "opened": opened,
            "pending_urls": [m["url"] for m in missing],
            "summary": ws.get("summary"),
            "next_step": ws.get("next_step"),
            "note": "Restores links and context. Unsaved forms and web-app state are not recovered.",
        }
