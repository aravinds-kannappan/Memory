from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup

from memory_agent.config import SENSITIVE_PARAMS
from memory_agent.store import MemoryStore

try:
    from readability import Document
except ImportError:  # pragma: no cover
    Document = None  # type: ignore


def chrome_history_paths() -> list[Path]:
    home = Path.home()
    candidates = [
        home / "Library/Application Support/Google/Chrome/Default/History",
        home / "Library/Application Support/Chromium/Default/History",
        home / ".config/google-chrome/Default/History",
        home / ".config/chromium/Default/History",
        home / "snap/chromium/common/chromium/Default/History",
    ]
    return [p for p in candidates if p.exists()]


def sanitize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if parsed.scheme not in {"http", "https"}:
        return url
    query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in SENSITIVE_PARAMS
    ]
    return urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, parsed.params, urlencode(query), "")
    )


def domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def chrome_time_to_iso(chrome_ts: int) -> str:
    # Chrome stores microseconds since 1601-01-01 UTC
    unix = chrome_ts / 1_000_000 - 11_644_473_600
    return datetime.fromtimestamp(max(unix, 0), tz=timezone.utc).isoformat()


FORM_TAGS = {"input", "textarea", "select", "button", "option"}


def extract_readable_text(html: str, url: str = "") -> tuple[str, str, bool]:
    """Return (title, text, ok). Strips forms and scripts."""
    title = ""
    text = ""
    ok = False
    try:
        if Document is not None:
            doc = Document(html)
            title = (doc.short_title() or "").strip()
            content_html = doc.summary(html_partial=True)
        else:
            content_html = html
        soup = BeautifulSoup(content_html, "lxml")
        for tag in soup(FORM_TAGS):
            tag.decompose()
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()
        if not title:
            full = BeautifulSoup(html, "lxml")
            if full.title and full.title.string:
                title = full.title.string.strip()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        ok = len(text) >= 40
    except Exception:
        ok = False
    if not title:
        title = url
    return title, text, ok


class Capturer:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def sync_chrome_history(self, limit: int = 200) -> dict[str, Any]:
        paths = chrome_history_paths()
        if not paths:
            return {"imported": 0, "skipped": 0, "reason": "no_chrome_history"}
        imported = 0
        skipped = 0
        for hist in paths:
            imported_n, skipped_n = self._import_history_file(hist, limit=limit)
            imported += imported_n
            skipped += skipped_n
        return {"imported": imported, "skipped": skipped, "sources": [str(p) for p in paths]}

    def _import_history_file(self, path: Path, limit: int) -> tuple[int, int]:
        # Chrome locks History; copy first
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "History"
            try:
                shutil.copy2(path, copy)
            except OSError:
                return 0, 0
            conn = sqlite3.connect(copy)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """
                    SELECT u.url, u.title, v.visit_time
                    FROM visits v
                    JOIN urls u ON u.id = v.url
                    WHERE u.url LIKE 'http%'
                    ORDER BY v.visit_time DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            finally:
                conn.close()

        imported = 0
        skipped = 0
        for row in rows:
            url = sanitize_url(row["url"])
            if self.store.is_excluded(url):
                skipped += 1
                continue
            title = row["title"] or url
            domain = domain_of(url)
            page_id = self.store.upsert_page(url, title=title, domain=domain)
            when = chrome_time_to_iso(int(row["visit_time"]))
            self.store.record_visit(page_id, when=when, source="chrome_history")
            imported += 1
        return imported, skipped

    def ingest_page(
        self,
        url: str,
        title: str = "",
        html: str | None = None,
        fetch: bool = True,
        when: str | None = None,
        source: str = "observed",
    ) -> dict[str, Any]:
        url = sanitize_url(url)
        if self.store.is_excluded(url):
            return {"ok": False, "reason": "excluded", "url": url}

        domain = domain_of(url)
        text = ""
        extraction_ok = False
        if html:
            title2, text, extraction_ok = extract_readable_text(html, url)
            title = title or title2
        elif fetch:
            fetched = self.fetch_url(url)
            if fetched:
                title2, text, extraction_ok = extract_readable_text(fetched["html"], url)
                title = title or fetched.get("title") or title2

        page_id = self.store.upsert_page(url, title=title or url, domain=domain)
        snapshot_id = None
        created_new = False
        if text:
            from memory_agent.search.embeddings import embed_text

            embedding = embed_text(text)
            snapshot_id, created_new = self.store.save_snapshot(
                page_id,
                text=text,
                title=title or url,
                domain=domain,
                embedding=embedding,
                extraction_ok=extraction_ok,
                captured_at=when,
            )
        elif not extraction_ok:
            # Link-only memory
            snapshot_id, created_new = self.store.save_snapshot(
                page_id,
                text=f"[link only] {title or url}",
                title=title or url,
                domain=domain,
                embedding=None,
                extraction_ok=False,
                captured_at=when,
            )

        visit_id = self.store.record_visit(
            page_id, when=when, source=source, snapshot_id=snapshot_id
        )
        return {
            "ok": True,
            "page_id": page_id,
            "snapshot_id": snapshot_id,
            "visit_id": visit_id,
            "extraction_ok": extraction_ok,
            "created_new_snapshot": created_new,
            "url": url,
            "title": title or url,
        }

    def fetch_url(self, url: str) -> dict[str, Any] | None:
        try:
            with httpx.Client(follow_redirects=True, timeout=12.0) as client:
                resp = client.get(
                    url,
                    headers={
                        "User-Agent": "BrowserMemoryAgent/0.1 (+local; readability)",
                        "Accept": "text/html,application/xhtml+xml",
                    },
                )
                if resp.status_code >= 400:
                    return None
                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype and "text/" not in ctype:
                    return None
                return {"html": resp.text, "title": "", "final_url": str(resp.url)}
        except Exception:
            return None

    def enrich_recent_without_snapshots(self, limit: int = 30) -> dict[str, Any]:
        """Fetch readable content for recently visited pages that lack good snapshots."""
        pages = self.store.recent_pages(limit=limit)
        enriched = 0
        failed = 0
        for p in pages:
            if p.get("extraction_ok"):
                continue
            result = self.ingest_page(p["url"], title=p.get("title") or "", fetch=True, source="enrich")
            if result.get("ok") and result.get("extraction_ok"):
                enriched += 1
            else:
                failed += 1
        return {"enriched": enriched, "failed": failed}
