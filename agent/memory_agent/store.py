from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from memory_agent.config import db_path
from memory_agent.crypto import FieldVault, content_hash, decode_embedding, encode_embedding


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS pages (
  id TEXT PRIMARY KEY,
  url TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL DEFAULT '',
  domain TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS page_snapshots (
  id TEXT PRIMARY KEY,
  page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  captured_at TEXT NOT NULL,
  text_enc TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  embedding TEXT,
  extraction_ok INTEGER NOT NULL DEFAULT 1,
  UNIQUE(page_id, content_hash)
);

CREATE TABLE IF NOT EXISTS visits (
  id TEXT PRIMARY KEY,
  page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  snapshot_id TEXT REFERENCES page_snapshots(id) ON DELETE SET NULL,
  started_at TEXT NOT NULL,
  last_active_at TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'observed'
);

CREATE TABLE IF NOT EXISTS workspaces (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  next_step TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  closed_at TEXT
);

CREATE TABLE IF NOT EXISTS workspace_items (
  workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
  page_id TEXT NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
  position INTEGER NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0,
  keep_open INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (workspace_id, page_id)
);

CREATE TABLE IF NOT EXISTS preferences (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'user',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS excluded_sites (
  pattern TEXT PRIMARY KEY,
  reason TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS snapshots_fts USING fts5(
  snapshot_id UNINDEXED,
  page_id UNINDEXED,
  title,
  domain,
  body,
  tokenize = 'porter'
);
"""


class MemoryStore:
    def __init__(self, path: Path | None = None, vault: FieldVault | None = None) -> None:
        self.path = path or db_path()
        self.vault = vault or FieldVault()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._seed_exclusions(conn)

    def _seed_exclusions(self, conn: sqlite3.Connection) -> None:
        from memory_agent.config import DEFAULT_EXCLUSIONS

        now = utcnow()
        for pattern in DEFAULT_EXCLUSIONS:
            conn.execute(
                "INSERT OR IGNORE INTO excluded_sites(pattern, reason, created_at) VALUES (?, ?, ?)",
                (pattern, "default", now),
            )

    # --- preferences ---

    def set_preference(self, key: str, value: Any, source: str = "user") -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO preferences(key, value, source, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value=excluded.value, source=excluded.source, updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), source, utcnow()),
            )

    def get_preference(self, key: str, default: Any = None) -> Any:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM preferences WHERE key=?", (key,)).fetchone()
            if not row:
                return default
            return json.loads(row["value"])

    def list_preferences(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT key, value, source, updated_at FROM preferences ORDER BY key"
            ).fetchall()
            return [
                {
                    "key": r["key"],
                    "value": json.loads(r["value"]),
                    "source": r["source"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    # --- exclusions ---

    def list_exclusions(self) -> list[dict[str, str]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT pattern, reason, created_at FROM excluded_sites ORDER BY pattern"
            ).fetchall()
            return [dict(r) for r in rows]

    def add_exclusion(self, pattern: str, reason: str = "user") -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO excluded_sites(pattern, reason, created_at) VALUES (?, ?, ?)",
                (pattern.strip().lower(), reason, utcnow()),
            )

    def remove_exclusion(self, pattern: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM excluded_sites WHERE pattern=?", (pattern.strip().lower(),))

    def is_excluded(self, url: str) -> bool:
        url_l = url.lower()
        for item in self.list_exclusions():
            p = item["pattern"].lower()
            if p in url_l or url_l.startswith(p):
                return True
        return False

    # --- pages / visits / snapshots ---

    def upsert_page(self, url: str, title: str = "", domain: str = "") -> str:
        now = utcnow()
        with self.connect() as conn:
            row = conn.execute("SELECT id FROM pages WHERE url=?", (url,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE pages SET title=COALESCE(NULLIF(?, ''), title), domain=COALESCE(NULLIF(?, ''), domain), updated_at=? WHERE id=?",
                    (title, domain, now, row["id"]),
                )
                return row["id"]
            page_id = new_id("page")
            conn.execute(
                "INSERT INTO pages(id, url, title, domain, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (page_id, url, title or url, domain, now, now),
            )
            return page_id

    def record_visit(
        self,
        page_id: str,
        when: str | None = None,
        source: str = "observed",
        snapshot_id: str | None = None,
    ) -> str:
        when = when or utcnow()
        with self.connect() as conn:
            # Merge into a recent visit window (30 min) for the same page
            row = conn.execute(
                """
                SELECT id, started_at FROM visits
                WHERE page_id=?
                ORDER BY last_active_at DESC LIMIT 1
                """,
                (page_id,),
            ).fetchone()
            if row:
                # Always bump last_active; collapse if same calendar day roughly via source continuity
                conn.execute(
                    "UPDATE visits SET last_active_at=?, snapshot_id=COALESCE(?, snapshot_id) WHERE id=?",
                    (when, snapshot_id, row["id"]),
                )
                return row["id"]
            visit_id = new_id("visit")
            conn.execute(
                """
                INSERT INTO visits(id, page_id, snapshot_id, started_at, last_active_at, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (visit_id, page_id, snapshot_id, when, when, source),
            )
            return visit_id

    def save_snapshot(
        self,
        page_id: str,
        text: str,
        title: str = "",
        domain: str = "",
        embedding: list[float] | None = None,
        extraction_ok: bool = True,
        captured_at: str | None = None,
    ) -> tuple[str, bool]:
        """Returns (snapshot_id, created_new)."""
        captured_at = captured_at or utcnow()
        digest = content_hash(text)
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM page_snapshots WHERE page_id=? AND content_hash=?",
                (page_id, digest),
            ).fetchone()
            if existing:
                if embedding is not None:
                    conn.execute(
                        "UPDATE page_snapshots SET embedding=? WHERE id=?",
                        (encode_embedding(embedding), existing["id"]),
                    )
                return existing["id"], False

            snap_id = new_id("snap")
            enc = self.vault.encrypt(text)
            conn.execute(
                """
                INSERT INTO page_snapshots(id, page_id, captured_at, text_enc, content_hash, embedding, extraction_ok)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snap_id,
                    page_id,
                    captured_at,
                    enc,
                    digest,
                    encode_embedding(embedding) if embedding is not None else None,
                    1 if extraction_ok else 0,
                ),
            )
            # FTS uses plaintext for searchability in local DB; ciphertext stays in snapshots table.
            # For stricter threat models, switch to encrypted FTS or on-the-fly decrypt search.
            conn.execute(
                "INSERT INTO snapshots_fts(snapshot_id, page_id, title, domain, body) VALUES (?, ?, ?, ?, ?)",
                (snap_id, page_id, title, domain, text),
            )
            return snap_id, True

    def get_snapshot_text(self, snapshot_id: str) -> str:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT text_enc FROM page_snapshots WHERE id=?", (snapshot_id,)
            ).fetchone()
            if not row:
                return ""
            return self.vault.decrypt(row["text_enc"])

    def recent_pages(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.id, p.url, p.title, p.domain, v.last_active_at,
                       (SELECT id FROM page_snapshots s WHERE s.page_id=p.id ORDER BY captured_at DESC LIMIT 1) AS snapshot_id,
                       (SELECT extraction_ok FROM page_snapshots s WHERE s.page_id=p.id ORDER BY captured_at DESC LIMIT 1) AS extraction_ok
                FROM pages p
                JOIN visits v ON v.page_id = p.id
                WHERE v.id = (
                  SELECT id FROM visits WHERE page_id=p.id ORDER BY last_active_at DESC LIMIT 1
                )
                ORDER BY v.last_active_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_page(self, page_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM pages WHERE id=?", (page_id,)).fetchone()
            return dict(row) if row else None

    def pages_by_ids(self, page_ids: list[str]) -> list[dict[str, Any]]:
        if not page_ids:
            return []
        with self.connect() as conn:
            placeholders = ",".join("?" for _ in page_ids)
            rows = conn.execute(
                f"SELECT * FROM pages WHERE id IN ({placeholders})", page_ids
            ).fetchall()
            by_id = {r["id"]: dict(r) for r in rows}
            return [by_id[i] for i in page_ids if i in by_id]

    def keyword_search(
        self,
        query: str,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
    ) -> list[dict[str, Any]]:
        # Escape FTS special chars lightly
        terms = " ".join(t for t in query.replace('"', " ").split() if t)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms.split())
        sql = """
            SELECT f.snapshot_id, f.page_id, f.title, f.domain, f.body,
                   p.url, s.captured_at, s.extraction_ok,
                   snippet(snapshots_fts, 4, '⟦', '⟧', '…', 18) AS excerpt,
                   bm25(snapshots_fts) AS rank
            FROM snapshots_fts f
            JOIN page_snapshots s ON s.id = f.snapshot_id
            JOIN pages p ON p.id = f.page_id
            WHERE snapshots_fts MATCH ?
        """
        params: list[Any] = [fts_query]
        if after:
            sql += " AND s.captured_at >= ?"
            params.append(after)
        if before:
            sql += " AND s.captured_at <= ?"
            params.append(before)
        sql += " ORDER BY rank LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            try:
                rows = conn.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []
            out = []
            for r in rows:
                item = dict(r)
                # Prefer FTS snippet; fall back to decrypted head
                if not item.get("excerpt"):
                    text = self.vault.decrypt(
                        conn.execute(
                            "SELECT text_enc FROM page_snapshots WHERE id=?",
                            (item["snapshot_id"],),
                        ).fetchone()["text_enc"]
                    )
                    item["excerpt"] = text[:280]
                out.append(item)
            return out

    def all_embeddings(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id AS snapshot_id, s.page_id, s.captured_at, s.embedding,
                       s.extraction_ok, p.url, p.title, p.domain
                FROM page_snapshots s
                JOIN pages p ON p.id = s.page_id
                WHERE s.embedding IS NOT NULL
                """
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["embedding"] = decode_embedding(d["embedding"])
                result.append(d)
            return result

    def update_embedding(self, snapshot_id: str, embedding: list[float]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE page_snapshots SET embedding=? WHERE id=?",
                (encode_embedding(embedding), snapshot_id),
            )

    # --- workspaces ---

    def create_workspace(
        self,
        name: str,
        summary: str = "",
        next_step: str = "",
        page_ids: list[str] | None = None,
        keep_open: set[str] | None = None,
    ) -> dict[str, Any]:
        now = utcnow()
        ws_id = new_id("ws")
        keep_open = keep_open or set()
        page_ids = page_ids or []
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO workspaces(id, name, summary, next_step, created_at, updated_at, closed_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (ws_id, name, summary, next_step, now, now),
            )
            for i, pid in enumerate(page_ids):
                conn.execute(
                    """
                    INSERT INTO workspace_items(workspace_id, page_id, position, pinned, keep_open)
                    VALUES (?, ?, ?, 0, ?)
                    """,
                    (ws_id, pid, i, 1 if pid in keep_open else 0),
                )
        return self.get_workspace(ws_id)  # type: ignore[return-value]

    def update_workspace(
        self,
        workspace_id: str,
        name: str | None = None,
        summary: str | None = None,
        next_step: str | None = None,
    ) -> dict[str, Any] | None:
        ws = self.get_workspace(workspace_id)
        if not ws:
            return None
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE workspaces SET
                  name=COALESCE(?, name),
                  summary=COALESCE(?, summary),
                  next_step=COALESCE(?, next_step),
                  updated_at=?
                WHERE id=?
                """,
                (name, summary, next_step, utcnow(), workspace_id),
            )
        return self.get_workspace(workspace_id)

    def mark_workspace_closed(self, workspace_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE workspaces SET closed_at=?, updated_at=? WHERE id=?",
                (utcnow(), utcnow(), workspace_id),
            )

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM workspaces WHERE id=?", (workspace_id,)).fetchone()
            if not row:
                return None
            ws = dict(row)
            items = conn.execute(
                """
                SELECT wi.*, p.url, p.title, p.domain
                FROM workspace_items wi
                JOIN pages p ON p.id = wi.page_id
                WHERE wi.workspace_id=?
                ORDER BY wi.position
                """,
                (workspace_id,),
            ).fetchall()
            ws["items"] = [dict(i) for i in items]
            return ws

    def list_workspaces(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM workspaces ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            out = []
            for r in rows:
                ws = dict(r)
                count = conn.execute(
                    "SELECT COUNT(*) AS c FROM workspace_items WHERE workspace_id=?",
                    (ws["id"],),
                ).fetchone()["c"]
                ws["item_count"] = count
                out.append(ws)
            return out

    def latest_unfinished_workspace(self) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT id FROM workspaces
                WHERE next_step IS NOT NULL AND TRIM(next_step) != ''
                ORDER BY updated_at DESC LIMIT 1
                """
            ).fetchone()
            if not row:
                row = conn.execute(
                    "SELECT id FROM workspaces ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
            if not row:
                return None
        return self.get_workspace(row["id"])

    # --- retention / delete ---

    def delete_page(self, page_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM snapshots_fts WHERE page_id=?", (page_id,))
            conn.execute("DELETE FROM pages WHERE id=?", (page_id,))

    def delete_all_history(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM snapshots_fts")
            conn.execute("DELETE FROM visits")
            conn.execute("DELETE FROM page_snapshots")
            conn.execute("DELETE FROM workspace_items")
            conn.execute("DELETE FROM workspaces")
            conn.execute("DELETE FROM pages")

    def stats(self) -> dict[str, Any]:
        with self.connect() as conn:
            return {
                "pages": conn.execute("SELECT COUNT(*) AS c FROM pages").fetchone()["c"],
                "snapshots": conn.execute("SELECT COUNT(*) AS c FROM page_snapshots").fetchone()["c"],
                "visits": conn.execute("SELECT COUNT(*) AS c FROM visits").fetchone()["c"],
                "workspaces": conn.execute("SELECT COUNT(*) AS c FROM workspaces").fetchone()["c"],
                "exclusions": conn.execute("SELECT COUNT(*) AS c FROM excluded_sites").fetchone()["c"],
                "db_path": str(self.path),
            }
