from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse


class ChromeCDP:
    """Talk to Chrome via remote debugging (no extension).

    Start Chrome with --remote-debugging-port=9222 (the agent installer can
    configure a helper profile). If CDP is unavailable, the agent still works
    from history sync + fetched page text.
    """

    def __init__(self, endpoint: str = "http://127.0.0.1:9222") -> None:
        self.endpoint = endpoint.rstrip("/")

    def available(self) -> bool:
        try:
            self._get("/json/version")
            return True
        except Exception:
            return False

    def list_tabs(self) -> list[dict[str, Any]]:
        try:
            pages = self._get("/json/list")
        except Exception:
            return []
        tabs = []
        for p in pages:
            if p.get("type") != "page":
                continue
            url = p.get("url") or ""
            if not url.startswith("http"):
                continue
            tabs.append(
                {
                    "id": p.get("id"),
                    "title": p.get("title") or url,
                    "url": url,
                    "domain": urlparse(url).netloc.lower(),
                    "webSocketDebuggerUrl": p.get("webSocketDebuggerUrl"),
                }
            )
        return tabs

    def open_url(self, url: str) -> dict[str, Any]:
        # /json/new?{url} — Chrome accepts this for opening a tab
        try:
            data = self._put(f"/json/new?{url}")
            return {"ok": True, "tab": data}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def close_tab(self, target_id: str) -> dict[str, Any]:
        try:
            data = self._get(f"/json/close/{target_id}")
            return {"ok": True, "result": data}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(f"{self.endpoint}{path}", timeout=2.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _put(self, path: str) -> Any:
        req = urllib.request.Request(f"{self.endpoint}{path}", method="PUT")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
