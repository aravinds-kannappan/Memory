import webbrowser
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from memory_agent import __version__
from memory_agent.autostart import autostart_status, install_autostart, uninstall_autostart
from memory_agent.capture import Capturer
from memory_agent.capture.cdp import ChromeCDP
from memory_agent.config import APP_NAME, base_url
from memory_agent.search import RetrievalEngine
from memory_agent.store import MemoryStore
from memory_agent.workspaces import WorkspaceService

STATIC_DIR = Path(__file__).parent / "ui" / "static"


def create_app(store: Optional[MemoryStore] = None) -> FastAPI:
    store = store or MemoryStore()
    capturer = Capturer(store)
    cdp = ChromeCDP()
    retrieval = RetrievalEngine(store)
    workspaces = WorkspaceService(store, cdp)

    app = FastAPI(title=APP_NAME, version=__version__)
    app.state.store = store
    app.state.capturer = capturer
    app.state.cdp = cdp
    app.state.retrieval = retrieval
    app.state.workspaces = workspaces

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "version": __version__,
            "name": APP_NAME,
            "url": base_url(),
            "cdp": cdp.available(),
            "stats": store.stats(),
            "autostart": autostart_status(),
        }

    @app.get("/api/recent")
    def recent(limit: int = 20) -> dict[str, Any]:
        return {"pages": store.recent_pages(limit=limit)}

    class AskBody(BaseModel):
        query: str = Field(min_length=1)

    @app.post("/api/ask")
    def ask(body: AskBody) -> dict[str, Any]:
        return retrieval.search(body.query)

    class IngestBody(BaseModel):
        url: str
        title: str = ""
        html: Optional[str] = None
        fetch: bool = True

    @app.post("/api/ingest")
    def ingest(body: IngestBody) -> dict[str, Any]:
        return capturer.ingest_page(
            body.url, title=body.title, html=body.html, fetch=body.fetch, source="manual"
        )

    @app.post("/api/sync/history")
    def sync_history(limit: int = 200) -> dict[str, Any]:
        result = capturer.sync_chrome_history(limit=limit)
        enrich = capturer.enrich_recent_without_snapshots(limit=min(limit, 40))
        return {**result, "enrich": enrich}

    @app.get("/api/tabs")
    def tabs() -> dict[str, Any]:
        available = cdp.available()
        return {"cdp": available, "tabs": cdp.list_tabs() if available else []}

    @app.get("/api/workspaces")
    def list_workspaces() -> dict[str, Any]:
        return {
            "workspaces": store.list_workspaces(),
            "unfinished": store.latest_unfinished_workspace(),
        }

    @app.get("/api/workspaces/{workspace_id}")
    def get_workspace(workspace_id: str) -> dict[str, Any]:
        ws = store.get_workspace(workspace_id)
        if not ws:
            raise HTTPException(404, "Workspace not found")
        return ws

    class ProposeBody(BaseModel):
        next_step: str = ""

    @app.post("/api/workspaces/propose")
    def propose(body: ProposeBody) -> dict[str, Any]:
        return workspaces.propose_from_open_tabs(next_step=body.next_step)

    class SaveWorkspaceBody(BaseModel):
        name: str
        summary: str = ""
        next_step: str = ""
        page_ids: list[str]
        keep_open_page_ids: list[str] = []
        close_tab_ids: list[str] = []

    @app.post("/api/workspaces/save")
    def save_workspace(body: SaveWorkspaceBody) -> dict[str, Any]:
        ws = workspaces.save(
            name=body.name,
            summary=body.summary,
            next_step=body.next_step,
            page_ids=body.page_ids,
            keep_open_page_ids=body.keep_open_page_ids,
        )
        close_result = None
        if body.close_tab_ids:
            close_result = workspaces.close_approved_tabs(
                body.close_tab_ids, workspace_id=ws["id"]
            )
        return {"workspace": ws, "close": close_result}

    class CloseBody(BaseModel):
        tab_ids: list[str]
        workspace_id: Optional[str] = None

    @app.post("/api/actions/close-tabs")
    def close_tabs(body: CloseBody) -> dict[str, Any]:
        return workspaces.close_approved_tabs(body.tab_ids, workspace_id=body.workspace_id)

    @app.post("/api/workspaces/{workspace_id}/resume")
    def resume(workspace_id: str) -> dict[str, Any]:
        result = workspaces.resume(workspace_id)
        if not result.get("ok"):
            raise HTTPException(404, result.get("reason", "not found"))
        # If CDP unavailable, open pending URLs via webbrowser (capped)
        for url in result.get("pending_urls", [])[:12]:
            try:
                webbrowser.open_new_tab(url)
            except Exception:
                pass
        return result

    class UpdateWorkspaceBody(BaseModel):
        name: Optional[str] = None
        summary: Optional[str] = None
        next_step: Optional[str] = None

    @app.patch("/api/workspaces/{workspace_id}")
    def patch_workspace(workspace_id: str, body: UpdateWorkspaceBody) -> dict[str, Any]:
        ws = store.update_workspace(
            workspace_id, name=body.name, summary=body.summary, next_step=body.next_step
        )
        if not ws:
            raise HTTPException(404, "Workspace not found")
        return ws

    # Memory controls
    @app.get("/api/memory/exclusions")
    def exclusions() -> dict[str, Any]:
        return {"exclusions": store.list_exclusions()}

    class ExclusionBody(BaseModel):
        pattern: str
        reason: str = "user"

    @app.post("/api/memory/exclusions")
    def add_exclusion(body: ExclusionBody) -> dict[str, Any]:
        store.add_exclusion(body.pattern, reason=body.reason)
        return {"exclusions": store.list_exclusions()}

    @app.delete("/api/memory/exclusions")
    def delete_exclusion(pattern: str) -> dict[str, Any]:
        store.remove_exclusion(pattern)
        return {"exclusions": store.list_exclusions()}

    class PreferenceBody(BaseModel):
        key: str
        value: Any
        source: str = "user"

    @app.get("/api/memory/preferences")
    def preferences() -> dict[str, Any]:
        return {"preferences": store.list_preferences()}

    @app.post("/api/memory/preferences")
    def set_preference(body: PreferenceBody) -> dict[str, Any]:
        store.set_preference(body.key, body.value, source=body.source)
        return {"ok": True}

    @app.post("/api/memory/delete-page/{page_id}")
    def delete_page(page_id: str) -> dict[str, Any]:
        store.delete_page(page_id)
        return {"ok": True}

    @app.post("/api/memory/delete-all")
    def delete_all() -> dict[str, Any]:
        store.delete_all_history()
        return {"ok": True, "stats": store.stats()}

    @app.post("/api/autostart/install")
    def api_install_autostart() -> dict[str, Any]:
        return install_autostart()

    @app.post("/api/autostart/uninstall")
    def api_uninstall_autostart() -> dict[str, Any]:
        return uninstall_autostart()

    @app.get("/api/open")
    def open_url(url: str) -> dict[str, Any]:
        if cdp.available():
            return cdp.open_url(url)
        webbrowser.open_new_tab(url)
        return {"ok": True, "via": "webbrowser"}

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

    return app
