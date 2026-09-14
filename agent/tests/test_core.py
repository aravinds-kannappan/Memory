from __future__ import annotations

from memory_agent.demo_seed import seed_demo_memory
from memory_agent.search import RetrievalEngine
from memory_agent.store import MemoryStore
from memory_agent.workspaces import suggest_workspace


def test_flagship_recall(tmp_path):
    store = MemoryStore(path=tmp_path / "memory.db", vault=__import__("memory_agent.crypto", fromlist=["FieldVault"]).FieldVault(tmp_path / "key"))
    seed_demo_memory(store)
    engine = RetrievalEngine(store)
    result = engine.search("What was that supplier with the minimum order of 500?")
    assert result["answer"]["found"] is True
    top = result["results"][0]
    assert "northwind" in top["url"].lower() or "500" in (top.get("excerpt") or "")
    assert result["answer"]["sources"]


def test_time_filter_last_night(tmp_path):
    from memory_agent.search import interpret_query

    parsed = interpret_query("Find the supplier I looked at last night")
    assert parsed["after"] is not None
    assert "supplier" in parsed["terms"].lower()


def test_workspace_suggestion():
    tabs = [
        {"title": "Northwind Parts", "url": "https://a.example/x", "domain": "a.example"},
        {"title": "Harbor Alloy catalog", "url": "https://b.example/y", "domain": "b.example"},
    ]
    s = suggest_workspace(tabs)
    assert s["name"]
    assert "2 related tabs" in s["summary"]


def test_exclusion_blocks_ingest(tmp_path):
    from memory_agent.capture import Capturer
    from memory_agent.crypto import FieldVault

    store = MemoryStore(path=tmp_path / "memory.db", vault=FieldVault(tmp_path / "key"))
    store.add_exclusion("secret-bank.example")
    cap = Capturer(store)
    result = cap.ingest_page("https://secret-bank.example/account", fetch=False)
    assert result["ok"] is False
    assert result["reason"] == "excluded"


def test_memory_kinds_separated(tmp_path):
    from memory_agent.crypto import FieldVault

    store = MemoryStore(path=tmp_path / "memory.db", vault=FieldVault(tmp_path / "key"))
    seed = seed_demo_memory(store)
    ws = store.get_workspace(seed["workspace_id"])
    assert ws["next_step"] == "Tomorrow, compare delivery times."
    # Observed content is in snapshots, not in next_step
    assert "Minimum order" not in ws["next_step"]
