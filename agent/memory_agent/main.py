from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
import webbrowser

import uvicorn

from memory_agent import __version__
from memory_agent.api import create_app
from memory_agent.autostart import install_autostart, uninstall_autostart, autostart_status
from memory_agent.capture import Capturer
from memory_agent.config import APP_NAME, base_url, data_dir, host, port, log_path
from memory_agent.demo_seed import seed_demo_memory
from memory_agent.store import MemoryStore


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path()),
        ],
    )


def cmd_serve(args: argparse.Namespace) -> int:
    _setup_logging()
    log = logging.getLogger("memory_agent")
    store = MemoryStore()
    app = create_app(store)

    if args.sync:
        capturer = Capturer(store)
        result = capturer.sync_chrome_history(limit=args.sync_limit)
        log.info("history sync: %s", result)

    if args.seed_demo:
        seed_demo_memory(store)
        log.info("demo memory seeded")

    def _open_ui() -> None:
        time.sleep(0.8)
        url = base_url()
        log.info("Opening %s", url)
        try:
            webbrowser.open(url)
        except Exception as exc:
            log.warning("Could not open browser: %s", exc)

    if args.open:
        threading.Thread(target=_open_ui, daemon=True).start()

    log.info("%s v%s listening on %s (data: %s)", APP_NAME, __version__, base_url(), data_dir())
    uvicorn.run(app, host=host(), port=port(), log_level="info")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    result = install_autostart()
    print(result)
    return 0 if result.get("ok") else 1


def cmd_uninstall(_: argparse.Namespace) -> int:
    print(uninstall_autostart())
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    store = MemoryStore()
    print(
        {
            "version": __version__,
            "url": base_url(),
            "data_dir": str(data_dir()),
            "stats": store.stats(),
            "autostart": autostart_status(),
        }
    )
    return 0


def cmd_seed(_: argparse.Namespace) -> int:
    store = MemoryStore()
    summary = seed_demo_memory(store)
    print(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="browser-memory",
        description=f"{APP_NAME}: close tabs without losing your place.",
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the local agent (opens UI)")
    serve.add_argument("--open", action="store_true", help="Open the UI in a browser")
    serve.add_argument("--sync", action="store_true", help="Sync Chrome history on start")
    serve.add_argument("--sync-limit", type=int, default=200)
    serve.add_argument("--seed-demo", action="store_true", help="Load supplier-demo memory")
    serve.set_defaults(func=cmd_serve)

    install = sub.add_parser("install", help="Start automatically when you log in")
    install.set_defaults(func=cmd_install)

    uninstall = sub.add_parser("uninstall", help="Remove login auto-start")
    uninstall.set_defaults(func=cmd_uninstall)

    status = sub.add_parser("status", help="Show agent status")
    status.set_defaults(func=cmd_status)

    seed = sub.add_parser("seed-demo", help="Seed the 60-second supplier demo")
    seed.set_defaults(func=cmd_seed)

    return p


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(cli())
