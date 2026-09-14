from __future__ import annotations

import os
import platform
from pathlib import Path


APP_NAME = "Browser Memory"
APP_ID = "com.browsermemory.agent"


def data_dir() -> Path:
    override = os.environ.get("MEMORY_DATA_DIR")
    if override:
        path = Path(override).expanduser()
    elif platform.system() == "Darwin":
        path = Path.home() / "Library" / "Application Support" / "BrowserMemory"
    else:
        path = Path.home() / ".local" / "share" / "browser-memory"
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path() -> Path:
    return data_dir() / "memory.db"


def key_path() -> Path:
    return data_dir() / "master.key"


def log_path() -> Path:
    return data_dir() / "agent.log"


def host() -> str:
    return os.environ.get("MEMORY_HOST", "127.0.0.1")


def port() -> int:
    return int(os.environ.get("MEMORY_PORT", "8765"))


def base_url() -> str:
    return f"http://{host()}:{port()}"


DEFAULT_EXCLUSIONS = [
    "accounts.google.com",
    "login.microsoftonline.com",
    "id.apple.com",
    "chase.com",
    "bankofamerica.com",
    "wellsfargo.com",
    "paypal.com",
    "stripe.com",
    "chrome://",
    "chrome-extension://",
    "about:",
    "devtools://",
    "file://",
]

# Sensitive query params stripped before storage
SENSITIVE_PARAMS = {
    "token",
    "access_token",
    "auth",
    "password",
    "passwd",
    "session",
    "sid",
    "api_key",
    "apikey",
    "key",
    "secret",
    "code",
    "oauth",
}
