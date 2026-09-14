from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet

from memory_agent.config import key_path


class FieldVault:
    """Application-level encryption for sensitive snapshot text.

    SQLCipher is ideal long-term; Fernet keeps packaging simple for the MVP
    while still encrypting saved page content at rest.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or key_path()
        self._fernet = Fernet(self._load_or_create_key())

    def _load_or_create_key(self) -> bytes:
        if self.path.exists():
            return self.path.read_bytes().strip()
        key = Fernet.generate_key()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(key)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        return key

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")

    def decrypt(self, token: str) -> str:
        return self._fernet.decrypt(token.encode("ascii")).decode("utf-8")

    def encrypt_json(self, value: Any) -> str:
        return self.encrypt(json.dumps(value, ensure_ascii=False))

    def decrypt_json(self, token: str) -> Any:
        return json.loads(self.decrypt(token))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def encode_embedding(vec: list[float]) -> str:
    return base64.b64encode(json.dumps(vec).encode("utf-8")).decode("ascii")


def decode_embedding(payload: str) -> list[float]:
    return json.loads(base64.b64decode(payload.encode("ascii")).decode("utf-8"))
