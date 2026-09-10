"""Private Z.ai credential storage for Coding Plan routing."""

from __future__ import annotations

from .paths import zai_credential_path
from .storage import atomic_write_text

ZAI_UPSTREAM = "https://api.z.ai/api/anthropic"


def validate_zai_key_shape(key: str) -> None:
    if len(key) < 20 or any(character.isspace() for character in key):
        raise ValueError("the Z.ai key has an unexpected format (a token of 20+ characters)")


def read_zai_credential() -> str:
    path = zai_credential_path()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Z.ai credential not found at {path}; run `clr config --zai-key`"
        ) from exc
    validate_zai_key_shape(key)
    return key


def write_zai_credential(key: str) -> None:
    validate_zai_key_shape(key)
    atomic_write_text(zai_credential_path(), f"{key}\n", 0o600)
