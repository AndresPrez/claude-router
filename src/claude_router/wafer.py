"""Private Wafer Serverless credential storage for wafer.ai routing."""

from __future__ import annotations

from .paths import wafer_credential_path
from .storage import atomic_write_text

WAFER_UPSTREAM = "https://pass.wafer.ai"


def validate_wafer_key_shape(key: str) -> None:
    if len(key) < 20 or any(character.isspace() for character in key):
        raise ValueError(
            "the Wafer API key has an unexpected format (a key of 20+ characters)"
        )


def read_wafer_credential() -> str:
    path = wafer_credential_path()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Wafer API key not found at {path}; generate one at app.wafer.ai and "
            "run `clr config --wafer-key`"
        ) from exc
    validate_wafer_key_shape(key)
    return key


def write_wafer_credential(key: str) -> None:
    validate_wafer_key_shape(key)
    atomic_write_text(wafer_credential_path(), f"{key}\n", 0o600)
