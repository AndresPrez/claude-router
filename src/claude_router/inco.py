"""Private Inco AI credential storage for api.inco.ai routing."""

from __future__ import annotations

from .paths import inco_credential_path
from .storage import atomic_write_text

INCO_UPSTREAM = "https://api.inco.ai"


def validate_inco_key_shape(key: str) -> None:
    if len(key) < 20 or any(character.isspace() for character in key):
        raise ValueError(
            "the Inco AI API key has an unexpected format (a key of 20+ characters)"
        )


def read_inco_credential() -> str:
    path = inco_credential_path()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Inco AI API key not found at {path}; create one at "
            "platform.inco.ai and run `clr config --inco-key`"
        ) from exc
    validate_inco_key_shape(key)
    return key


def write_inco_credential(key: str) -> None:
    validate_inco_key_shape(key)
    atomic_write_text(inco_credential_path(), f"{key}\n", 0o600)
