"""Private Fireworks AI credential storage for serverless routing."""

from __future__ import annotations

from .paths import fireworks_credential_path
from .storage import atomic_write_text

FIREWORKS_UPSTREAM = "https://api.fireworks.ai/inference"


def validate_fireworks_key_shape(key: str) -> None:
    if len(key) < 20 or any(character.isspace() for character in key):
        raise ValueError(
            "the Fireworks API key has an unexpected format (a key of 20+ characters)"
        )


def read_fireworks_credential() -> str:
    path = fireworks_credential_path()
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Fireworks API key not found at {path}; create one at "
            "app.fireworks.ai and run `clr config --fireworks-key`"
        ) from exc
    validate_fireworks_key_shape(key)
    return key


def write_fireworks_credential(key: str) -> None:
    validate_fireworks_key_shape(key)
    atomic_write_text(fireworks_credential_path(), f"{key}\n", 0o600)
