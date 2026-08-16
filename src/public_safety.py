"""Helpers for keeping generated metadata safe to publish.

Provider responses often contain short-lived signed media URLs.  They are useful
for downloading a result locally, but should never be written to a repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def redact_url(value: str) -> str:
    """Keep a URL's host/path while removing query credentials."""
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", "")) + "<redacted-query>"


def redact_urls(value: Any) -> Any:
    """Recursively redact URLs and common credential-shaped response fields."""
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return redact_url(value)
        return value
    if isinstance(value, list):
        return [redact_urls(item) for item in value]
    if isinstance(value, tuple):
        return [redact_urls(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in {"api_key", "apikey", "authorization", "access_token", "secret"}:
                out[key_text] = "<redacted>"
            else:
                out[key_text] = redact_urls(item)
        return out
    return value


def public_path(path: Path) -> str:
    """Store a portable basename instead of a machine-specific absolute path."""
    return path.name
