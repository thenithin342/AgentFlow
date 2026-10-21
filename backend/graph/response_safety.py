"""Small deterministic guards for high-risk requests and emitted responses."""

from __future__ import annotations

import re
from typing import Any

_OVERRIDE = re.compile(
    r"<<\s*(?:end|untrusted)[^>]*>>|\b(?:ignore|disregard|override)\s+"
    r"(?:all\s+|your\s+|any\s+)?(?:previous\s+|prior\s+)?"
    r"(?:instructions|rules|system\s+prompt)\b",
    re.IGNORECASE,
)
_TAX_FILING = re.compile(r"\b(?:file|submit)\s+(?:my|our)\s+tax(?:es|\s+return)\b", re.IGNORECASE)
_DIAGNOSIS = re.compile(r"\b(?:give|provide|make)\s+(?:me\s+)?(?:a\s+)?medical\s+diagnosis\b", re.IGNORECASE)
_LEGAL_CONTRACT = re.compile(
    r"\b(?:write|draft|prepare|create)\s+(?:me\s+)?(?:a\s+)?(?:legal\s+)?"
    r"(?:contract|sale\s+agreement)\s+for\b[^\n]*\b(?:house|property|real\s+estate)\b",
    re.IGNORECASE,
)
_LOTTERY_FUTURE = re.compile(
    r"\bwinning\s+lottery\s+numbers?\b.*\b(?:tomorrow|next\s+draw|future)\b",
    re.IGNORECASE,
)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_API_KEY = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{4,}|gsk_[A-Za-z0-9_]{4,})\b")
_BEARER = re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*", re.IGNORECASE)
_ENV_SECRET = re.compile(
    r"\b[A-Z0-9_]*(?:API_KEY|SECRET|PASSWORD|TOKEN|PRIVATE_KEY)\s*=\s*"
    r'''(?:"[^"\n]*"|'[^'\n]*'|[^\s`]+)'''
)


def refusal_for(message: str) -> str | None:
    """Return a narrow refusal for requests the app must not perform."""
    if _OVERRIDE.search(message):
        return "I can't follow instructions that override my operating rules."
    if _TAX_FILING.search(message):
        return "I can't file taxes for you, but I can help explain the process."
    if _DIAGNOSIS.search(message):
        return "I can't provide a medical diagnosis. Please consult a qualified clinician."
    if _LEGAL_CONTRACT.search(message):
        return "I can't draft a legally operative contract for you. A qualified lawyer can help with the agreement."
    if _LOTTERY_FUTURE.search(message):
        return "I can't know future winning lottery numbers."
    return None


def redact_pii(text: str) -> str:
    """Mask common personal identifiers before saving or returning an answer."""
    text = _EMAIL.sub("[email redacted]", text)
    text = _SSN.sub("[SSN redacted]", text)
    return _PHONE.sub("[phone redacted]", text)


def redact_protected(text: str) -> str:
    text = _ENV_SECRET.sub("[config value redacted]", text)
    text = _API_KEY.sub("[key redacted]", text)
    return _BEARER.sub("[bearer token redacted]", text)


def redact_sensitive(text: str) -> str:
    return redact_pii(redact_protected(text))


def redact_content(value: Any) -> Any:
    """Redact text inside structured generated content without changing keys."""
    if isinstance(value, str):
        return redact_sensitive(value)
    if isinstance(value, list):
        return [redact_content(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_content(item) for key, item in value.items()}
    return value
