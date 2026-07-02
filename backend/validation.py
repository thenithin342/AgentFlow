"""Shared validation helpers for thread_id and related untrusted inputs."""

import re

THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


def validate_thread_id(thread_id: str) -> str:
    """Reject thread_ids that are unsafe as filesystem path segments."""
    if not isinstance(thread_id, str) or not THREAD_ID_RE.fullmatch(thread_id):
        got = thread_id if isinstance(thread_id, str) else type(thread_id).__name__
        raise ValueError(
            "invalid thread_id: must match [A-Za-z0-9_-]{1,128} "
            f"(got {got!r})"
        )
    return thread_id
