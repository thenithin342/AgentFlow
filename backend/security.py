"""Security helpers to prevent pickle RCE."""

import hmac
import hashlib
import os
from pathlib import Path

# Get or create a persistent secret for HMAC
_SECRET_FILE = Path(os.environ.get("CHECKPOINT_DB_PATH", "agentflow.db")).parent / ".faiss_secret"

def _get_secret() -> bytes:
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_bytes()
    
    secret = os.urandom(32)
    try:
        _SECRET_FILE.write_bytes(secret)
    except Exception:
        pass
    return secret

def sign_file(filepath: Path) -> None:
    """Compute HMAC of a file and write it to filepath.hmac"""
    if not filepath.exists():
        return
    secret = _get_secret()
    h = hmac.new(secret, digestmod=hashlib.sha256)
    h.update(filepath.read_bytes())
    hmac_file = filepath.with_name(filepath.name + ".hmac")
    hmac_file.write_bytes(h.digest())

def verify_file(filepath: Path) -> bool:
    """Verify HMAC of a file. Returns True if valid."""
    if not filepath.exists():
        return False
    hmac_file = filepath.with_name(filepath.name + ".hmac")
    if not hmac_file.exists():
        return False
        
    secret = _get_secret()
    h = hmac.new(secret, digestmod=hashlib.sha256)
    h.update(filepath.read_bytes())
    
    expected = hmac_file.read_bytes()
    return hmac.compare_digest(h.digest(), expected)
