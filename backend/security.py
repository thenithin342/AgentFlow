"""Security helpers to prevent pickle RCE."""

import hashlib
import hmac
import os
from pathlib import Path

# Get or create a persistent secret for HMAC.
#
# IMPORTANT: the location of the secret file MUST be stable across runs. If it
# moves (e.g. because CHECKPOINT_DB_PATH is overridden in tests), then existing
# .pkl.hmac files verify against a DIFFERENT secret, every check fails, and LTM
# silently returns empty. Pin the secret to a fixed path under the workspace so
# it survives CHECKPOINT_DB_PATH changes. Allow AGENTFLOW_SECRET_DIR to override
# for deployments / tests that need isolation.
_SECRET_DIR = Path(os.environ.get("AGENTFLOW_SECRET_DIR") or Path.cwd())
_SECRET_FILE = _SECRET_DIR / ".faiss_secret"

def _get_secret() -> bytes:
    if _SECRET_FILE.exists():
        return _SECRET_FILE.read_bytes()
    
    secret = os.urandom(32)
    import tempfile
    
    _SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=_SECRET_FILE.parent, text=False)
    try:
        os.write(fd, secret)
        if os.name != 'nt':
            os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
        
    os.replace(temp_path, _SECRET_FILE)
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
