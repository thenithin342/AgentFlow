"""
Configuration loader for the evals package.

Reads evals/config.yaml and exposes thresholds, judge config,
and retriever config as module-level defaults.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, falling back to a dict on error."""
    try:
        import yaml  # type: ignore

        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return data if isinstance(data, dict) else {}
    except ImportError:
        # PyYAML not available — return empty dict; callers should handle
        return {}
    except Exception:
        return {}


def load_config() -> dict[str, Any]:
    """Load and return the full evals/config.yaml as a dict."""
    return _load_yaml(CONFIG_PATH)


def get_thresholds() -> dict[str, Any]:
    """Return the thresholds dict from config.yaml."""
    config = load_config()
    return config.get("thresholds", {})


def get_judge_config() -> dict[str, Any]:
    """Return the judges config dict from config.yaml."""
    config = load_config()
    return config.get("judges", {})


def get_retriever_config() -> dict[str, Any]:
    """Return the retriever config dict from config.yaml."""
    config = load_config()
    return config.get("retriever", {})


# Module-level cached values (loaded once at import)
_THRESHOLDS: dict[str, Any] | None = None
_JUDGE_CONFIG: dict[str, Any] | None = None
_RETRIEVER_CONFIG: dict[str, Any] | None = None


def _get_thresholds_cached() -> dict[str, Any]:
    global _THRESHOLDS
    if _THRESHOLDS is None:
        _THRESHOLDS = get_thresholds()
    return _THRESHOLDS


def _get_judge_config_cached() -> dict[str, Any]:
    global _JUDGE_CONFIG
    if _JUDGE_CONFIG is None:
        _JUDGE_CONFIG = get_judge_config()
    return _JUDGE_CONFIG


def _get_retriever_config_cached() -> dict[str, Any]:
    global _RETRIEVER_CONFIG
    if _RETRIEVER_CONFIG is None:
        _RETRIEVER_CONFIG = get_retriever_config()
    return _RETRIEVER_CONFIG
