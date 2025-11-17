import json
import os
from pathlib import Path
from typing import Any, Dict


DEFAULT_CONFIG_PATH = Path(os.getenv("HPO_CONFIG", "config/hpo_config.json"))


def load_runtime_config(path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """Load JSON config if it exists, otherwise return empty dict."""
    cfg_path = Path(path or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {}
            return data
    except Exception:
        return {}


def apply_runtime_config(path: str | os.PathLike | None = None) -> None:
    """
    Apply runtime settings by setting environment variables.
    Existing env vars take precedence.
    """

    config = load_runtime_config(path)
    if not config:
        return

    runtime = config.get("runtime", config)

    mapping = {
        "keep_categories": ("KEEP_CATEGORIES", lambda v: ",".join(v) if isinstance(v, list) else str(v)),
        "hpo_fallback_topk": ("HPO_FALLBACK_TOPK", str),
        "hpo_extra_candidates": ("HPO_EXTRA_CANDIDATES", str),
        "hpo_debug": ("HPO_DEBUG", lambda v: "1" if v else "0"),
        "hpo_log_file": ("HPO_LOG_FILE", str),
    }

    for key, (env_var, formatter) in mapping.items():
        if env_var in os.environ:
            continue
        if key in runtime and runtime[key] is not None:
            os.environ[env_var] = formatter(runtime[key])

    llm_cfg = config.get("llm", {})
    llm_mapping = {
        "api_key": "LLM_API_KEY",
        "base_url": "LLM_BASE_URL",
        "model_name": "LLM_MODEL_NAME",
    }
    for key, env_var in llm_mapping.items():
        if env_var in os.environ:
            continue
        value = llm_cfg.get(key)
        if value:
            os.environ[env_var] = str(value)
