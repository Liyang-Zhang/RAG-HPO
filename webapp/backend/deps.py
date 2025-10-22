"""
Configuration helpers for the RAG-HPO web backend.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


def _str_to_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration pulled from environment variables."""

    meta_path: str = os.getenv("HPO_META_PATH", "data/hpo_meta_bge_small_zh.json")
    vec_path: str = os.getenv("HPO_VEC_PATH", "data/hpo_embedded_bge_small_zh.npz")
    use_sbert: bool = _str_to_bool(os.getenv("HPO_USE_SBERT"), False)
    sbert_model: str = os.getenv(
        "HPO_SBERT_MODEL",
        "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb",
    )
    bge_model: str = os.getenv("HPO_BGE_MODEL", "BAAI/bge-small-zh-v1.5")
    system_prompts_file: str = os.getenv("SYSTEM_PROMPTS_FILE", "system_prompts.json")

    llm_api_key: Optional[str] = os.getenv("LLM_API_KEY")
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    )
    llm_model_name: str = os.getenv("LLM_MODEL_NAME", "qwen3-max")
    max_tokens_per_day: int = int(os.getenv("LLM_MAX_TOKENS_PER_DAY", "500000"))
    max_queries_per_minute: int = int(os.getenv("LLM_MAX_QPM", "30"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Optional debug flag to include raw output in API response
    include_raw: bool = _str_to_bool(os.getenv("INCLUDE_RAW_RESPONSE"), False)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings instance.

    Using an LRU cache ensures that settings are only computed once per process
    and can be accessed cheaply throughout request handling.
    """

    return Settings()
