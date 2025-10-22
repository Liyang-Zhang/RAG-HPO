"""
Service layer that bridges the FastAPI endpoints with the existing pipeline.
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd

from rag_hpo.pipeline import run_rag_pipeline

from .deps import Settings

_ZH_PATTERN = re.compile(r"[\u4e00-\u9fff]")


@lru_cache(maxsize=4)
def _get_translation_map(meta_path: str) -> Dict[str, str]:
    if not meta_path:
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

    constants = data.get("constants", {})
    mapping: Dict[str, str] = {}
    for hp_id, info in constants.items():
        translations = (info or {}).get("translations") or ""
        primary = (info or {}).get("primary_label") or ""

        chosen = ""
        for part in translations.split(";"):
            candidate = part.strip()
            if candidate and _ZH_PATTERN.search(candidate):
                chosen = candidate
                break

        if not chosen and primary and _ZH_PATTERN.search(str(primary)):
            chosen = str(primary).strip()

        if not chosen and translations:
            first = translations.split(";")[0].strip()
            chosen = first

        mapping[str(hp_id)] = chosen
    return mapping


def _format_results(
    csv_path: Path,
    translation_map: Dict[str, str],
) -> List[Dict[str, str]]:
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    normalized = {col: col.strip().lower().replace(" ", "_") for col in df.columns}
    df = df.rename(columns=normalized)
    records: List[Dict[str, str]] = []
    for row in df.to_dict(orient="records"):
        hpo_id = str(row.get("hpo_id", ""))
        records.append(
            {
                "patient_id": str(row.get("patient_id", "1")),
                "category": str(row.get("category", "")),
                "phrase": str(row.get("phenotype_name", "")),
                "hpo_id": hpo_id,
                "translation": translation_map.get(hpo_id, ""),
            }
        )
    return records


def _format_raw(raw_path: Path) -> List[Dict[str, str]]:
    if not raw_path.exists():
        return []
    df = pd.read_csv(raw_path)
    records: List[Dict[str, str]] = df.to_dict(orient="records")
    # Try to convert stringified dicts in HPO_Terms to Python objects for readability.
    for record in records:
        terms = record.get("HPO_Terms")
        if isinstance(terms, str):
            try:
                record["HPO_Terms"] = json.loads(terms.replace("'", '"'))
            except json.JSONDecodeError:
                # Keep original string if parsing fails.
                continue
    return records


def run_case(text: str, settings: Settings) -> Dict:
    """
    Run the RAG pipeline for a single clinical note and return structured data.
    """

    note = text.strip()
    if not note:
        raise ValueError("临床文本不能为空。")
    if not settings.llm_api_key:
        raise ValueError("缺少 LLM_API_KEY，无法调用模型。")

    input_df = pd.DataFrame({"clinical_note": [note]})

    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "output.csv"
        raw_path = tmpdir_path / "raw.csv"

        run_rag_pipeline(
            input_data=input_df,
            output_csv_path=str(csv_path),
            output_json_raw_path=str(raw_path),
            display_results=False,
            meta_path=settings.meta_path,
            vec_path=settings.vec_path,
            use_sbert=settings.use_sbert,
            sbert_model=settings.sbert_model,
            bge_model=settings.bge_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            llm_model_name=settings.llm_model_name,
            max_tokens_per_day=settings.max_tokens_per_day,
            max_queries_per_minute=settings.max_queries_per_minute,
            temperature=settings.temperature,
            system_prompts_file=settings.system_prompts_file,
        )

        runtime = time.perf_counter() - started
        translation_map = _get_translation_map(settings.meta_path)
        formatted = _format_results(csv_path, translation_map)
        if not formatted:
            return {
                "patient_id": 1,
                "phenotypes": [],
                "runtime_seconds": runtime,
                "raw_entries": [] if settings.include_raw else None,
            }

        # Convert to response structure
        phenotypes = [
            {
                "phrase": entry["phrase"],
                "category": entry["category"],
                "hpo_id": entry["hpo_id"],
                "translation": entry.get("translation", ""),
                "keep": True,
            }
            for entry in formatted
        ]

        try:
            patient_id = int(formatted[0]["patient_id"])
        except (ValueError, TypeError):
            patient_id = 1

        response = {
            "patient_id": patient_id,
            "phenotypes": phenotypes,
            "runtime_seconds": runtime,
        }

        if settings.include_raw:
            response["raw_entries"] = _format_raw(raw_path)
        else:
            response["raw_entries"] = None

        return response
