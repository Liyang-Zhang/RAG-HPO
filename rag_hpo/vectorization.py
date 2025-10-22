import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pronto
import requests
from fastembed import TextEmbedding
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .utils import clean_text_for_embedding, logger

# ────────── 0. Automatic OBO Download & Load ──────────
_hpo_ontology = None
_terms_cache = None
parent_map = None
label_map = None


def initialize_hpo_resources(
    obo_url: str = "https://purl.obolibrary.org/obo/hp.obo",
    obo_path: str = "hp.obo",
    refresh_days: int = 14,
) -> pronto.Ontology:
    """
    Download or refresh the HPO OBO file if older than `refresh_days`,
    then load it via pronto.
    """
    global _hpo_ontology
    if _hpo_ontology is None:
        # Ensure obo_path is absolute
        abs_obo_path = os.path.abspath(obo_path)
        obo_file = Path(abs_obo_path)
        if not obo_file.exists() or ((time.time() - obo_file.stat().st_mtime) / 86400) > refresh_days:
            logger.log(f"Downloading HPO ontology from {obo_url} …")
            resp = requests.get(obo_url, timeout=30)
            resp.raise_for_status()
            obo_file.write_text(resp.text, encoding="utf-8")
        with open(obo_file, "rb") as f:
            _hpo_ontology = pronto.Ontology(f)
        logger.log(f"Loaded ontology with {len(list(_hpo_ontology.terms()))} terms")
    return _hpo_ontology


ROOT_ID = "HP:0000001"  # “All”
PHENO_ID = "HP:0000118"  # “Phenotypic abnormality”


def load_chpo_mapping(chpo_path: str | None) -> dict[str, list[str]]:
    """
    加载中文 HPO 术语映射，支持 Excel(.xlsx/.xls) 或 CSV。
    返回 dict[hp_id] = [cn_translation, ...]
    """
    if not chpo_path:
        return {}
    if not os.path.exists(chpo_path):
        logger.log(f"Warning: CHPO file '{chpo_path}' not found. " "Skipping Chinese translations.")
        return {}
    try:
        if chpo_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(chpo_path)
        else:
            df = pd.read_csv(chpo_path)
    except Exception as exc:
        logger.log(f"Warning: Failed to load CHPO file '{chpo_path}': {exc}.")
        return {}

    def _find_col(candidates):
        for cand in candidates:
            for col in df.columns:
                if cand.lower() == col.lower():
                    return col
        return None

    id_col = _find_col(["HPO编号", "HPO ID", "HPO", "HPO_Id", "HPO编号 "])
    cn_col = _find_col(["中文翻译", "中文名称", "Chinese", "中文"])

    if not id_col or not cn_col:
        logger.log(f"Warning: Could not find HPO编号/中文翻译 columns in '{chpo_path}'.")
        return {}

    mapping: dict[str, set] = {}
    for _, row in df.iterrows():
        hp_raw = str(row[id_col]).strip()
        if not hp_raw.startswith("HP:"):
            continue
        cn = str(row[cn_col]).strip()
        if not cn or cn.lower() == "nan":
            continue
        mapping.setdefault(hp_raw, set()).add(cn)

    clean_mapping = {hp: sorted(values) for hp, values in mapping.items() if values}
    logger.log(f"Loaded {len(clean_mapping)} Chinese translations from {chpo_path}")
    return clean_mapping


def ensure_hpo_cached(
    obo_url: str = "https://purl.obolibrary.org/obo/hp.obo",
    obo_path: str = "hp.obo",
    refresh_days: int = 14,
) -> pronto.Ontology:
    """
    Lazily load ontology and derived lookup tables. Keeps compatibility with
    CLI imports that only need module metadata without immediately touching
    joblib multiprocessing.
    """
    global _terms_cache, parent_map, label_map

    ontology = initialize_hpo_resources(
        obo_url=obo_url,
        obo_path=obo_path,
        refresh_days=refresh_days,
    )
    if ontology is None:
        raise RuntimeError("Failed to load HPO ontology.")

    if _terms_cache is None:
        _terms_cache = list(ontology.terms())

    if parent_map is None or label_map is None:
        parent_map = {term.id: [p.id for p in term.superclasses(distance=1)] for term in _terms_cache}
        label_map = {term.id: term.name for term in _terms_cache}

    return ontology


# ────────── 1. Multi-Path Lineage Helper ──────────
_lineage_memo = {}


def _build_lineage_paths(hp_id, parent_map, seen=None):
    """
    Return all paths from ROOT_ID → ... → hp_id.
    Each path is a list of HP_ID strings.
    Avoids cycles by tracking `seen`.
    """
    if seen is None:
        seen = set()
    if hp_id in seen:  # cycle guard
        return []
    seen = seen | {hp_id}

    # cached?
    if hp_id in _lineage_memo:
        return _lineage_memo[hp_id]

    # base case: we hit the root
    if hp_id == ROOT_ID:
        paths = [[ROOT_ID]]
    else:
        parents = parent_map.get(hp_id, [])
        if not parents:
            # orphan: just attach root + self
            paths = [[ROOT_ID, hp_id]]
        else:
            paths = []
            for p in parents:
                for ppath in _build_lineage_paths(p, parent_map, seen):
                    paths.append(ppath + [hp_id])

    _lineage_memo[hp_id] = paths
    return paths


# ────────── 2. Build DataFrame from OBO ──────────
CLEAN_ABNORMALITY = re.compile(r"(?i)^Abnormality of(?: the)?\s*")


def _sort_by_numeric(entries: list[str]) -> list[str]:
    def key_fn(e: str):
        digs = "".join(filter(str.isdigit, e))
        return int(digs) if digs else float("inf")

    return sorted(entries, key=key_fn)


def build_hpo_dataframe(
    limit: int = None,
    obo_url: str = "https://purl.obolibrary.org/obo/hp.obo",
    obo_path: str = "hp.obo",
    refresh_days: int = 14,
    chpo_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    ontology = ensure_hpo_cached(
        obo_url=obo_url,
        obo_path=obo_path,
        refresh_days=refresh_days,
    )
    global _terms_cache, parent_map, label_map
    chpo_map = chpo_map or {}
    records = []
    terms_source = _terms_cache if _terms_cache is not None else list(ontology.terms())
    terms = terms_source[:limit] if limit else terms_source

    for term in tqdm(terms, desc="Building HPO DataFrame", unit="term"):
        hp_id = term.id
        label = term.name
        definition = term.definition or ""
        synonyms = [syn.description for syn in term.synonyms]

        # ── ALT IDs ──
        alt_ids = _sort_by_numeric(list(term.alternate_ids))

        # ── XREFS ──
        snomedct, umls = [], []
        for xr in term.xrefs:
            txt = str(xr)
            m = re.search(r"'(.+?:.+?)'", txt)
            ent = m.group(1) if m else txt
            pre, _, _ = ent.partition(":")
            if pre.upper() == "UMLS":
                umls.append(ent)
            elif pre.upper().startswith("SNOMED"):
                snomedct.append(ent)
        snomedct = _sort_by_numeric(snomedct)
        umls = _sort_by_numeric(umls)

        # ── LINEAGE ──
        paths = _build_lineage_paths(hp_id, parent_map) or [[ROOT_ID, hp_id]]
        for path_ids in paths:
            lineage_str = " -> ".join(f"{label_map[i]} ({i})" for i in path_ids)

            # ── ORGAN SYSTEM ──
            if PHENO_ID in path_ids:
                idx = path_ids.index(PHENO_ID)
                organ = label_map.get(path_ids[idx + 1], "Other") if idx + 1 < len(path_ids) else "Other"
            else:
                organ = "Other"
            organ_system = CLEAN_ABNORMALITY.sub("", organ).title()

            translations = chpo_map.get(hp_id, [])
            translations_str = ";".join(translations)

            # ── EMIT ROWS ──
            for phrase in [label] + synonyms:
                if not phrase:
                    continue
                clean_phrase = phrase.strip()
                records.append(
                    {
                        "hp_id": hp_id,
                        "phrase": clean_phrase,
                        "raw_phrase": phrase.strip(),
                        "display_phrase": phrase.strip(),
                        "language": "en",
                        "primary_label": label,
                        "translations": translations_str,
                        "organ_system": organ_system,
                        "lineage": lineage_str,
                        "definition": definition,
                        "alt_ids": ";".join(alt_ids),
                        "snomedct": ";".join(snomedct),
                        "umls": ";".join(umls),
                    }
                )

            # 附加中文翻译记录
            for cn_phrase in translations:
                cn_phrase = str(cn_phrase).strip()
                if not cn_phrase:
                    continue
                records.append(
                    {
                        "hp_id": hp_id,
                        "phrase": cn_phrase,
                        "raw_phrase": cn_phrase,
                        "display_phrase": cn_phrase,
                        "language": "zh",
                        "primary_label": label,
                        "translations": translations_str,
                        "organ_system": organ_system,
                        "lineage": lineage_str,
                        "definition": definition,
                        "alt_ids": ";".join(alt_ids),
                        "snomedct": ";".join(snomedct),
                        "umls": ";".join(umls),
                    }
                )

    return pd.DataFrame(
        records,
        columns=[
            "hp_id",
            "phrase",
            "raw_phrase",
            "display_phrase",
            "language",
            "primary_label",
            "translations",
            "organ_system",
            "lineage",
            "definition",
            "alt_ids",
            "snomedct",
            "umls",
        ],
    )


# ────────── 4. Embedding Model Selector ──────────
def get_embedding_model(
    use_sbert: bool = True,
    sbert_model: str = "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb",
    bge_model: str = "BAAI/bge-small-en-v1.5",
):
    if use_sbert:
        logger.log(f"Loading SBERT model: {sbert_model}")
        return SentenceTransformer(sbert_model)
    logger.log(f"Loading BGE model: {bge_model}")
    return TextEmbedding(model_name=bge_model)


# ────────── 5. Vectorize & Save ──────────
def vectorize_dataframe(
    df: pd.DataFrame,
    meta_out: str,
    vec_out: str,
    use_sbert: bool = True,
    sbert_model: str = "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb",
    bge_model: str = "BAAI/bge-small-en-v1.5",
):
    model = get_embedding_model(use_sbert, sbert_model=sbert_model, bge_model=bge_model)

    # ——— New: collect constant metadata once per HP term ———
    constants: dict[str, dict] = {}

    # ——— New: per-embedding metadata (minimal) ———
    entries: list[dict] = []
    embs: list[np.ndarray] = []

    # ——— Compile direction‐detection regexes once per call ———
    NEG_PATTERN = re.compile(r"\b(?:decreas(?:e|ed|ing)?|loss(?:es)?|hypo[-]?\w+)\b", re.IGNORECASE)
    POS_PATTERN = re.compile(r"\b(?:increas(?:e|ed|ing)?|gain(?:s|ed)?|hyper[-]?\w+)\b", re.IGNORECASE)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows", unit="row"):
        info = clean_text_for_embedding(row.phrase)
        raw_phrase = getattr(row, "raw_phrase", row.phrase)
        display_phrase = getattr(row, "display_phrase", raw_phrase)
        language = getattr(row, "language", "en")
        primary_label = getattr(row, "primary_label", raw_phrase)
        translations = getattr(row, "translations", "")

        # ——— More robust direction detection ———
        direction = 0
        if NEG_PATTERN.search(info):
            direction = -1
        elif POS_PATTERN.search(info):
            direction = 1

        # ——— Embedding call unchanged ———
        if use_sbert:
            vec = model.encode(info, convert_to_numpy=True)
        else:
            vec = np.asarray(list(model.embed([info]))[0], dtype=np.float32)

        # ——— New: store only minimal per-info metadata ———
        entries.append(
            {
                "hp_id": row.hp_id,
                "info": info,
                "direction": direction,
                "raw_phrase": raw_phrase,
                "display_phrase": display_phrase,
                "language": language,
                "primary_label": primary_label,
                "translations": translations,
            }
        )

        # ——— New: record the constant fields once for each hp_id ———
        if row.hp_id not in constants:
            constants[row.hp_id] = {
                "organ_system": row.organ_system,
                "lineage": row.lineage,
                "definition": row.definition,
                "alt_ids": row.alt_ids,
                "snomedct": row.snomedct,
                "umls": row.umls,
                "primary_label": primary_label,
                "translations": translations,
            }
        else:
            if translations and not constants[row.hp_id].get("translations"):
                constants[row.hp_id]["translations"] = translations

        embs.append(vec.astype(np.float16))

    # ——— Save embeddings as before ———
    emb_matrix = np.vstack(embs)

    # ——— Write out a single JSON with both parts ———
    combined = {"constants": constants, "entries": entries}
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, separators=(",", ":"))

    np.savez_compressed(vec_out, emb=emb_matrix)
    logger.log(f"Saved {len(entries)} embeddings → {meta_out}, {vec_out}")


def build_knowledge_base(
    obo_url: str = "https://purl.obolibrary.org/obo/hp.obo",
    obo_path: str = "hp.obo",
    refresh_days: int = 14,
    meta_output_path: str = "hpo_meta.json",
    vec_output_path: str = "hpo_embedded.npz",
    hpo_full_csv_path: str = "hpo_terms_full.csv",
    use_sbert: bool = True,
    sbert_model: str = "pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb",
    bge_model: str = "BAAI/bge-small-en-v1.5",
    limit: int = None,  # For testing purposes
    chpo_path: str = None,
):
    """
    Main function to build the HPO knowledge base.
    Downloads OBO, builds dataframe, vectorizes, and saves.
    """
    ensure_hpo_cached(
        obo_url=obo_url,
        obo_path=obo_path,
        refresh_days=refresh_days,
    )
    chpo_map = load_chpo_mapping(chpo_path)
    df = build_hpo_dataframe(
        limit=limit,
        obo_url=obo_url,
        obo_path=obo_path,
        refresh_days=refresh_days,
        chpo_map=chpo_map,
    )
    df.to_csv(hpo_full_csv_path, index=False)
    logger.log(f"Built DataFrame with {len(df)} rows → {hpo_full_csv_path}")
    vectorize_dataframe(
        df,
        meta_out=meta_output_path,
        vec_out=vec_output_path,
        use_sbert=use_sbert,
        sbert_model=sbert_model,
        bge_model=bge_model,
    )
