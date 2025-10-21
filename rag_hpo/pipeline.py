import os
import sys
import time
import json
import re
import traceback
import tiktoken
import pandas as pd
import numpy as np
import requests
import faiss
from typing import List, Dict, Any
from fastembed import TextEmbedding
from rapidfuzz import fuzz as rfuzz
from tabulate import tabulate
from tqdm import tqdm
from collections import defaultdict
from sentence_transformers import SentenceTransformer

from .utils import logger, clean_clinical_note, load_state, save_state_checkpoint, cleanup


# ======================= LLM Client =======================
class LLMClient:
    def __init__(self, api_key, base_url, model_name="llama3-groq-70b-8192-tool-use-preview", max_tokens_per_day=500000, max_queries_per_minute=30, temperature=0.7):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_tokens_per_day = max_tokens_per_day
        self.max_queries_per_minute = max_queries_per_minute
        self.temperature = temperature
        self.total_tokens_used = 0
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        try:
            self.encoder = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    def query(self, user_input, system_message):
        # Sends a query to the LLM API and tracks token usage
        tokens_ui = len(self.encoder.encode(user_input))
        tokens_sys = len(self.encoder.encode(system_message))
        estimated = tokens_ui + tokens_sys
        if self.total_tokens_used + estimated > self.max_tokens_per_day:
            raise Exception("Token limit exceeded for the day.")
        time.sleep(60 / self.max_queries_per_minute)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_input}
            ],
            "temperature": self.temperature,
        }
        resp = requests.post(self.base_url, headers=self.headers, json=payload)
        resp.raise_for_status()
        result = resp.json()
        if "usage" in result and "total_tokens" in result["usage"]:
            self.total_tokens_used += result["usage"]["total_tokens"]
        else:
            self.total_tokens_used += estimated
        choices = result.get("choices") or []
        return choices[0].get("message", {}).get("content", "") if choices else ""

# ======================= Prompt Loading =======================
def load_prompts(file_path="system_prompts.json"):
    if not os.path.exists(file_path):
        logger.log(f"Error: Prompt file '{file_path}' not found.")
        sys.exit(1)
    with open(file_path, "r") as f:
        return json.load(f)

prompts = load_prompts()
system_message_I = prompts.get("system_message_I", "")
system_message_II = prompts.get("system_message_II", "")

# ======================= Environment Setup (to be refactored) =======================
llm_client = None

def check_and_initialize_llm(api_key=None, base_url=None, model_name=None, max_tokens_per_day=500000, max_queries_per_minute=30, temperature=0.7):
    global llm_client
    if api_key and base_url and model_name:
        llm_client = LLMClient(api_key, base_url, model_name, max_tokens_per_day, max_queries_per_minute, temperature)
        return True
    return False

# ======================= Embeddings & FAISS =======================
PAT_CLEAN_TEXT_EMBEDDING = re.compile(r'\\s*\\([^)]*\\)') # This was in HPO_Vectorization.ipynb, but also used in RAG-HPO.ipynb
                                                    # I'll use the one from utils.py
def initialize_embeddings_model(use_sbert: bool = True, sbert_model: str = 'pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb', bge_model: str = 'BAAI/bge-small-en-v1.5'):
    try:
        if use_sbert:
            return SentenceTransformer(sbert_model)
        return TextEmbedding(model_name=bge_model)
    except Exception as e:
        logger.log(f"[FATAL] Could not initialize embedding model: {e}")
        sys.exit(1)

def load_vector_db(meta_path: str = 'hpo_meta.json',
                   vec_path:  str = 'hpo_embedded.npz'):
    # ─── Sanity checks ───
    if not os.path.exists(meta_path) or not os.path.exists(vec_path):
        logger.log(f"[FATAL] DB files not found: {meta_path}, {vec_path}")
        sys.exit(1)

    # ─── Load the condensed JSON ───
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            combined = json.load(f)
            constants = combined.get('constants', {})
            entries  = combined.get('entries', [])
    except Exception as e:
        logger.log(f"[FATAL] Could not load metadata JSON: {e}")
        sys.exit(1)

    # ─── Load the embeddings ───
    try:
        arr = np.load(vec_path)
        emb_matrix = arr['emb'].astype(np.float32)
    except Exception as e:
        logger.log(f"[FATAL] Could not load embedding npz: {e}")
        sys.exit(1)

    # ─── Warn if lengths mismatch ───
    if len(entries) != emb_matrix.shape[0]:
        logger.log(f"[WARN] Metadata entries count and embedding rows mismatch "
              f"({len(entries)} vs {emb_matrix.shape[0]})")

    # ─── Reconstruct docs list in the original output format ───
    docs = []
    for entry, vec in zip(entries, emb_matrix):
        hp_id = entry.get('hp_id')
        const = constants.get(hp_id, {})

        doc = {
            'hp_id':          hp_id,
            'info':           entry.get('info'),
            'lineage':        const.get('lineage'),
            'organ_system':   const.get('organ_system'),
            'direction':      entry.get('direction'),
            # preserve these keys even if absent in the new JSON:
            'depth':          const.get('depth'),
            'parent_count':   const.get('parent_count'),
            'child_count':    const.get('child_count'),
            'descendant_count': const.get('descendant_count'),
            'embedding':      vec
        }
        docs.append(doc)

    return docs, emb_matrix

def create_faiss_index(emb_matrix: np.ndarray, metric: str = 'cosine'):
    # Build FAISS index for embeddings
    dim = emb_matrix.shape[1]
    if metric == 'cosine':
        faiss.normalize_L2(emb_matrix)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(emb_matrix)
    return index

def embed_query(text: str, model, metric: str = 'cosine'):
    if hasattr(model, 'encode'):
        vec = model.encode(text, convert_to_numpy=True)
    else:
        vec = np.array(list(model.embed([text]))[0], dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)
    if metric == 'cosine':
        faiss.normalize_L2(vec)
    return vec

# ======================= Phenotype Processing =======================
def _collect_metadata_best(
    phrase: str,
    query_vec: np.ndarray,
    index: faiss.Index,
    docs: List[Dict[str, Any]],
    top_k: int = 500,
    similarity_threshold: float = 0.35,
    min_unique: int = 15,
    max_unique: int = 20
) -> List[Dict[str, Any]]:
    """
    Single‐pass hybrid retrieval: token overlap, threshold, fill to min_unique.
    """
    clean_tokens = set(re.findall(r'\\w+', phrase.lower()))
    dists, idxs = index.search(query_vec, top_k)
    sims, indices = dists[0], idxs[0]

    seen_hp = set()
    results = []

    for sim, idx in sorted(zip(sims, indices), key=lambda x: x[0], reverse=True):
        if len(results) >= max_unique:
            break
        doc = docs[idx]
        hp = doc.get('hp_id')
        if not hp or hp in seen_hp:
            continue

        info = doc.get('info', '') or ''
        token_overlap = bool(clean_tokens & set(re.findall(r'\\w+', info.lower())))

        # accept if token overlap, or above similarity threshold, or to reach min_unique
        if token_overlap or sim >= similarity_threshold or len(results) < min_unique:
            seen_hp.add(hp)
            results.append({
                'hp_id': hp,
                'phrase': info,
                'definition': doc.get('definition'),
                'organ_system': doc.get('organ_system'),
                'similarity': float(sim)
            })

    return results

def split_exact_nonexact(df: pd.DataFrame, hpo_term_col='HPO_Term'):
    if 'category' not in df.columns:
        raise KeyError(f"[FATAL] 'category' missing; columns: {df.columns.tolist()}")
    if hpo_term_col not in df.columns:
        raise KeyError(f"[FATAL] '{hpo_term_col}' missing; columns: {df.columns.tolist()}")

    # only keep Abnormal rows
    df_ab = df[df['category'] == 'Abnormal']

    exact_df     = df_ab.dropna(subset=[hpo_term_col]).copy()
    non_exact_df = df_ab[df_ab[hpo_term_col].isna()].copy()
    return exact_df, non_exact_df

def process_findings(findings, clinical_note: str, embeddings_model, index, docs,
                    metric: str = 'cosine',
                    keep_top: int = 15):
    """
    Processes findings and returns DataFrame with phrase, category,
    metadata, sentence, patient_id.
    - keep_top: number of unique metadata entries to retrieve.
    """
    # ─── Position A: Split note into sentences for context matching ───
    sentences = [s.strip() for s in clinical_note.split('.') if s.strip()]
    rows = []

    for f in findings:
        phrase = f.get('phrase', '').strip()
        category = f.get('category', '')
        if not phrase:
            continue

        # ─── Position B: Embed the phrase ───
        qv = embed_query(phrase, embeddings_model, metric=metric)

        # ─── Position C: Retrieve best metadata candidates ───
        unique_metadata = _collect_metadata_best(
            phrase=phrase,               # literal text for token-phase
            query_vec=qv,                # embedded vector
            index=index,                 # FAISS index
            docs=docs,                   # list of HPO docs
            top_k=500,                   # FAISS retrieval size
            similarity_threshold=0.35,   # max distance for semantic matches
            min_unique=keep_top,         # ensure at least keep_top entries
            max_unique=keep_top          # cap at keep_top entries
        )

        # ─── Position D: Find the best-matching sentence ───
        fw = set(re.findall(r'\\b\\w+\\b', phrase.lower()))
        best_sent, best_score = None, 0
        for s in sentences:
            sw = set(re.findall(r'\\b\\w+\\b', s.lower()))
            score = len(fw & sw)
            if score > best_score:
                best_score, best_sent = score, s

        # ─── Position E: Collect row ───
        rows.append({
            'phrase':           phrase,
            'category':         category,
            'unique_metadata':  unique_metadata,
            'original_sentence': best_sent,
            'patient_id':       f.get('patient_id')
        })

    return pd.DataFrame(rows)

def clean_and_parse(s: str):
    # Extracts and parses JSON from string
    try:
        m = re.search(r'\\{.*\\}', s, flags=re.S)
        js_str = m.group(0) if m else s.strip()
        return json.loads(js_str)
    except Exception:
        return None

def extract_findings(response: str) -> list:
    # Extracts findings from LLM response
    if not response:
        return []
    parsed = clean_and_parse(response)
    if not isinstance(parsed, dict):
        return []
    return parsed.get("phenotypes", [])

# ======================= Single-Row Processing =======================
def process_row(clinical_note, system_message, embeddings_model, index, embedded_documents):
    # Clean the clinical note before sending to the LLM
    clinical_note = clean_clinical_note(clinical_note)
    # Query the LLM
    raw = llm_client.query(clinical_note, system_message)
    logger.log(f"LLM raw response: {raw[:500]}...")  # Log first 500 chars

    # Try to parse findings robustly
    findings = extract_findings(raw)
    logger.log(f"Parsed findings: {findings[:5]}...") # Log first 5 findings
    # If findings is empty, try to parse as a list of dicts (sometimes LLMs return just a list)
    if not findings:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                findings = parsed
        except Exception:
            pass

    # If still empty, try to extract any dicts with 'phrase' and 'category' keys
    if not findings:
        try:
            matches = re.findall(r'\\{[^\\}]*\\}', raw)
            findings = []
            for m in matches:
                try:
                    d = json.loads(m)
                    if 'phrase' in d and 'category' in d:
                        findings.append(d)
                except Exception:
                    continue
        except Exception:
            pass

    findings = [f for f in findings if isinstance(f, dict) and f.get('category') == 'Abnormal']

    # Return empty DataFrame if no valid findings, but with required columns
    required_cols = ['phrase', 'category', 'unique_metadata', 'original_sentence', 'patient_id']
    if not findings:
        return pd.DataFrame(columns=required_cols)
    # Continue processing
    df = process_findings(findings, clinical_note, embeddings_model, index, embedded_documents)
    # Ensure all required columns are present
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df

# ======================= HPO Term Extraction =======================
def _iter_term_hp(item):
    # Yields (term_text, hp_id) pairs from metadata entries
    if isinstance(item, str):
        try:
            d = json.loads(item)
        except Exception:
            return
    elif isinstance(item, dict):
        d = item
    else:
        return
    if "hp_id" in d:
        term_text = d.get("info") or d.get("label")
        hp = d["hp_id"]
        if hp and term_text:
            yield term_text, hp
        return
    for k, v in d.items():
        if isinstance(v, str) and v.startswith("HP:"):
            yield k, v

def build_cluster_index(metadata_list):
    # Builds cluster index for bag-of-words matching
    idx = defaultdict(lambda: defaultdict(list))
    for entry in metadata_list:
        for term, hp in _iter_term_hp(entry):
            ct = clean_clinical_note(term) # Using clean_clinical_note for consistency
            if not ct:
                continue
            toks = ct.split()
            sig = " ".join(sorted(toks))
            idx[sig][len(toks)].append(hp)
    return idx

def extract_hpo_term(phrase, metadata_list, cluster_index):
    # Maps phenotype phrase to best HPO term using multiple strategies
    if not metadata_list or (isinstance(metadata_list, float) and pd.isna(metadata_list)):
        return None
    cp = clean_clinical_note(phrase) # Using clean_clinical_note for consistency
    if not cp:
        return None
    toks = cp.split()
    sig = " ".join(sorted(toks))
    if sig in cluster_index and len(toks) in cluster_index[sig]:
        return cluster_index[sig][len(toks)][0]
    pairs = []
    for entry in metadata_list:
        for term, hp in _iter_term_hp(entry):
            ct = clean_clinical_note(term) # Using clean_clinical_note for consistency
            if ct:
                pairs.append((ct, hp))
    pset = set(toks)
    for ct, hp in pairs:
        if set(ct.split()) == pset:
            return hp
    for ct, hp in pairs:
        if ct == cp:
            return hp
    if len(pset) > 1:
        for ct, hp in pairs:
            if re.search(rf"\\b{re.escape(ct)}\\b", cp):
                return hp
    best_hp, best_score = None, 0
    for ct, hp in pairs:
        score = rfuzz.token_sort_ratio(cp, ct)
        if score > best_score:
            best_hp, best_score = hp, score
    return best_hp if best_score >= 80 else None

def parse_llm_mapping(resp_text: str, candidate_ids: set) -> (str, str, dict):
    """
    Simplified parsing: strict JSON → key lookup → regex fallback.
    """
    # 1. Strict JSON parse
    try:
        js = json.loads(resp_text)
    except json.JSONDecodeError:
        js = None

    # 2. Look for an HPO ID in known keys
    if isinstance(js, dict):
        candidate = next(
            (js[k].strip().strip('\"') for k in ("hpo_id", "HPO_ID", "hp_id", "id")
             if isinstance(js.get(k), str)),
            None
        )
        if candidate:
            low = candidate.lower()
            if low in ("null", "no candidate fit"):
                return None, "null_label", js
            if candidate in candidate_ids:
                return candidate, "ok", js
            return candidate, "hp_not_in_candidates", js

    # 3. Regex fallback for HP:NNNNNNN
    for m in re.findall(r"(HP:\\d{6,7})", resp_text):
        if m in candidate_ids:
            return m, "regex_fallback", None

    # 4. Nothing found
    return None, "no_hpo_found", None

def generate_hpo_terms(df_row: pd.DataFrame, system_message: str) -> pd.DataFrame:
    """
    Streamlined LLM + fallback logic with phrase normalization.
    """
    # 1. Extract & normalize inputs
    phrase = df_row['phrase'].iloc[0].strip()
    normalized = phrase.lower().replace('-', ' ').strip()
    category = df_row['category'].iloc[0]
    original = df_row['original_sentence'].iloc[0]
    metadata_list = df_row['unique_metadata'].iloc[0] or []

    # 2. Build candidate list
    candidates = []
    seen = set()
    for m in metadata_list:
        term = m.get('phrase') or m.get('info')
        hp = m.get('hp_id')
        if term and hp and hp not in seen:
            candidates.append({'term': term, 'id': hp})
            seen.add(hp)
    candidate_ids = {c['id'] for c in candidates}

    # 3. Call LLM and parse
    payload = json.dumps({
        'phrase': normalized,
        'category': category,
        'original_sentence': original,
        'candidates': candidates
    })
    resp = llm_client.query(payload, system_message)
    hpo_id, reason, _ = parse_llm_mapping(resp, candidate_ids)

    # 4. Local fallback if needed
    if not hpo_id:
        cluster_idx = build_cluster_index(metadata_list)
        local_id = extract_hpo_term(normalized, metadata_list, cluster_idx)
        if local_id:
            hpo_id, reason = local_id, "fallback_local"

    # 5. Return unified record
    return pd.DataFrame([{
        'HPO_Terms': [{'phrase': phrase, 'HPO_Term': hpo_id}],
        'raw_llm_resp': resp,
        'llm_parse_reason': reason
    }])

def validate_input(df):
    if 'clinical_note' not in df.columns:
        raise KeyError("Missing required column: 'clinical_note'.")
    df = df.dropna(subset=['clinical_note']).copy()
    df['clinical_note'] = df['clinical_note'].astype(str)
    # Clean all clinical notes to prevent encoding-related bugs
    df['clinical_note'] = df['clinical_note'].apply(clean_clinical_note)
    if 'patient_id' not in df.columns:
        df = df.reset_index(drop=True)
        df['patient_id'] = df.index + 1
    else:
        df['patient_id'] = df['patient_id'].astype(int)
    return df

def process_results(final_df, output_csv_path=None, output_json_raw_path=None, display_results=False):
    # Handles output of final results (CSV or display)
    if final_df.empty:
        logger.log("No final results to process.")
        return

    if output_csv_path:
        rows = []
        for idx, r in final_df.iterrows():
            pid = r.get('patient_id', idx)
            for term in r.get('HPO_Terms', []):
                ph = term.get('phrase', '').strip()
                cat = term.get('category', '')
                hp = term.get('HPO_Term') or ''
                if isinstance(hp, str):
                    hp = hp.replace("HP:HP:", "HP:")
                if hp:
                    rows.append({
                        'Patient ID': pid,
                        'Category': cat,
                        'Phenotype name': ph,
                        'HPO ID': hp
                    })
                else:
                    logger.log(f"Warning: Blank HPO_Term for patient {pid}, phrase '{ph}', category '{cat}' - not included in CSV.")
        if rows:
            output_df = pd.DataFrame(rows)
            output_df.to_csv(output_csv_path, index=False)
            logger.log(f"Saved tabular results to {output_csv_path}")
        else:
            logger.log("No valid HPO terms to save in tabular format.")
        if output_json_raw_path:
            final_df.to_csv(output_json_raw_path, index=False)
            logger.log(f"Saved raw JSON results to {output_json_raw_path}")
    
    if display_results:
        tbl = []
        for idx, r in final_df.iterrows():
            pid = r.get('patient_id', idx)
            for term in r.get('HPO_Terms', []):
                ph = term.get('phrase', '').strip()
                cat = term.get('category', '')
                hp = term.get('HPO_Term') or ''
                if isinstance(hp, str):
                    hp = hp.replace("HP:HP:", "HP:")
                tbl.append({
                    'Case': f"Case {pid}",
                    'Category': cat,
                    'Phenotype name': ph,
                    'HPO ID': hp
                })
        if tbl:
            print(tabulate(pd.DataFrame(tbl), headers='keys', tablefmt='psql'))
        else:
            logger.log("No terms to display.")

def run_rag_pipeline(
    input_data: pd.DataFrame,
    output_csv_path: str = None,
    output_json_raw_path: str = None,
    display_results: bool = False,
    meta_path: str = 'hpo_meta.json',
    vec_path: str = 'hpo_embedded.npz',
    sbert_model: str = 'pritamdeka/SapBERT-mnli-snli-scinli-scitail-mednli-stsb',
    bge_model: str = 'BAAI/bge-small-en-v1.5',
    use_sbert: bool = True,
    api_key: str = None,
    base_url: str = None,
    llm_model_name: str = "llama3-groq-70b-8192-tool-use-preview",
    max_tokens_per_day: int = 500000,
    max_queries_per_minute: int = 30,
    temperature: float = 0.7,
    system_prompts_file: str = "system_prompts.json"
):
    """
    Main function to run the RAG pipeline.
    """
    global llm_client, system_message_I, system_message_II

    # Initialize LLM client
    if not check_and_initialize_llm(api_key, base_url, llm_model_name, max_tokens_per_day, max_queries_per_minute, temperature):
        logger.log("[FATAL] LLM client could not be initialized. Please provide API key, base URL, and model name.")
        sys.exit(1)

    # Load prompts
    prompts = load_prompts(system_prompts_file)
    system_message_I = prompts.get("system_message_I", "")
    system_message_II = prompts.get("system_message_II", "")
    if not system_message_I or not system_message_II:
        logger.log("[FATAL] System prompts not loaded correctly. Check system_prompts.json.")
        sys.exit(1)

    logger.log("Starting HPO extraction pipeline...")
    start_time = time.time()

    # 2) load checkpointed state (simplified for CLI, no interactive input)
    # For a CLI, we assume fresh run or explicit resume. 
    # If checkpointing is desired, it should be managed by the CLI arguments.

    # 3) Ingest notes
    df_input = validate_input(input_data)

    # 4) initialize models and indices
    emb_model    = initialize_embeddings_model(use_sbert=use_sbert, sbert_model=sbert_model, bge_model=bge_model)
    docs, emb_matrix = load_vector_db(meta_path=meta_path, vec_path=vec_path)
    index        = create_faiss_index(emb_matrix, metric='cosine')
    cluster_index = build_cluster_index(docs)

    success = False
    try:
        # 5) process raw clinical notes
        logger.log("Processing clinical notes...")
        combined = pd.DataFrame()
        pids     = sorted(df_input['patient_id'].unique())
        total    = len(pids)
        for i, pid in enumerate(tqdm(pids, desc="Processing Notes", unit="note")):
            note = df_input.loc[df_input['patient_id'] == pid, 'clinical_note'].iloc[0]
            res  = process_row(note, system_message_I, emb_model, index, docs)
            if not res.empty:
                res['patient_id'] = pid
                combined = pd.concat([combined, res], ignore_index=True)
        
        # 6) split into exact vs non-exact
        logger.log("Splitting exact vs non-exact…")
        df = combined.copy()

        # ensure HPO_Term column is present
        if 'HPO_Term' not in df.columns:
            df['HPO_Term'] = np.nan

        # compute HPO_Term for any missing
        if not df.empty:
            df['HPO_Term'] = (
                df.apply(
                    lambda r: extract_hpo_term(r['phrase'], r['unique_metadata'], cluster_index)
                            if pd.isna(r['HPO_Term']) else r['HPO_Term'],
                    axis=1
                )
                .astype(object)
                .where(lambda x: pd.notna(x), np.nan)
            )

        exact_df, non_exact_df = split_exact_nonexact(df, hpo_term_col='HPO_Term')

        # 7) post-process non-exact entries
        non_ex = non_exact_df.copy()
        for col in ("llm_parse_reason", "raw_llm_resp"):
            non_ex[col] = non_ex.get(col, pd.Series(dtype="object")).astype("object")

        idxs = non_ex[
            (non_ex['category'] == 'Abnormal') &
            (non_ex['HPO_Term'].isna()) &
            (non_ex['HPO_Term'] != "No Candidate Fit")
        ].index

        if not idxs.empty:
            logger.log(f"Generating HPO for {len(idxs)} entries...")
            for i, idx in enumerate(tqdm(idxs, desc="Generating HPO", unit="entry")):
                row_df = non_ex.loc[[idx]]
                out_df = generate_hpo_terms(row_df, system_message_II)
                hp     = (out_df.at[0, 'HPO_Terms'][0]['HPO_Term']
                          if not out_df.empty else None)
                # attach parse reason & raw response
                if 'llm_parse_reason' in out_df.columns:
                    non_ex.at[idx, 'llm_parse_reason'] = out_df.at[0, 'llm_parse_reason']
                if 'raw_llm_resp' in out_df.columns:
                    non_ex.at[idx, 'raw_llm_resp']       = out_df.at[0, 'raw_llm_resp']
                non_ex.at[idx, 'HPO_Term'] = hp or "No Candidate Fit"
        else:
            logger.log("No non-exact entries to process.")

        # 8) compile final results
        logger.log("Compiling final results...")
        merged = pd.concat([exact_df, non_ex], ignore_index=True)
        merged = merged.dropna(subset=['HPO_Term'])
        if not merged.empty:
            grouped = (
                merged
                  .groupby('patient_id')[['phrase','category','HPO_Term']]
                  .apply(lambda g: g.to_dict('records'))
                  .reset_index(name='HPO_Terms')
            )
            final_df = grouped.copy()
        else:
            final_df = pd.DataFrame(columns=['patient_id','HPO_Terms'])

        # 9) output
        process_results(final_df, output_csv_path, output_json_raw_path, display_results)
        success = True
        logger.log(f"Pipeline completed in {time.time() - start_time:.2f}s")

    except Exception as e:
        logger.log(f"Pipeline error: {e}. Exiting.")
        traceback.print_exc()
        sys.exit(1) # Exit on error

    finally:
        # Cleanup is handled by the CLI now, not automatically here.
        pass
