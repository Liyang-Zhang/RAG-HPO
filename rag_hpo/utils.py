import time
import re
import unicodedata
import os
import sys
import json
import shutil
import pandas as pd
from contextlib import contextmanager
import subprocess

# ======================= Logger Class =======================
class Logger:
    def __init__(self):
        self.printed_messages = set()

    def log(self, msg, once=False):
        """Prints a timestamped message. If once=True, message is only printed once per session."""
        if once:
            # Use a hash of the message to check for duplicates
            msg_hash = hash(msg)
            if msg_hash in self.printed_messages:
                return
            self.printed_messages.add(msg_hash)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

logger = Logger()

def timestamped_print(msg):
    logger.log(msg)

# ======================= Text Cleaning Functions =======================
PAT_PARENTHESES = re.compile(r'\\s*\\([^)]*\\)\\s*')
PAT_NON_ALPHANUMERIC_END = re.compile(r'[^a-zA-Z0-9\\s]+$')

def clean_text_for_embedding(txt: str) -> str:
    """
    Cleans text for embedding by removing content in parentheses,
    extra spaces, trimming, lowercasing, and removing trailing non-alphanumeric characters.
    This version is from HPO_Vectorization.ipynb.
    """
    txt = PAT_PARENTHESES.sub(' ', txt)
    txt = re.sub(r'\\s+', ' ', txt).strip().lower()
    txt = PAT_NON_ALPHANUMERIC_END.sub('', txt)
    return txt

def clean_clinical_note(text: str) -> str:
    """
    Cleans clinical note text by fixing encoding issues, normalizing unicode,
    removing non-ASCII characters, and normalizing spaces.
    This version is from RAG-HPO.ipynb.
    """
    # Fix typical encoding issues
    text = text.encode('latin1', errors='ignore').decode('utf-8', errors='ignore')
    # Normalize unicode (e.g., smart quotes)
    text = unicodedata.normalize("NFKD", text)
    # Remove non-ASCII characters (optional: keep certain ones like µ or – if needed)
    text = re.sub(r'[^\\x00-\\x7F]+', ' ', text)
    # Remove multiple spaces and trim
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

# ======================= Subprocess & Logging =======================
@contextmanager
def managed_subprocess(*args, **kwargs):
    proc = subprocess.Popen(*args, **kwargs)
    try:
        yield proc
    finally:
        proc.terminate()
        proc.wait()

# ======================= State Management (from RAG-HPO.ipynb) =======================
def load_state(temp_files):
    # Loads pipeline state from temp files
    state = {}
    for key, path in temp_files.items():
        try:
            if os.path.exists(path):
                df = pd.read_pickle(path)
                state[key] = df
                # This print is now handled in main
            else:
                state[key] = pd.DataFrame()
        except Exception as e:
            state[key] = pd.DataFrame()
            logger.log(f"Warning loading '{key}': {e}. Starting fresh for this key.")
    return state

def save_state_checkpoint(state, temp_files, keys=('input','combined', 'exact', 'non_exact', 'final')):
    # Saves pipeline state to disk
    for key in keys:
        df = state.get(key)
        if df is None or df.empty:
            continue
        rel_path = temp_files.get(key)
        if not rel_path:
            logger.log(f"Warning: No path configured for '{key}'. Skipping.")
            continue
        abs_path = os.path.abspath(rel_path)
        tmp_path = abs_path + ".tmp"
        try:
            df.to_pickle(tmp_path)
            shutil.move(tmp_path, abs_path)
            logger.log(f"[SAVE] Checkpointed '{key}' ({len(df)} rows) -> {abs_path}", once=True)
        except Exception as e:
            logger.log(f"Error saving '{key}': {e}")
            if os.path.exists(tmp_path): os.remove(tmp_path)

def cleanup(temp_files, success):
    # Removes temp files if pipeline succeeded
    if success:
        logger.log("Pipeline succeeded. Cleaning up temporary files...")
        for path in temp_files.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
                    # logger.log(f"Removed temp file: {path}")
            except OSError as e:
                logger.log(f"Error removing temp file {path}: {e}")
    else:
        logger.log("Pipeline failed. Keeping temporary files for debugging/resume.")
