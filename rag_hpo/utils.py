import os
import re
import shutil
import subprocess
import time
import unicodedata
from contextlib import contextmanager

import pandas as pd


HPO_DEBUG = os.getenv("HPO_DEBUG", "").lower() in {"1", "true", "yes", "on"}
HPO_LOG_FILE = os.getenv("HPO_LOG_FILE")


# ======================= Logger Class =======================
class Logger:
    def __init__(self):
        self.printed_messages = set()

    def _write(self, message: str):
        if not HPO_LOG_FILE:
            return
        try:
            with open(HPO_LOG_FILE, "a", encoding="utf-8") as fh:
                fh.write(message + "\n")
        except OSError:
            pass

    def log(self, msg, once=False):
        """
        Print a timestamped message.
        If once=True, the message is only printed once per session.
        """
        if once:
            # Use a hash of the message to check for duplicates
            msg_hash = hash(msg)
            if msg_hash in self.printed_messages:
                return
            self.printed_messages.add(msg_hash)
        full_message = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {msg}"
        print(full_message)
        self._write(full_message)

    def debug(self, msg):
        if HPO_DEBUG:
            self.log(f"[DEBUG] {msg}")


logger = Logger()


def timestamped_print(msg):
    logger.log(msg)


# ======================= Text Cleaning Functions =======================
PAT_PARENTHESES = re.compile(r"\s*\([^)]*\)\s*")
PAT_NON_ALPHANUMERIC_END = re.compile(r"[^a-zA-Z0-9\u4e00-\u9fff\s]+$")


def clean_text_for_embedding(txt: str) -> str:
    """
    Cleans text for embedding by removing content in parentheses,
    extra spaces, trimming, and lowering仅限 ASCII 字母，保留中文等其他字符。
    """
    txt = PAT_PARENTHESES.sub(" ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    txt = "".join(ch.lower() if "A" <= ch <= "Z" else ch for ch in txt)
    txt = PAT_NON_ALPHANUMERIC_END.sub("", txt)
    return txt


def clean_clinical_note(text: str) -> str:
    """
    规范临床文本，同时保留中文、英文及常见符号：
    - Unicode 归一化；
    - 去除控制字符与不可见字符；
    - 折叠多余空白。
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"[ \u00A0\u1680\u2000-\u200B\u202F\u205F\u3000]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
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


def save_state_checkpoint(state, temp_files, keys=("input", "combined", "exact", "non_exact", "final")):
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
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


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
