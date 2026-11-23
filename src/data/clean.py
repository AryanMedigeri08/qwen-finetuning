
# src/data/clean.py -- deterministic cleaning for the Bitext rows (T11)
import re
import pandas as pd

_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u00a0": " ", "\u200b": ""}


def clean_text(s):
    if s is None or (isinstance(s, float) and s != s):
        return ""
    s = str(s)
    for k, v in _QUOTES.items():
        s = s.replace(k, v)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)       # collapse runs of spaces/tabs, keep newlines (numbered lists)
    s = re.sub(r" *\n *", "\n", s)      # no spaces around newlines
    s = re.sub(r"\n{3,}", "\n\n", s)    # at most one blank line
    return s.strip()


def clean_dataframe(df):
    # expects columns: row_id, flags, instruction, category, intent, response
    log = {"rows_in": int(len(df))}
    df = df.copy()
    for col in ["instruction", "response"]:
        df[col] = df[col].map(clean_text)
    for col in ["flags", "intent", "category"]:
        df[col] = df[col].astype(str).str.strip()
    empty = (df["instruction"] == "") | (df["response"] == "")
    log["dropped_empty"] = int(empty.sum())
    df = df[~empty]
    before = len(df)
    df = df.drop_duplicates(subset=["instruction", "response", "intent"], keep="first")
    log["dropped_exact_duplicate_pairs"] = int(before - len(df))
    df = df.reset_index(drop=True)
    log["rows_out"] = int(len(df))
    return df, log
