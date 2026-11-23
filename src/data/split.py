
# src/data/split.py -- held-out intents, stratified 80/10/10 split, eval-mini, hashed manifest (T13-T15)
import json
import hashlib
import os
import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_FILES = ["train", "val", "test", "held_out_test", "eval_mini"]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_splits(df, held_out_intents, seed=42):
    held = df[df["intent"].isin(held_out_intents)].copy()
    rest = df[~df["intent"].isin(held_out_intents)].copy()
    train, tmp = train_test_split(rest, test_size=0.2, stratify=rest["intent"], random_state=seed)
    val, test = train_test_split(tmp, test_size=0.5, stratify=tmp["intent"], random_state=seed)
    return {"train": train, "val": val, "test": test, "held_out_test": held}


def make_eval_mini(val, n=300, seed=42):
    mini, _ = train_test_split(val, train_size=n, stratify=val["intent"], random_state=seed)
    return mini


def write_jsonl(df, path):
    df = df.sort_values("row_id")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in df.to_dict("records"):
            f.write(json.dumps(rec, ensure_ascii=False, default=int) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def manifest_hash(seed, held_out_intents, file_hashes):
    payload = json.dumps({"seed": seed, "held_out_intents": sorted(held_out_intents),
                          "files": file_hashes}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(split_dir, splits, held_out_intents, seed, extra=None):
    files, hashes = {}, {}
    for name in SPLIT_FILES:
        path = os.path.join(split_dir, name + ".jsonl")
        write_jsonl(splits[name], path)
        sha = sha256_file(path)
        hashes[name] = sha
        files[name] = {"file": name + ".jsonl", "rows": int(len(splits[name])), "sha256": sha,
                       "per_intent": {k: int(v) for k, v in splits[name]["intent"].value_counts().sort_index().items()}}
    m = {"seed": seed, "held_out_intents": sorted(held_out_intents), "files": files,
         "manifest_hash": manifest_hash(seed, held_out_intents, hashes)}
    if extra:
        m.update(extra)
    with open(os.path.join(split_dir, "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m


def verify_manifest(split_dir):
    # Notebooks 2 and 3 call this instead of recomputing the data. Raises if any file changed.
    with open(os.path.join(split_dir, "manifest.json")) as f:
        m = json.load(f)
    hashes = {}
    for name, info in m["files"].items():
        actual = sha256_file(os.path.join(split_dir, info["file"]))
        if actual != info["sha256"]:
            raise RuntimeError("Split file changed: " + info["file"])
        hashes[name] = actual
    if manifest_hash(m["seed"], m["held_out_intents"], hashes) != m["manifest_hash"]:
        raise RuntimeError("Manifest hash mismatch")
    return m
