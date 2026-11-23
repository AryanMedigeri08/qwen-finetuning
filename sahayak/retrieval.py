import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import ROOT, RUNTIME

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
INDEX = RUNTIME / "policy.faiss"
BGE_QUERY = "Represent this sentence for searching relevant passages: "


def read_policy():
    digest = hashlib.sha256()
    chunks = []
    for path in sorted((ROOT / "policies").glob("*.md")):
        raw = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + raw.replace(b"\r\n", b"\n"))
        title, *sections = re.split(r"^## ", raw.decode("utf-8"), flags=re.M)
        intents = re.search(r"<!-- intents: *(.*?) *-->", title).group(1).split()
        for number, section in enumerate(sections, 1):
            heading, paragraph = section.split("\n", 1)
            paragraphs = paragraph.strip().split("\n\n")
            chunks.append({
                "id": f"{path.name} #{number}", "title": title.splitlines()[0].removeprefix("# "),
                "heading": heading.strip(), "answer": paragraphs[0], "text": paragraph.strip(), "intents": intents,
            })
    return chunks, digest.hexdigest()


def tokens(text):
    stop = {"a", "an", "the", "to", "by", "i", "my", "you", "how", "what", "why", "can", "does", "is", "of", "for", "in", "on"}
    return [term for term in re.findall(r"[a-z0-9]+", text.lower()) if term not in stop]


class LexicalSearch:
    """Offline smoke-test retrieval."""
    def __init__(self):
        self.chunks, _ = read_policy()
        self.vectors = [Counter(tokens(f'{c["title"]} {c["heading"]} {c["answer"]}')) for c in self.chunks]

    def search(self, query, k=3):
        words = set(tokens(query))
        ranked = sorted(enumerate(self.vectors), key=lambda item: -sum(item[1][word] for word in words))
        return [self.chunks[index] for index, vector in ranked[:k] if words.intersection(vector)]


class PolicySearch:
    def __init__(self, path=INDEX, force=False):
        import faiss
        from filelock import FileLock
        from sentence_transformers import SentenceTransformer

        self.chunks, digest = read_policy()
        cache = None if os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE") else str(RUNTIME / "hf/hub")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL, revision=EMBEDDING_REVISION, device="cpu", cache_folder=cache)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = path.with_suffix(".json")
        signature = {"policy_sha256": digest, "model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION}
        with FileLock(str(path) + ".lock"):
            if not force and path.exists() and metadata.exists() and json.loads(metadata.read_text()) == signature:
                self.index = faiss.read_index(str(path))
            else:
                docs = [f'{c["title"]} | {c["heading"]}\n{c["text"]}' for c in self.chunks]
                matrix = self.encoder.encode(docs, normalize_embeddings=True, show_progress_bar=False)
                self.index = faiss.IndexFlatIP(matrix.shape[1])
                self.index.add(matrix)
                faiss.write_index(self.index, str(path))
                metadata.write_text(json.dumps(signature), encoding="utf-8")

    def rank(self, queries, k=3):
        matrix = self.encoder.encode(
            [BGE_QUERY + query for query in queries], normalize_embeddings=True, show_progress_bar=False
        )
        _, indices = self.index.search(matrix, min(k, len(self.chunks)))
        return [[self.chunks[int(index)] for index in row] for row in indices]

    def search(self, query, k=3):
        return self.rank([query], k)[0]


def evaluate(search):
    suite = json.loads((ROOT / "benchmarks/retrieval.json").read_text(encoding="utf-8"))
    ranks = []
    for row, results in zip(suite, search.rank([row["question"] for row in suite], 3)):
        ids = [hit["id"] for hit in results]
        ranks.append(dict(row, retrieved=ids, rank=ids.index(row["gold"]) + 1 if row["gold"] in ids else None))
    return {
        "model": EMBEDDING_MODEL, "revision": EMBEDDING_REVISION,
        "policy_sha256": read_policy()[1], "n": len(ranks),
        "recall_at_3": sum(row["rank"] is not None for row in ranks) / len(ranks),
        "mrr_at_3": sum(1 / row["rank"] for row in ranks if row["rank"]) / len(ranks),
        "queries": ranks,
    }


def sweep(search):
    """Compare token windows against the labelled policy sections."""
    import faiss

    suite = json.loads((ROOT / "benchmarks/retrieval.json").read_text(encoding="utf-8"))
    queries = search.encoder.encode([BGE_QUERY + row["question"] for row in suite],
                                    normalize_embeddings=True, show_progress_bar=False)
    by_document = defaultdict(list)
    for chunk in search.chunks:
        by_document[chunk["id"].split(" #")[0]].append(chunk)
    rows = []
    tokenizer = search.encoder.tokenizer
    for size in (200, 300, 500):
        windows = []
        for sections in by_document.values():
            ids, labels = [], []
            for section in sections:
                text = f'{section["title"]}\n{section["heading"]}\n{section["text"]}\n'
                block = tokenizer.encode(text, add_special_tokens=False)
                ids.extend(block)
                labels.extend([section["id"]] * len(block))
            start = 0
            while start < len(ids):
                end = start + size
                windows.append((tokenizer.decode(ids[start:end]), set(labels[start:end])))
                if end >= len(ids):
                    break
                start = end - 50
        encoded = search.encoder.encode([text for text, _ in windows], normalize_embeddings=True, show_progress_bar=False)
        index = faiss.IndexFlatIP(encoded.shape[1])
        index.add(encoded)
        _, hits = index.search(queries, 5)
        for k in (3, 5):
            ranks = [
                next((rank for rank, window in enumerate(indices[:k], start=1)
                      if row["gold"] in windows[int(window)][1]), None)
                for indices, row in zip(hits, suite)
            ]
            rows.append({
                "chunk_tokens": size, "overlap_tokens": 50, "k": k, "n": len(suite), "windows": len(windows),
                "recall_at_k": sum(rank is not None for rank in ranks) / len(ranks),
                "mrr_at_k": sum(1 / rank for rank in ranks if rank) / len(ranks),
            })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and evaluate the policy index")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--sweep", type=Path)
    args = parser.parse_args()
    search = PolicySearch(force=args.rebuild)
    if args.evaluate:
        result = evaluate(search)
        args.evaluate.parent.mkdir(parents=True, exist_ok=True)
        args.evaluate.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print({key: result[key] for key in ("n", "recall_at_3", "mrr_at_3")})
    else:
        print(f"Indexed {len(search.chunks)} policy sections at {INDEX}")
    if args.sweep:
        import csv

        rows = sweep(search)
        args.sweep.parent.mkdir(parents=True, exist_ok=True)
        with args.sweep.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, rows[0])
            writer.writeheader()
            writer.writerows(rows)
