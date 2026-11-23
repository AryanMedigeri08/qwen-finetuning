import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from . import ROOT, RUNTIME
from .benchmark import aggregate, load, run_case
from .db import Store
from .graph import Assistant
from .model import Qwen
from .research import write_csv
from .retrieval import PolicySearch, WindowPolicySearch, WINDOW_OVERLAP, WINDOW_SIZES, read_policy
from .tools import Tools

SCENARIOS = ROOT / "benchmarks/scenarios.json"


def run_setting(cases, model, search, k, seed, target):
    store = Store(target)
    assistant = Assistant(Tools(store, search), model, policy_k=k)
    rows = []
    for case in cases:
        shutil.copyfile(seed, target)
        expected_tools = [
            [name, arguments | {"k": k} if name == "search_policy" else arguments]
            for name, arguments in case["tools"]
        ]
        row = run_case(case | {"tools": expected_tools}, store, model, assistant, "C4", False)
        row["gold_citation"] = case["source"]
        row["retrieved_gold"] = any(
            chunk["id"] == case["source"] for turn in row["turns"] for chunk in turn["chunks"]
        )
        row["cited_gold"] = f'[{case["source"]}]' in row["turns"][-1]["answer"]
        rows.append(row)
    return rows


def evaluate(output, model):
    scenarios = [row for row in load(SCENARIOS) if row["type"] == "information"]
    sections = PolicySearch()
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    model.classify.cache_clear()
    for query in dict.fromkeys(query.strip() for case in scenarios for query in case["turns"]):
        model.classify(query)
    with TemporaryDirectory(dir=RUNTIME) as temp:
        seed, target = [Path(temp) / filename for filename in ("seed.sqlite3", "case.sqlite3")]
        Store(seed).seed()
        for size in (None, *WINDOW_SIZES):
            search = sections if size is None else WindowPolicySearch(sections, size)
            for k in (3,) if size is None else (3, 5):
                model.policy_draft.cache_clear()
                rows = run_setting(scenarios, model, search, k, seed, target)
                label = f"window_{size}_k{k}" if size else "section_k3"
                metrics = dict(
                    setting=label, chunk_tokens=size, overlap_tokens=WINDOW_OVERLAP if size else None, k=k,
                    raw_gold_retrieval_pct=100 * sum(row["retrieved_gold"] for row in rows) / len(rows),
                    exact_gold_citation_pct=100 * sum(row["cited_gold"] for row in rows) / len(rows),
                ) | aggregate(rows)
                summaries.append(metrics)
                (output / f"{label}.jsonl").write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
                )
                print(label, metrics["tsr_pct"], metrics["exact_gold_citation_pct"], flush=True)

    winner = min(
        summaries, key=lambda row: (-row["tsr_pct"], -row["fact_success_pct"], row["k"], row["chunk_tokens"] or 0)
    )
    write_csv(output / "e5_end_to_end.csv", summaries)
    metadata = {
        "scenarios_sha256": hashlib.sha256(SCENARIOS.read_bytes()).hexdigest(),
        "policy_sha256": read_policy()[1],
        "base_revision": model.base_revision,
        "adapter_revision": model.adapter_revision,
        "device": model.device, "precision": model.precision,
        "routing": "Reuse one model decision per query across settings. Timings use precomputed routes.",
        "best_observed_setting": winner["setting"],
        "rank_rule": "Higher cited task success, then higher factual success, then smaller k and smaller windows.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure end-to-end answers and citations across the E5 retrieval settings.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nf4", action="store_true")
    args = parser.parse_args()
    evaluate(args.output, Qwen(nf4=args.nf4))
