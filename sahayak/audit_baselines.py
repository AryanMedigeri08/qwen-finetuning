"""Recompute archived baseline metrics from the frozen references and predictions."""

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from zipfile import ZipFile

from eval.metrics import intent_metrics, parse_route_json, placeholder_leakage, rouge_l

from . import ROOT
from .model import CATEGORIES
from .research import write_csv


def audit(archive_path):
    manifest = json.loads((ROOT / "data/splits/manifest.json").read_text(encoding="utf-8"))
    audit_rows = []
    with ZipFile(archive_path) as archive:
        def csv_rows(path):
            return list(csv.DictReader(io.StringIO(archive.read(f"SahayakLLM/{path}").decode("utf-8"))))

        def jsonl(path):
            return [json.loads(line) for line in archive.read(f"SahayakLLM/{path}").splitlines() if line.strip()]

        logged = {row["run_id"]: row for row in csv_rows("results/results_log.csv")}
        for name, split in (("test", "test"), ("evalmini", "eval_mini")):
            entry = manifest["files"][split]
            ref_path = f'data/splits/{entry["file"]}'
            if hashlib.sha256(archive.read(f"SahayakLLM/{ref_path}")).hexdigest() != entry["sha256"]:
                raise ValueError(f"Unexpected checksum for {split}")
            refs = {row["row_id"]: row for row in jsonl(ref_path)}
            for config in ("B0", "B1"):
                run = f"{config}_{name}"
                routes = {row["id"]: row["output"] for row in jsonl(f"preds/{run}_route.jsonl")}
                responses = {row["id"]: row["output"] for row in jsonl(f"preds/{run}_response.jsonl")}
                if refs.keys() != routes.keys() or refs.keys() != responses.keys():
                    raise ValueError(f"Unmatched prediction IDs in {run}")
                metrics = intent_metrics(
                    [row["intent"] for row in refs.values()],
                    [parse_route_json(routes[row_id], CATEGORIES)[0] for row_id in refs],
                )
                texts = [responses[row_id] for row_id in refs]
                metrics["invalid_json_rate"] = metrics.pop("invalid_rate")
                metrics["rougeL"], _ = rouge_l(texts, [row["response"] for row in refs.values()])
                metrics["leak_pct"] = placeholder_leakage(texts)
                scored = csv_rows(f"preds/{run}_scored.csv")
                if {row["row_id"] for row in scored} != {str(row_id) for row_id in refs} or len(scored) != len(refs):
                    raise ValueError(f"Unmatched BERTScore IDs in {run}")
                metrics["bertscore_f1"] = statistics.mean(float(row["bertscore_f1"]) for row in scored)
                for metric, measured in metrics.items():
                    recorded = float(logged[run][metric])
                    audit_rows.append({
                        "run_id": run, "n": len(refs), "metric": metric, "recorded": recorded, "measured": measured,
                        "method": "Saved per-example BERTScore mean" if metric == "bertscore_f1" else "Prediction replay",
                        "match": math.isclose(recorded, measured, abs_tol=1e-6),
                    })
    return audit_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = audit(args.archive)
    write_csv(args.output, rows)
    failures = [row for row in rows if not row["match"]]
    print(f"{len(rows) - len(failures)} of {len(rows)} baseline checks matched.")
    if failures:
        raise SystemExit(1)
