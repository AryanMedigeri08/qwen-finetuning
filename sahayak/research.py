import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from . import ROOT


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(row) for row in source if row.strip()]


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def report(runs, results, figures):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = json.loads((runs / "metadata.json").read_text(encoding="utf-8"))
    rows = sorted((row for row in metadata if row["scope"] == "scenario" and row.get("n") == 100),
                  key=lambda row: row["config"])
    if not rows:
        raise ValueError("A full configuration run is required for the comparison table.")
    provenance = {(row["device"], row.get("hardware"), row["precision"], row["scenarios_sha256"], row["data_manifest_hash"])
                  for row in rows}
    if len(provenance) != 1:
        raise ValueError("Run the configurations with the same hardware, precision and scenario fixtures.")
    write_csv(results / "main_table.csv", rows)
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(rows))
    ax.bar_label(ax.bar([i - .18 for i in x], [r["tsr_pct"] for r in rows], .36, label="Task success"), fmt="%.0f%%")
    ax.bar_label(ax.bar([i + .18 for i in x], [r["hallucination_flags_pct"] for r in rows], .36,
                       label="Unsupported-claim flag"), fmt="%.0f%%")
    ax.set(xticks=list(x), xticklabels=[r["config"] for r in rows], ylabel="Scenarios (%)", ylim=(0, 112))
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fig1_main.png", dpi=160)
    plt.close(fig)
    ids = [row["config"] for row in rows]
    raw = {cid: read_jsonl(runs / f"{cid.lower()}_scenarios.jsonl") for cid in ids}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([[turn["latency_s"] for row in raw[cid] for turn in row["turns"]] for cid in ids])
    ax.set(xticks=list(range(1, len(ids) + 1)), xticklabels=ids, ylabel="Latency per turn (s)")
    fig.tight_layout()
    fig.savefig(figures / "fig5_latency.png", dpi=160)
    plt.close(fig)
    errors = [{"config": cid, "code": code, "count": count} for cid in ("C4", "C5") if cid in raw
              for code, count in sorted(Counter(row["failure"] for row in raw[cid] if row["failure"]).items())]
    write_csv(results / "error_taxonomy.csv", errors)
    write_csv(results / "e6_safety.csv", [{"config": r["config"], "n": r["n_red"], "safe_pct": r["safe_handling_pct"]}
                                        for r in rows if r.get("n_red") == 40])
    if "C4" in raw:
        from sklearn.metrics import ConfusionMatrixDisplay

        observed = raw["C4"]
        labels = sorted({r["expected_intent"] for r in observed} | {r["intent"] for r in observed if r["intent"]})
        fig, ax = plt.subplots(figsize=(12, 10))
        ConfusionMatrixDisplay.from_predictions(
            [r["expected_intent"] for r in observed], [r["intent"] or "unknown" for r in observed],
            labels=labels, xticks_rotation="vertical", colorbar=False, ax=ax,
        )
        fig.tight_layout()
        fig.savefig(figures / "fig4_scenario_router.png", dpi=160)
        plt.close(fig)


def human_sheet(runs, path):
    c2 = {row["id"]: row for row in read_jsonl(runs / "c2_scenarios.jsonl")}
    c4 = {row["id"]: row for row in read_jsonl(runs / "c4_scenarios.jsonl")}
    common = sorted(c2.keys() & c4.keys())
    if len(common) < 50:
        raise ValueError("Need at least 50 matched scenarios to prepare 100 response ratings.")
    randomizer = random.Random(42)
    rows = []
    for cid in randomizer.sample(common, 50):
        for config, data in (("C2", c2), ("C4", c4)):
            row = data[cid]
            context = "\n".join(f'Customer: {turn["query"]}\nAssistant: {turn["answer"]}' for turn in row["turns"][:-1])
            query = f'{context}\nCustomer: {row["turns"][-1]["query"]}'.strip()
            rows.append({"scenario": cid, "config": config, "query": query,
                         "response": row["turns"][-1]["answer"]})
    randomizer.shuffle(rows)
    write_csv(path.with_name(path.stem + "_key.csv"), [
        {"response_id": i, "scenario": row["scenario"], "config": row["config"]} for i, row in enumerate(rows, 1)
    ])
    write_csv(path, [{"response_id": i, "scenario": row["scenario"], "query": row["query"], "response": row["response"],
                      "correctness": "", "helpfulness": "", "tone": ""}
                     for i, row in enumerate(rows, 1)])


def agreement(paths):
    from sklearn.metrics import cohen_kappa_score

    def load(path):
        with path.open(encoding="utf-8", newline="") as source:
            return {row["response_id"]: row for row in csv.DictReader(source)}

    first, second = (load(path) for path in paths)
    if len(first) != 100 or first.keys() != second.keys():
        raise ValueError("Each rater must score the same 100 responses.")
    scores = {}
    for metric in ("correctness", "helpfulness", "tone"):
        ratings = [[int(rater[response_id][metric]) for response_id in first] for rater in (first, second)]
        if any(value not in range(1, 6) for rater in ratings for value in rater):
            raise ValueError("Each rating must be an integer from 1 to 5.")
        scores[metric] = float(cohen_kappa_score(*ratings, labels=list(range(1, 6)), weights="quadratic"))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the comparison figures and human-evaluation artifacts")
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--figures", type=Path, default=ROOT / "figures")
    parser.add_argument("--human-sheet", type=Path)
    parser.add_argument("--raters", nargs=2, type=Path)
    args = parser.parse_args()
    if args.raters:
        print(json.dumps(agreement(args.raters), indent=2))
    elif args.human_sheet:
        human_sheet(args.runs, args.human_sheet)
    elif args.runs:
        report(args.runs, args.results, args.figures)
    else:
        parser.error("Provide --runs, --human-sheet or --raters.")
