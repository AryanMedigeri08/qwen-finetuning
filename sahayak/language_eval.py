import argparse
import gc
import hashlib
import json
from pathlib import Path

from eval.generation import build_response_messages, build_route_messages, generate_chat
from eval.metrics import bertscore_f1, intent_metrics, parse_route_json, placeholder_leakage, rouge_l

from . import ROOT
from .model import BASE_REVISION, CATEGORIES, Qwen, adapter_fingerprint
from .research import read_jsonl, write_csv


def evaluate(config, split, output, adapter, batch_size, nf4):
    import torch

    manifest = json.loads((ROOT / "data/splits/manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"][split]
    source = ROOT / "data/splits" / entry["file"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"{split} does not match the training export.")
    rows = read_jsonl(source)
    ids = [row["row_id"] for row in rows]
    qwen = Qwen(tuned=config == "C2", adapter=adapter, nf4=nf4)
    examples = qwen.examples if config == "C1" else []
    response = [build_response_messages(qwen.system, row["instruction"], examples) for row in rows]
    route = [build_route_messages(qwen.router, row["instruction"], examples) for row in rows]
    adapter_id = qwen.adapter
    if adapter_id and Path(adapter_id).is_dir():
        adapter_id = adapter_fingerprint(adapter_id)
    metadata = {
        "config": config, "split": split, "adapter": adapter_id, "base": qwen.base, "base_revision": BASE_REVISION,
        "adapter_revision": qwen.adapter_revision,
        "device": qwen.device, "precision": qwen.precision, "data_manifest_hash": manifest["manifest_hash"],
        "prompt_sha256": hashlib.sha256((qwen.system + "\n" + qwen.router).encode()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=True)
    signature = output / "language_manifest.json"
    if signature.exists() and json.loads(signature.read_text(encoding="utf-8")) != metadata:
        raise ValueError("Existing predictions use different evaluation settings.")
    signature.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    routes = generate_chat(qwen.model, qwen.tokenizer, route, ids, 48, batch_size, output / "routes.jsonl")
    answers = generate_chat(qwen.model, qwen.tokenizer, response, ids, 256, batch_size, output / "answers.jsonl")
    device = qwen.device
    del qwen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pred = [parse_route_json(row["output"], CATEGORIES)[0] for row in routes]
    gold = [row["intent"] for row in rows]
    responses, refs = [row["output"] for row in answers], [row["response"] for row in rows]
    bert, _ = bertscore_f1(responses, refs, batch_size=batch_size, device=device)
    rouge, _ = rouge_l(responses, refs)
    result = metadata | intent_metrics(gold, pred) | {
        "n": len(rows), "bertscore_f1": bert, "rougeL": rouge, "leak_pct": placeholder_leakage(responses),
    }
    write_csv(output / "language.csv", [result])
    if split == "test":
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        labels = sorted(set(gold))
        # Keep rejected and held-out predictions visible with a separate column.
        mapped = [value if value in labels else "Other" for value in pred]
        fig, ax = plt.subplots(figsize=(13, 11))
        ConfusionMatrixDisplay.from_predictions(
            gold, mapped, labels=labels + ["Other"], colorbar=False, xticks_rotation="vertical", ax=ax,
        )
        fig.tight_layout()
        fig.savefig(output / "fig4_router_confusion.png", dpi=160)
        plt.close(fig)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the frozen Qwen prompts against a hashed Bitext split")
    parser.add_argument("--config", choices=["C0", "C1", "C2"], default="C2")
    parser.add_argument("--split", choices=["test", "held_out_test", "eval_mini"], default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--adapter")
    parser.add_argument("--nf4", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.config, args.split, args.output, args.adapter, args.batch_size, args.nf4)
