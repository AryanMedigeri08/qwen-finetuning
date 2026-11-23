import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from . import ROOT, RUNTIME

SPLITS = ROOT / "data/splits"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja")


def restore_splits(source):
    manifest_path = SPLITS / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sft = json.loads((SPLITS / "sft_manifest.json").read_text(encoding="utf-8"))
    expected = {entry["file"]: entry["sha256"] for entry in manifest["files"].values()}
    sft_hashes = {name.removesuffix("_sha256") + ".jsonl": digest for name, digest in sft.items()
                  if name in {"sft_train_sha256", "sft_val_sha256"}}
    def verify(read):
        verified = {}
        for filename, digest in (expected | sft_hashes).items():
            data = read(filename)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Unexpected SHA-256 for {filename}")
            verified[filename] = data
        return verified

    if source.is_dir():
        verified = verify(lambda filename: (source / filename).read_bytes())
    else:
        with ZipFile(source) as archive:
            verified = verify(lambda filename: archive.read(f"SahayakLLM/data/splits/{filename}"))
    for filename, data in verified.items():
        (SPLITS / filename).write_bytes(data)
    print(f"Verified {len(verified)} files for {manifest['manifest_hash']}")


def restore_adapter(archive_path):
    import shutil

    with ZipFile(archive_path) as archive:
        for filename in ADAPTER_FILES:
            archive.getinfo(filename)
        config = json.loads(archive.read("adapter_config.json"))
        if config["base_model_name_or_path"] != "Qwen/Qwen2.5-1.5B-Instruct":
            raise ValueError("This adapter expects a different base model.")
        target = RUNTIME / "C2_adapter"
        target.mkdir(parents=True, exist_ok=True)
        for filename in ADAPTER_FILES:
            with archive.open(filename) as source, (target / filename).open("wb") as destination:
                shutil.copyfileobj(source, destination)
    print(f"Restored {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore the supplied SahayakLLM training artifacts")
    parser.add_argument("--splits", type=Path, help="Frozen split ZIP or its extracted data/splits directory")
    parser.add_argument("--adapter", type=Path)
    options = parser.parse_args()
    if options.splits:
        restore_splits(options.splits)
    if options.adapter:
        restore_adapter(options.adapter)
