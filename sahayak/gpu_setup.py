"""Install evaluation dependencies on a preconfigured CUDA notebook image."""

from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from . import ROOT


def install():
    # The notebook image provides a compatible CUDA and numerical stack.
    preserved = {name: version(name) for name in ("torch", "numpy", "scikit-learn", "scipy", "pandas")}
    dependencies = []
    for filename in ("requirements.txt", "requirements-eval.txt", "requirements-gpu.txt"):
        for line in (ROOT / filename).read_text(encoding="utf-8").splitlines():
            name, pin, _ = line.partition("==")
            if pin and name not in preserved and name != "streamlit":
                dependencies.append(line)

    print(f"Python {sys.version.split()[0]} with notebook packages {preserved}", flush=True)
    with TemporaryDirectory() as folder:
        constraints = Path(folder) / "notebook-constraints.txt"
        constraints.write_text("".join(f"{name}=={release}\n" for name, release in preserved.items()), encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet",
             "--constraint", str(constraints), *dependencies], check=True,
        )


def check_cuda():
    import bitsandbytes as bnb
    import torch

    from transformers import Trainer

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA accelerator before setting up the GPU evaluation.")
    bnb.functional.quantize_4bit(torch.zeros(256, device="cuda", dtype=torch.float16), quant_type="nf4")
    print(f"NF4 CUDA preflight passed on {torch.cuda.get_device_name()} with {Trainer.__module__}", flush=True)


if __name__ == "__main__":
    install()
    check_cuda()
