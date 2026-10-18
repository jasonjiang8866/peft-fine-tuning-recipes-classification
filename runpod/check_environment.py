"""Offline image contract check; never runs GPU kernels or downloads a model."""
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    assert sys.version_info[:2] == (3, 12), sys.version
    assert sys.prefix == "/opt/venv", sys.prefix
    assert sys.prefix != sys.base_prefix, "Expected an isolated virtual environment"
    assert sys.executable == os.environ["RUNPOD_PYTHON"] == "/opt/venv/bin/python"
    assert not os.environ.get("HF_TOKEN"), "Offline checks must not receive HF_TOKEN"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("HF_DATASETS_OFFLINE") == "1"
    assert os.environ.get("MUON_SOURCE_COMMIT") == os.environ["EXPECTED_SOURCE_COMMIT"]
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

    # Check the same direct pins used by the Dockerfile, without duplicating them.
    installed = {}
    for line in (root / "runpod/requirements.txt").read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==")
        installed[name] = version(name)
        assert installed[name] == expected, (name, installed[name], expected)
    # Metadata/pip check alone will not detect broken shared libraries or lazy imports.
    for name in ("torch", "transformers", "accelerate", "peft", "bitsandbytes",
                 "datasets", "numpy", "sklearn", "huggingface_hub", "safetensors"):
        importlib.import_module(name)
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorWithPadding, Trainer,
                              TrainerCallback, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from accelerate.optimizer import AcceleratedOptimizer
    from datasets import DatasetDict, load_dataset, load_from_disk
    from sklearn.metrics import accuracy_score, f1_score
    from bitsandbytes.nn import Linear8bitLt
    import torch

    installed["torch"] = torch.__version__
    assert torch.__version__.split("+")[0] == "2.9.0", torch.__version__
    assert torch.version.cuda == "12.8", torch.version.cuda
    assert hasattr(torch.optim, "Muon"), "Native Muon missing"
    assert not torch.cuda.is_available(), "This check is intentionally CPU-only"
    sys.path.insert(0, str(root))
    from benchmark_muon import BenchmarkRun, trainable_parameters
    from runpod.validate import read_config
    for profile in ("smoke", "performance"):
        read_config(root / f"runpod/configs/{profile}.json")
    print(json.dumps({"python": sys.version, "executable": sys.executable,
                      "prefix": sys.prefix, "versions": installed,
                      "torch_cuda_build": torch.version.cuda,
                      "cuda_available": torch.cuda.is_available(),
                      "source_commit": os.environ["MUON_SOURCE_COMMIT"],
                      "scope": "image imports/configs only; no GPU training"}, indent=2))


if __name__ == "__main__":
    main()
