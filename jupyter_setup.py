# Jupyter setup cell: install deps, configure HF cache, and login to Hugging Face
# Run this once at the top of the notebook.

# 1) Install required packages (quiet)
import sys
import os
import getpass
from subprocess import check_call, CalledProcessError

try:
    # pip install required libs
    check_call([sys.executable, "-m", "pip", "install",
                "bitsandbytes", "accelerate", "transformers", "datasets",
                "bertviz", "polars", "peft", "tqdm", "evaluate",
                "scikit-learn", "py7zr", "huggingface_hub"],
               stdout=None)
except CalledProcessError as e:
    print("pip install failed:", e)
    # If behind proxy/firewall, install manually or retry

# 2) Optional: change HF cache dir (useful on shared systems)
# Set to a writable directory where you want HF to store caches & tokens.
# Adjust path as needed (e.g., '/project_cache' or '~/hf_cache')
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/project_cache/huggingface")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
print("HF_HOME ->", os.environ["HF_HOME"])

# 3) Disable WANDB auto-logging (optional)
os.environ["WANDB_DISABLED"] = "true"

# 4) Login to Hugging Face securely
# Option A (interactive, recommended in notebooks): supply token via getpass()
from huggingface_hub import login as hf_login

if "HUGGINGFACE_HUB_TOKEN" in os.environ and os.environ["HUGGINGFACE_HUB_TOKEN"].strip():
    token = os.environ["HUGGINGFACE_HUB_TOKEN"].strip()
    print("Using HUGGINGFACE_HUB_TOKEN from environment.")
    hf_login(token=token)
else:
    # This will not echo the token in the notebook output
    token = getpass.getpass("Enter your Hugging Face token (input hidden): ").strip()
    if token:
        hf_login(token=token)
    else:
        print("No token provided. You can set HUGGINGFACE_HUB_TOKEN env var and re-run.")

# 5) Quick smoke tests: transformers + HF hub access
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("PyTorch device available:", "cuda" if torch.cuda.is_available() else "cpu")
# quick model download check (small model)
try:
    small_model = "sshleifer/tiny-gpt2"  # tiny model for smoke check
    tok = AutoTokenizer.from_pretrained(small_model, use_fast=True)
    _ = AutoModelForSequenceClassification.from_pretrained(small_model, num_labels=2)
    print("Hugging Face access OK; tiny model loaded.")
except Exception as e:
    print("Smoke test failed (this is non-fatal):", e)

# 6) Helpful reminders
print("\nReminders:")
print("- If running on Colab/remote, ensure CUDA + bitsandbytes compatibility.")
print("- To avoid interactive prompt in CI, set environment variable HUGGINGFACE_HUB_TOKEN.")
print("- For private repos ensure token has 'read' (or 'write') scope for repo operations.")
