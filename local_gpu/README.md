# Local GPU smoke: RTX 5090 32GB

This opt-in launcher reuses `runpod/run.sh`, `runpod/validate.py` and the existing
`runpod/configs/smoke.json`. It does not change RunPod settings, implement another
training loop, launch a paid GPU or use private manufacturing data. The intended
target is a **Linux x86-64 workstation (or Linux under WSL2) with one visible RTX
5090 32GB**. Native Windows/PowerShell is not supported by these Bash scripts.
RTX 5090, WSL2 and the CUDA/bitsandbytes stack remain unverified until a real run
passes; 32GB capacity alone does not establish compatibility.

## Start here

Run from the repository root, with a working NVIDIA driver / `nvidia-smi`, Bash,
Python 3.12 and its `venv` support. Driver installation is a host prerequisite;
these scripts do not modify it. The isolated environment uses the existing
torch 2.9.0 CUDA 12.8 wheel and `runpod/requirements.txt` direct dependency pins.
Allow disk space for wheels, model/data cache, both arms and resume checkpoints;
the existing RunPod storage guidance starts at 80GB. No GPU memory or disk-fit
guarantee has been measured here.

```bash
# Offline CPU configuration check; no ML dependencies, writes or downloads.
python3 -S local_gpu/smoke.py --check-config

# One-time install into .venv-local-gpu, not system Python or the project's uv env.
bash local_gpu/bootstrap.sh

# Llama is gated: obtain access to meta-llama/Llama-3.2-1B-Instruct first.
# Interactive login uses the same HF_HOME/cache as the later smoke run.
.venv-local-gpu/bin/hf auth login

# Real CUDA smoke. Return code 0 requires the shared suite to report passed.
python3 local_gpu/smoke.py
```

The login prompt is preferable to putting a token in shell history. An existing
read-only `HF_TOKEN` can also be supplied privately. The launcher does not dump
environment variables or tokens. Do not commit tokens, caches or result datasets.
The model is Llama 3.2 1B and the dataset is public AG News, exactly as in RunPod.

The GPU run prints its result directory and `console.log` path immediately;
use `tail -f <printed-path>/console.log` and later `suite/adamw-42.log` or
`suite/muon-42.log` to follow progress. It does not stream child logs to the terminal.

## Paths and overrides

| Setting | Default | Override |
|---|---|---|
| Bootstrap Python | `python3.12` | `LOCAL_GPU_BASE_PYTHON` executable |
| Isolated venv | `<repo>/.venv-local-gpu` | `LOCAL_GPU_VENV`; launcher also accepts `--venv` |
| GPU Python | `<venv>/bin/python` | `LOCAL_GPU_PYTHON` or `--python` (explicit executable path) |
| HF cache/auth root | `HF_HOME`, otherwise `$XDG_CACHE_HOME/huggingface` or `~/.cache/huggingface` | `--hf-home` |
| Result parent | `<repo>/local_gpu_runs` | `LOCAL_GPU_OUTPUT_ROOT` or `--output-root` |
| Exact result directory | Unique UTC timestamp + random suffix | `--output` (must not exist) |
| GPU selection | `CUDA_VISIBLE_DEVICES=0` | Set `CUDA_VISIBLE_DEVICES` to one GPU index/UUID |
| Smoke recipe | `<repo>/runpod/configs/smoke.json` | `--config` |

CLI options take precedence over environment variables. Relative overrides are
relative to the invoking working directory; default repository paths work from
any directory. An explicitly empty GPU visibility mask is rejected rather than
silently selecting GPU 0. Paths with spaces are supported. The GPU executable is invoked
through its venv path, preserving venv symlinks. The launcher clears inherited
`PYTHONHOME`/`PYTHONPATH` and disables user site packages in the GPU child.
Set cache variables **before login and execution**. If you already set
`HF_HUB_CACHE` / `HF_DATASETS_CACHE`, Hugging Face's more specific overrides remain
in effect; unset them to keep all caches under `HF_HOME`.

Example using a separate local disk (choose your own existing mount):

```bash
export LOCAL_GPU_VENV="$HOME/venvs/muon-cu128"
export HF_HOME="$HOME/.cache/huggingface"
export LOCAL_GPU_OUTPUT_ROOT="$HOME/muon-results"
export CUDA_VISIBLE_DEVICES=0
bash local_gpu/bootstrap.sh
"$LOCAL_GPU_VENV/bin/hf" auth login
python3 local_gpu/smoke.py --check-config
python3 local_gpu/smoke.py
```

Bootstrap refuses an existing target so it cannot silently upgrade another
environment. Reuse a ready environment without rerunning bootstrap, or choose a
fresh path after a failed install. `--python` selects an existing environment;
it never installs dependencies automatically. The launcher itself needs only
Python 3.9+; bootstrap deliberately requires 3.12 for the pinned GPU environment.

## Preserved validation and evidence

The default config runs 12 optimizer steps per AdamW/Muon arm, saves at step 6,
and resumes each arm in a fresh process to the same total step count. The shared
suite retains single-GPU/process preflight, native bf16 matmul, native Muon,
actual int8 bitsandbytes weights, no CPU/disk offload, observed bf16 autocast,
finite gradients/weights/loss, and checkpoint parameter/optimizer/scheduler/LR
comparisons. The local launcher rejects configs with resume disabled.

Each new run contains:

- `input-config.json`: validated input snapshot. No in-place edits of RunPod configs.
- `launcher.json`: selected paths, command, launcher hash and start time. No token dump.
- `launcher-status.json`: pass/fail, `gpu_validation_passed`, total elapsed seconds,
  exit code, and failure reason. A missing interpreter or failed preflight is a failure.
- `console.log`: combined output/traceback from the shared suite.
- `suite/`: existing `status.json`, environment/driver/package information, pip freeze,
  immutable resolved model/data revisions, worker logs, per-arm `result.json`,
  resume comparisons, `gpu-telemetry.csv` and summary CSV/JSON/Markdown.

Successful per-arm reports contain synchronized step timings, training wall
time and allocator load/train/evaluation peaks; telemetry includes device-wide
memory usage. A failure before model loading has **no measured training time or
peak VRAM**. A worker failure is diagnosed in the corresponding `suite/*.log`;
the shared status may only contain its subprocess exit message. Keep failed runs.
Smoke timings include probes and checkpoint I/O and are **not optimizer performance
claims**. The 3-seed performance recipe is unchanged and remains a separate workflow.

For repeated inputs, copy `suite/resolved-config.json` to your own config path
and pass `--config` on a fresh run; the resolved model/dataset SHAs are retained.
Archive the report and `pip-freeze.txt` with your source commit. Direct requirements
are pinned, but transitive dependencies/host driver are not fully locked, so a new
bootstrap is not promised to reproduce an identical environment or bitwise CUDA
results. Never loosen resume tolerances merely to obtain a pass. For an OOM,
change a separate config for both arms and retain both attempts.

## CPU checks and current boundary

```bash
python3 -S -m unittest discover -s tests -v
python3 -m py_compile local_gpu/smoke.py runpod/validate.py
bash -n local_gpu/bootstrap.sh
python3 -S local_gpu/smoke.py --check-config
```

`tests/test_local_gpu.py` covers offline configuration from another working
directory, path overrides/spaces, venv symlinks, child delegation, failure logs,
missing dependencies, fresh output directories and refusal to overwrite existing
environments/results. It is included in the existing stdlib CI matrix.
Mocked suite passes in those tests are only launcher unit tests.

See [VALIDATION.md](VALIDATION.md) for the implementation-time evidence. CPU
configuration success does not establish CUDA execution, 5090 wheel/kernel
compatibility, 8-bit/bf16 training, GPU resume consistency or peak VRAM. A real
workstation run must verify all of those before this setup is considered validated.
