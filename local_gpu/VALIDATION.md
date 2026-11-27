# Local launcher validation — 2026-10-23

## Result integrity follow-up — 2026-10-26

Independently checked PR #9 at `6af1d5c164c6f013c3273c78f6517a79ec34cfbe`.
The old launcher accepted a zero-exit child with only `status=passed`. Before
the fix, all 11 missing-artifact subcases incorrectly returned success (missing
summaries, original/resumed results, resume comparison, final state/weights and
checkpoint evidence). This reproduces a launcher false-positive path, not an
observed CUDA training failure or a fault in the normal shared suite ordering.

The local launcher now validates the actual smoke configuration against complete
per-seed AdamW/Muon results and summary records, including both resumed reports,
checkpoint steps/state artifacts and successful comparisons at the configured
tolerances. It fails closed without loading tensor files or changing RunPod code.
CPU fixtures use synthetic tensor placeholders and the real stdlib summarizer;
they are not GPU validation evidence.

- `python3 -S -m unittest discover -s tests -v`: 42 tests, 40 passed and 2
  optional CPU ML integration tests skipped because dependencies are absent.
- Regressions cover 11 missing-artifact cases, 39 inconsistent/malformed-artifact
  cases, status-only success, and custom seeds `[9, 17]` with an 8-step/step-4
  checkpoint configuration (including a missing second-seed arm/resume).
- Compile checks, Bash syntax checks and `git diff --check` passed.
- GitHub Actions results belong to the updated PR commit; CPU/offline success
  does not establish RTX 5090 or CUDA training success.

No paid GPU or real GPU training was launched. The workstation validation below
remains outstanding.

## Original implementation checks

Base: merged `main` commit `2499429ca7d6664e72eeb93756a43a563ac1ebef`
(PR #8). These observations describe the implementation environment, **not the
user's RTX 5090 workstation**.

| Check | Observed result |
|---|---|
| `python3 -S -m unittest discover -s tests -v` | 38 tests: 36 passed, 2 skipped |
| Skipped optional tests | Real CPU PyTorch optimizer and Trainer/PEFT integration; dependencies absent |
| `py_compile` on benchmark, training, shared validator and local launcher | Passed |
| `bash -n` on RunPod bootstrap/run/container-start and local bootstrap | Passed |
| `git diff --check` | Passed |
| `local_gpu/smoke.py --check-config` with site packages disabled | Passed; `gpu_validation_passed: false` |
| Actual child execution through `runpod/run.sh` using current Python | Expected failure, exit 1: `No module named 'torch'` |
| Failure report | Both launcher and shared suite reported failed; no completed runs or success summary |
| Failed attempt launcher wall time | Approximately 0.080 seconds; this is startup failure time, not training time |
| GPU visibility | No `nvidia-smi` executable and no `/dev/nvidia*` devices found |

The new tests cover real subprocess CLI execution with no site packages as well
as mocked launcher success/failure cases. Mocked successes test return/status
handling only and are not GPU evidence. No GPU environment installation, gated
model download, cloud GPU allocation or private-data access was performed.

## Outstanding workstation validation

- Install the isolated Python 3.12 / torch 2.9.0 cu128 / existing pinned HF and
  bitsandbytes stack; verify pip dependency checks on the workstation.
- Run the existing single-GPU preflight with the RTX 5090 driver/runtime and
  native bf16 CUDA matmul; verify native `torch.optim.Muon` availability.
- Confirm gated-model access and download/public AG News preparation.
- Verify actual int8 bitsandbytes execution and observed bf16 autocast on the 5090.
- Complete AdamW and Muon training and both step-6 resume comparisons.
- Record measured allocator/device peak memory, step time and training wall time.
  **No GPU VRAM, training time or performance result is available from this session.**

Use the commands in [README.md](README.md). Retain the entire fresh run directory,
including failures. Only a successful real shared suite should produce
`launcher-status.json` with `gpu_validation_passed: true`.
