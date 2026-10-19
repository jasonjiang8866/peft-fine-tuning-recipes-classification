# RunPod: native Muon / AdamW GPU validation

This is an **opt-in single-GPU Pod workload**, not a Serverless handler. It reuses
`BenchmarkRun.trainer_kwargs`, the parameter audit, `OptimizerPair`, and synchronized
step timer from `benchmark_muon.py`. Neither `training.py`, `pyproject.toml` nor
`uv.lock` is changed. The separate environment deliberately selects torch 2.9.0
with CUDA 12.8 and pinned direct HF/bitsandbytes dependencies.

## Fastest path: existing RunPod GPU Pod

Select a Linux x86-64 Pod with Python 3.12, `python3-venv`, git and a recent NVIDIA
driver compatible with CUDA 12.8. Use **one A40 48GB or A100** for comfortable memory
headroom; other Ampere-or-newer GPUs can be tried using the small smoke profile.
Do not pick a T4/V100 for this native-bf16 template. Mount storage at `/workspace`;
start with 80 GB for wheels, HF cache, datasets, checkpoints and reports.

Accept the Llama 3.2 model license/access request on Hugging Face and inject a read
`HF_TOKEN` as a Pod environment secret. Do not put the token in git, a Docker build
argument, a shared template, or a shell command recorded in history.

In the Pod terminal:

```bash
cd /workspace
git clone --branch main --single-branch \
  https://github.com/jasonjiang8866/peft-fine-tuning-recipes-classification.git
cd peft-fine-tuning-recipes-classification
# Optional: match the source revision of the published image below.
# git checkout bfdf5d3580036b11876fe2264c6d92499913c681
bash runpod/bootstrap.sh

# Each output directory must be new. Both arms run in separate Python processes.
bash runpod/run.sh --config runpod/configs/smoke.json \
  --output /workspace/results/smoke-001

# Run only after smoke (including both resume comparisons) passes.
bash runpod/run.sh --config runpod/configs/performance.json \
  --output /workspace/results/performance-001
```

`bootstrap.sh` creates `/workspace/venvs/muon-cu128` without inheriting the Pod's
site packages. It does not `uv sync` or install the whole project. Override
`RUNPOD_VENV` when installing and set `RUNPOD_PYTHON` to the resulting interpreter
when running if you choose another location. Provision this environment once per
Pod; do not run bootstrap concurrently with training.

The driver is supplied by the RunPod host. CUDA 12.8 wheels do not update it.
Preflight runs a real bf16 CUDA matmul, requires native Muon, checks the GPU count,
and records versions/driver/GPU before downloading the model. A missing GPU,
incompatible driver, missing gated-model permission or missing dependency causes
a nonzero exit, never a silently substituted CPU/full-precision experiment.

## Reusable custom container template

### Preferred: use the checked GHCR image

PR #6 is merged. The checked image from source commit
`bfdf5d3580036b11876fe2264c6d92499913c681` was published successfully on
2026-10-18 at 21:53 UTC by [Actions run 39169212976](https://github.com/jasonjiang8866/peft-fine-tuning-recipes-classification/actions/runs/39169212976).
Copy this immutable reference into RunPod's **Container Image** field:

```text
ghcr.io/jasonjiang8866/peft-fine-tuning-recipes-classification@sha256:631a1ae62d4592324aaab894c8e0887c375830463443f150e15582269c1af71a
```

`pod-template.json` pins this same digest. The publication receipt is
[`runpod-ghcr-publication-39169212976-1` (artifact 12151654106)](https://github.com/jasonjiang8866/peft-fine-tuning-recipes-classification/actions/runs/39169212976/artifacts/12151654106).
The run verified Docker build, offline checks, publication and pull-by-digest
identity. **Real RunPod CUDA, 8-bit/bf16 training, GPU checkpoint resume and
Muon/AdamW performance remain unverified.**

This pin deliberately remains on that published source revision even when
`main` advances or a newer image is published. The image's bundled README/template
also remain at that revision; use this repository's updated deployment reference.
For a future release, update the pin only from its successful publication receipt.
The Docker workflow publishes to
`ghcr.io/jasonjiang8866/peft-fine-tuning-recipes-classification` as follows:

| Event | Behavior |
|---|---|
| Any `pull_request`, including same-repository and fork PRs | Read-only build/offline checks; no GHCR login or publication |
| Push to `main` in this repository | Build, offline checks, then publish the same local image |
| Manual `workflow_dispatch`, default `publish=false` | Read-only build/offline checks |
| Manual `workflow_dispatch` on `main`, `publish=true` | Explicitly request build/check/publication; requires Actions write access |
| Manual run on another branch, even with `publish=true`, or a fork | Read-only checks; publication is blocked |

A maintainer can use **Actions → RunPod Docker build and offline
checks → Run workflow**, select `main`, and enable `publish` to retry a failed
publication. There is no arbitrary source-ref input, tag/release publishing,
`pull_request_target` or `workflow_run` publishing path. Protect `main` with the
repository's normal review/branch rules: merging code there is the trust boundary.

The read-only job has `contents: read`. Only the gated publisher has
`contents: read` and `packages: write`, authenticating with the ephemeral
`GITHUB_TOKEN`; no publishing PAT secret is needed. Third-party actions in this
workflow are pinned to commit SHAs. Both jobs share the same build/check action,
and exactly one builds per run. The publisher logs in only after checks succeed,
tags the verified Docker image ID without rebuilding, pushes it, then pulls its
registry digest and checks that it resolves to the same image ID.

A successful publication produces:

- A source tag: `ghcr.io/jasonjiang8866/peft-fine-tuning-recipes-classification:sha-<full-40-character-commit-SHA>`.
- An immutable reference: `ghcr.io/jasonjiang8866/peft-fine-tuning-recipes-classification@sha256:<64-hex-registry-digest>`.
- The reference in the Actions run summary and the publisher job's
  `image-reference`/`digest` outputs; the
  `runpod-ghcr-publication-<run-id>-<attempt>` artifact contains `runpod-image.txt`,
  `image.json` (source SHA, image ID, tag, digest and run URL) and publication logs.
  Download the receipt for long-term retention; the artifact expires after 90 days.

When updating the pinned image, copy the **actual complete line from `runpod-image.txt`** into the RunPod custom
Pod template's **Container Image** field. The angle-bracket values above are
placeholders, not published images. Do not use an offline build's local image ID
as a registry digest. The workflow publishes no `latest` tag. SHA tags identify
source but are still mutable: rerunning a commit can replace its tag because the
base image, OS packages and transitive dependencies are not fully locked. Reuse
the **digest** for repeatable deployment/rollback; rebuilds are not promised to be
bit-for-bit identical. Preserve referenced package versions when cleaning GHCR.

### GHCR visibility and access

Check the package's own **Package settings** after its first successful push.
New container packages default to private; a public GitHub repository does not
guarantee an anonymously pullable container. This workflow does not change package
visibility.

- **Public image:** a package admin can deliberately make the package public.
  RunPod then pulls anonymously; no registry credential is needed. Verify a pull
  by digest using a Docker client without cached GHCR credentials.
- **Private image:** keep it private and configure RunPod **Registry credentials**
  for `ghcr.io`: GitHub username plus a classic PAT with only `read:packages` and
  access to this package (authorize SSO if required). Select that credential in
  the private Pod template. Registry authentication is needed **before** the
  container starts; setting a token as a Pod environment variable is insufficient.
  Never copy the workflow's short-lived `GITHUB_TOKEN` into RunPod. Keep registry
  credentials out of git, image build args, logs and shared templates.

If publication returns `permission_denied`/403, check repository/organization
Actions policy for the job's requested `packages: write`, package-creation policy,
and the package's Actions access. A pre-existing package created outside this
workflow may need to be connected to this repository and grant it **Write** under
**Manage Actions access**. The OCI source/revision labels and `GITHUB_TOKEN`
publication associate new images with the repository. Repository write access
alone does not prove an existing package permits publication. Fix access and
retry the explicit `main` dispatch; do not treat a failed push as a release. If a
push succeeds but digest confirmation fails, a registry version may exist, but
there is no verified deployment receipt: investigate and rerun before using it.

GitHub references: [Container registry authentication, visibility and digest pulls](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry),
[package access controls](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).

### Optional: build and push to your own registry

You can still build/check/push from your machine or an image-build service
(Docker is not required inside the GPU Pod):

```bash
docker build --platform linux/amd64 -f runpod/Dockerfile \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  -t YOUR_REGISTRY/peft-muon:cu128 .
bash runpod/check-image.sh YOUR_REGISTRY/peft-muon:cu128 docker-checks
docker push YOUR_REGISTRY/peft-muon:cu128
```

Use `pod-template.json` as the RunPod Console field reference. It is **not** an API
payload or an importable template ID. Create a custom **Pod** template with your
published image (prefer its immutable digest), no exposed ports, a 30 GB container
disk, and an 80 GB volume at `/workspace`. Set these environment variables:

| Setting | Value |
|---|---|
| `RUNPOD_PYTHON` | `/opt/venv/bin/python` (already set by the image; keep this value) |
| `RUN_SUITE` | `smoke` initially; `all` runs smoke then performance only on success |
| `HF_HOME` | `/workspace/hf-cache` |
| `RESULTS_ROOT` | `/workspace/results` |
| `CUDA_VISIBLE_DEVICES` | `0` |
| `HF_TOKEN` | Inject your Hugging Face read token privately at deployment |

Keep the image's entrypoint/CMD, or use the start command
`bash /opt/muon/runpod/container-start.sh`. Each container start creates a new
result directory under `/workspace/results/muon-<UTC timestamp>-<random suffix>/`,
with `smoke/` and/or `performance/` subdirectories. There is no SSH/Jupyter server in this batch image; view Pod logs
and retrieve artifacts from the mounted storage. `RUN_SUITE=performance` is also
available for subsequent repeats after a successful smoke run.

The code runs inside `/opt/muon` so mounting `/workspace` does not hide it. The
container includes its own `/opt/venv`, CUDA runtime wheel libraries, and the
source revision/hash in reports. The separate Docker workflow builds this image
and checks it offline, with the gated GHCR publication described above. Launching
a paid Pod and actual GPU validation still require the target environment; neither
is implied by build or publication success. The existing bootstrap path is unchanged.

**Stop the Pod after the run.** Script/container exit does not guarantee that GPU
billing stops. A Pod volume is not a backup: copy results out before terminating
the Pod, or use a RunPod network volume. The scripts do not hold an API key or
terminate your Pod automatically.

## What the profiles validate

| Profile | Work | Pass criteria |
|---|---|---|
| `smoke.json` | 12 optimizer steps, batch 2 × accumulation 2, length 128; checkpoint at 6; each arm rerun from checkpoint in a fresh process | Actual int8 base layers; all parameters on CUDA; bf16 observed inside forward; 128 2D LoRA tensors and one score head; finite gradients/loss/weights; LoRA and head updated; real AcceleratedOptimizer wrapper; scheduler LR groups agree; final trainables, nested optimizer state and scheduler state match uninterrupted run within configured tolerance |
| `performance.json` | 200 optimizer steps × seeds 42/43/44 × both optimizers; batch 8, length 512; omit first 20 steps from timing | Same CUDA/quantization/autocast/routing checks; finite loss and evaluation metrics; no checkpoint writes, per-step gradient probes, or resume work in measured runs |

The smoke profile also tests gradient accumulation. Trainer checkpoint save/load
includes the PEFT adapter **and classification head**, optimizer state, scheduler,
RNG and sampler progress. The resumed run keeps the **same total max_steps** as the
uninterrupted run; it does not create a different LR schedule. The comparison
checks all trainable tensors, both optimizers' momentum/counter tensors, scheduler
state, and the remaining LR trace. A mismatch fails the suite and writes
`resume-comparison.json`. CUDA kernels can be nondeterministic: investigate a
failure rather than raising tolerances until it passes. Exact bitwise equality is
not promised. A resumed `Trainer.train_loss` can have different denominator
semantics; use the full-run result/loss trace for optimizer comparisons.

This harness preserves the original model, AG News split seed, LoRA target
modules/rank/alpha/dropout, 8-bit frozen base, and AdamW classification head. Its
explicit differences from `training.py` are: single-device placement instead of
`device_map="auto"`, bf16 loading/autocast with observed dtype checks, SDPA,
`use_cache=False`, explicit non-checkpointed k-bit preparation, small fixed dataset
subsets, step-limited training, and final test-only evaluation. It does not pass
`bnb_8bit_compute_dtype`: that is not a supported way to select LLM.int8 compute
precision. bf16 autocast does **not** mean every bitsandbytes operation, adapter,
optimizer state or layer norm is bf16. These settings are identical across arms.

**Scope:** single process / single GPU only. Trainer constructs Accelerate
internally, so `accelerate launch` is unnecessary. Do not use DDP/FSDP/DeepSpeed,
CPU offload, FP16 GradScaler, or multiple visible GPUs with this harness.
OptimizerPair's nested state format and empty outer `.state` have not been
validated for those integrations.

## Measurements and reproducibility

- `environment.json`, `pip-freeze.txt`: GPU/driver/CUDA, package versions, source
  commit/dirty status/hash. No environment dump or HF token is recorded.
- `resolved-config.json`: model and dataset revisions resolved to immutable SHAs,
  data fingerprints and content SHA256. Data are prepared once and shared across
  all seeds/arms. Reuse the resolved SHAs in later profile files to repeat inputs.
- `adamw-42/result.json`, `muon-42/result.json`, etc.: per-step loss and LR groups,
  synchronized step times, mean/median/p95, effective examples/s, training-loop
  wall time, accuracy/weighted F1, trainable names/shapes and integration evidence.
- `peak_cuda`: separate load, train and evaluation peaks; train allocated/reserved
  figures are PyTorch allocator measurements, not total device usage.
- `gpu-telemetry.csv`: one-second `nvidia-smi` utilization, total device memory,
  temperature and power samples. Unsupported telemetry fields may show `N/A`.
- `summary.json`, `summary.csv`, `summary.md`: per-run results and mean/std across
  seeds for performance-eligible runs. Smoke runs are excluded from aggregates.
- Per-process `*.log`, `status.json`, and smoke `resume-comparison.json`: diagnostics
  and pass/fail status. A failed suite never produces a successful final summary.

Each arm starts in a fresh process with the same seed, sample order, split, batch
and schedule. Arm order alternates between seeds. Steps are optimizer updates,
including gradient accumulation; examples/s is derived from effective batch size
and synchronized step time. It excludes model/data loading, evaluation and work
outside the step callbacks. `train_wall_seconds` includes training-loop overhead.
The step timer deliberately synchronizes CUDA and logging occurs each step, so
these are instrumented benchmark measurements, not peak asynchronous throughput.
No token/s estimate is reported because dynamically padded token counts vary.

The performance preset evaluates a fixed 1024-example test subset; it is not a
full-dataset quality claim. Muon LR 0.02 is a starting point, not a tuned optimum.
Report quality and speed together; do not infer a winner from a 12-step smoke or
one seed. Adjust the JSON for larger samples/longer training, or test a separate
Muon LR grid without changing the baseline. `train_samples` must be divisible by
effective batch size, `eval_samples` by eval batch size, and smoke checkpoints must
be at half the total steps. Both arms must use the same settings.

If a run OOMs, lower batch size for **both** arms (and, if desired, raise gradient
accumulation); rerun in a fresh directory and keep the new config with the result.
No automatic batch reduction, fallback optimizer or skipped Muon arm is allowed.
For live progress, `tail -f /workspace/results/smoke-001/muon-42.log`.

## Local / CI verification boundary

```bash
python -S -m unittest discover -s tests -v
python -S runpod/validate.py --config runpod/configs/smoke.json --check-config
bash -n runpod/bootstrap.sh runpod/run.sh runpod/container-start.sh
# In an isolated CPU torch 2.9 + pinned transformers/accelerate/peft environment:
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python -m unittest discover -s tests -p test_trainer_pair_cpu.py -v
```

CPU integration uses a tiny randomly initialized Llama + real PEFT, Trainer,
Accelerate and native Muon. No pretrained model or dataset is downloaded. It
validates checkpoint/resume semantics, including both LR groups, **not** CUDA,
bitsandbytes kernels, bf16 execution, or performance.

### Independent Docker build / offline checks

`.github/workflows/docker.yml` builds the actual `runpod/Dockerfile` for
`linux/amd64` on each PR and pushes to `main`, separately from the CPU regression
workflow. PR builds check out the exact PR head and embed its commit ID. The job
on PRs does not publish; the trusted `main` publisher uses the same offline checks
before pushing the verified image. No job launches a RunPod Pod. Build logs,
image metadata and offline-check logs are uploaded even on failure (14-day retention).

After the build, `check-image.sh` runs containers with `--network none`, an empty
`/workspace` volume, no GPU and no HF token. It checks:

- Python 3.12 in `/opt/venv`, `RUNPOD_PYTHON`, the source revision, `pip check`,
  all direct requirements pins, torch 2.9.0 with CUDA 12.8 build metadata and Muon.
- Real dependency imports, including Trainer/PEFT/Accelerate and bitsandbytes;
  these are import checks, not GPU kernel execution.
- `validate.py --help`, `run.sh --help`, the default `--check-config`, and explicit
  smoke/performance `--check-config` calls from outside the source directory.
- The original tini ENTRYPOINT and default CMD: smoke/performance/all startup
  must fail at the explicit missing-CUDA preflight; an invalid suite must fail
  with exit 2. This is expected-failure coverage, not a passing training run.

Reproduce from the checked-out repository on a Docker host:

```bash
docker build --platform linux/amd64 -f runpod/Dockerfile \
  --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" -t peft-muon:ci .
bash runpod/check-image.sh peft-muon:ci docker-checks
```

Consult the Docker workflow result for the specific commit before claiming its
image is buildable. A green container job does **not** establish CUDA execution,
8-bit/bf16 training, GPU checkpoint resume or performance. Those remain unverified
until an actual GPU Pod produces passing smoke and performance results.

References: [RunPod templates](https://docs.runpod.io/pods/templates/overview),
[template fields and private registry credentials](https://docs.runpod.io/pods/templates/manage-templates),
[custom container templates](https://docs.runpod.io/pods/templates/create-custom-template),
[PyTorch 2.9 CUDA wheel commands](https://pytorch.org/get-started/previous-versions/),
[Transformers 4.57.1 bitsandbytes](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes).
