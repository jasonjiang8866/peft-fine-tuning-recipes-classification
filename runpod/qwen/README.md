# Qwen3.5 paired optimizer experiments

## 0.8B baseline and learning-rate screen

The manual Qwen GPU workflow loads the text backbone of the pinned public
Qwen checkpoint and adds a four-class AG News head. The six full-attention
blocks receive rank-16 adapters on q/k/v/o projections. The 18 DeltaNet
blocks stay frozen. Each seed starts two independent processes on the same
Pod. The Muon arm uses native Muon for 48 adapter matrices and AdamW for the
classifier. The baseline uses Trainer's AdamW. Learning rates are fixed recipe
settings, so quality differences are specific to this configuration.

The 12-step smoke suite checks both step-6 restarts before performance testing.
Performance uses seeds 42/43/44, 200 steps each, batch size 8, maximum sequence
length 512, and omits the first 20 steps from timing. Weight and dataset revisions
are pinned in the profiles. The artifact contains per-run timing, peak memory,
loss, accuracy, weighted F1, trainable shapes, software versions and data hashes.

For the original fixed-rate comparison, dispatch qwen-gpu.yml on main with confirm_paid=true.
Select seed-42 for one 200-step pair, seeds-43-44 for the two remaining pairs,
or all for the three-seed matrix. A split run keeps the original arm order. Each
scope runs both smoke restart checks first. Enter the reconciled remaining
allowance for this allocation in remaining_usd (maximum USD 5).
The controller requires the repository or organization Actions secret RUNPOD_API_KEY.
It searches secure capacity globally for one 4090, 3090, or 5090 in that order,
requires CUDA 12.8 or newer, and rejects prices above USD 2/hour. The controller
reserves USD 1 for extras and five minutes for cleanup, then caps the workload
at the smaller of 105 minutes and the time affordable at the allocated rate.
Use a new dispatch only after reconciling previous charges against the total.
Runner interruption can prevent cleanup. Check inventory for the exact
qwen35-<Actions run ID> and its ownership markers before retrying. Keep the
worker.json, allocation.json and cleanup.json artifact together.

The first RTX 5090 allocation completed both 12-step AdamW trajectories but
failed the numerical restart check. The first subsequent logged loss difference
was at step 8. See [run 44303141495](https://github.com/jasonjiang8866/peft-fine-tuning-recipes-classification/actions/runs/44303141495).
For the Qwen restart test, use deterministic cuDNN convolution and math SDPA
and fingerprint steps 7 and 8, including both gradient accumulation microbatches.
The first differing digest in resume-comparison.json is diagnostic. The original
tensor tolerances still decide acceptance. The fresh performance processes use
normal kernel selection and collect no resume fingerprints.

[Run 44411232628](https://github.com/jasonjiang8866/peft-fine-tuning-recipes-classification/actions/runs/44411232628)
passed both restarts on an RTX 4090 with zero maximum tensor difference. After
105 minutes, its last snapshot showed performance had started. It contained
no completed performance reports. The four smoke evaluations took 124–127
seconds each. Those measurements cannot locate the delay inside performance.

The controller now saves a live snapshot on each successful poll, including
completed arms, GPU telemetry and the active arm's phase and step. Performance
progress updates every 20 steps with process CPU time, wall time, Torch thread
count and available cgroup v2 CPU quota/throttling counters. Compare deltas
between samples to investigate low utilization. A timed-out run may contain
individual results. Only a passed suite covers its selected scope.


## Validation-selected Muon learning rate

Dispatch `lr-sweep` on main for one secure RTX 4090. The single allocation
runs both smoke restarts, then screens AdamW and Muon at 0.02, 0.006 and 0.002
(Muon only for the latter two) using seed 42. Every full arm trains for 200
steps on the original 8,192 rows. Screening uses 1,024 stratified rows from
the held-out tenth of the AG News training split. The official test split is
only evaluated during confirmation.

The worker locks the Muon rate by maximum validation weighted F1, then minimum
validation loss, then lower learning rate. Fresh paired runs on seeds 43 and
44 evaluate both validation and the original 1,024 test examples. Those test
examples were also used by the published fixed-rate comparison. Compare the
two confirmation seeds separately from the exploratory seed. The controller
recomputes the choice from artifact metrics and checks identical train and
validation hashes across all full runs. Per-arm result files retain both sets
of confirmation metrics. Suite summary tables use validation for this scope.

Use a per-allocation allowance of USD 5 or less, including the existing USD 1
reserve. Reconcile actual usage before another dispatch. This sweep retains
the existing 105-minute ceiling and cleanup verification.

## 2B learning-rate transfer

The 2B scopes use the [pinned Qwen3.5-2B text configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json).
They carry over the Muon adapter rate of 0.002 selected on 0.8B. Head AdamW
and baseline AdamW both use 0.0002. The 24 text layers include six full-attention
layers with q/k/v/o targets, yielding 48 trainable LoRA matrices at rank 16.

| Manual scope | Work on one RTX 4090 |
| --- | --- |
| `2b-smoke` | Compare uninterrupted and step-6-restored runs for both optimizers over 12 steps. Use batch 8, a 512-token cap and one microbatch per step. |
| `2b-seed-42`, `2b-seed-43`, `2b-seed-44` | Run smoke, then one paired 200-step experiment. Seed 43 runs Muon first. |
| `2b-all` | Run smoke, then all three pairs in the same allocation. |

Use the smoke artifact to inspect peak allocated/reserved memory and observed
step times before choosing an allocation size. Smoke uses checkpoint probes,
deterministic kernels and a smaller data subset. Its timing and peak memory
are preliminary. A full seed pair measures the normal training path.
Each dispatch retains the existing per-allocation budget and cleanup checks.

The primary endpoint is paired validation weighted F1. Compare per-seed
differences, then their mean over seeds 42–44. Report test accuracy and
weighted F1 on the previously evaluated test subset as secondary endpoints.
Report step time over the final 180 steps and peak training memory.
Each full arm writes `predictions.json` with validation and test rows of
`[source_row, gold_class, predicted_class]`. Source rows are zero-based in
the pinned AG News train split for validation and test split for testing.
The controller checks each row against the split manifest and recomputes
accuracy and weighted F1. When combining separately dispatched seeds, check
that the manifests and dataset digests match across artifacts.
