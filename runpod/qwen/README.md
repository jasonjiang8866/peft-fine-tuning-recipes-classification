# Qwen3.5-0.8B paired optimizer experiment

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

For the USD 5 campaign, dispatch qwen-gpu.yml on main with confirm_paid=true.
Select seed-42 for one 200-step pair or all for the three-seed matrix. Each
scope runs both smoke restart checks first. Enter the reconciled remaining
campaign allowance in remaining_usd.
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
