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

For the USD 5 campaign dispatch qwen-gpu.yml once on main with confirm_paid=true.
The controller requires the repository or organization Actions secret RUNPOD_API_KEY.
It searches secure capacity globally for one 4090, 3090, or 5090 in that order,
requires CUDA 12.8 or newer, and rejects prices above USD 2/hour. A 105-minute
allocation window leaves room for cleanup and a USD 1 allowance for extras.
Use a new dispatch only after reconciling previous charges against the total.
Runner interruption can prevent cleanup. Check inventory for the exact
qwen35-<Actions run ID> and its ownership markers before retrying. Keep the
worker.json, allocation.json and cleanup.json artifact together.
