"""Inspect the first two optimizer updates following the smoke checkpoint."""
import hashlib
import json

import torch
from transformers import TrainerCallback


def tensors(items):
    digest = hashlib.sha256()
    for name, value in sorted(items, key=lambda pair: pair[0]):
        meta = [name]
        if value is not None:
            meta.extend([str(value.dtype), list(value.shape)])
        digest.update(json.dumps(meta).encode() + b"\0")
        if value is not None:
            raw = value.detach().reshape(-1).contiguous().view(torch.uint8).cpu()
            digest.update(raw.numpy())
    return digest.hexdigest()


class ResumeTrace(TrainerCallback):
    def __init__(self, boundary):
        self.boundary = boundary
        self.rows = []
        self.current = None

    @staticmethod
    def params(model):
        return [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def on_step_begin(self, args, state, control, model, optimizer, **kwargs):
        step = state.global_step + 1
        if not self.boundary < step <= self.boundary + 2:
            return
        params = self.params(model)
        raw = getattr(optimizer, "optimizer", optimizer)
        children = [raw.muon, raw.adamw] if hasattr(raw, "muon") else [raw]
        slots = [(name + "/" + key, value) for name, p in params for child in children
                 for key, value in child.state.get(p, {}).items() if isinstance(value, torch.Tensor)]
        self.current = {"step": step, "weights_before": tensors(params),
                        "optimizer_before": tensors(slots), "batches": []}
        self.rows.append(self.current)

    def before_forward(self, module, args, kwargs):
        if self.current is None:
            return
        batch = {k: kwargs[k] for k in ("input_ids", "attention_mask", "labels")}
        rng = [("cpu", torch.get_rng_state())]
        if batch["input_ids"].is_cuda:
            rng.append(("cuda", torch.cuda.get_rng_state(batch["input_ids"].device)))
        self.current["batches"].append({"input": tensors(batch.items()), "rng": tensors(rng)})

    def after_forward(self, module, args, result):
        if self.current is not None:
            self.current["batches"][-1]["logits"] = tensors([("logits", result.logits)])

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        if self.current is not None:
            self.current["clipped_gradients"] = tensors((n, p.grad) for n, p in self.params(model))

    def on_step_end(self, args, state, control, model, **kwargs):
        if self.current is not None:
            self.current["weights_after"] = tensors(self.params(model))
            self.current = None


def first_difference(left, right):
    if not left or len(left) != len(right):
        return {"field": "trace_length", "original": len(left), "resumed": len(right)}
    for a, b in zip(left, right):
        for field in ("step", "weights_before", "optimizer_before"):
            if a[field] != b[field]:
                return {"step": a["step"], "field": field}
        if len(a["batches"]) != len(b["batches"]):
            return {"step": a["step"], "field": "microbatch_count"}
        for i, (x, y) in enumerate(zip(a["batches"], b["batches"])):
            for field in ("input", "rng", "logits"):
                if x[field] != y[field]:
                    return {"step": a["step"], "microbatch": i + 1, "field": field}
        for field in ("clipped_gradients", "weights_after"):
            if a[field] != b[field]:
                return {"step": a["step"], "field": field}
    return None
