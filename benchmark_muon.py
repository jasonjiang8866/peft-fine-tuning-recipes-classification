"""Opt-in, separate-process AdamW vs built-in Muon benchmark for training.py.

No dependency or optimizer in the normal training path is changed. The private
environment variables below are set only by this runner for its child processes.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


PREFIX = "PEFT_MUON_BENCHMARK"


def trainable_parameters(model):
    """Audit the actual PEFT model, including modules_to_save (the score head)."""
    rows, muon_params, adamw_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_lora = ".lora_A." in name or ".lora_B." in name
        if is_lora and param.ndim != 2:
            raise ValueError(f"Muon requires a 2D LoRA parameter: {name} {tuple(param.shape)}")
        (muon_params if is_lora else adamw_params).append(param)
        rows.append({
            "name": name, "shape": list(param.shape), "ndim": param.ndim,
            "numel": param.numel(), "optimizer_in_muon_run": "muon" if is_lora else "adamw",
        })
    if not muon_params:
        raise ValueError("No trainable LoRA matrices found; refusing an empty Muon comparison")
    return rows, muon_params, adamw_params


def make_muon_and_adamw(muon_params, adamw_params, args, muon_lr):
    """Expose both optimizers as one to Trainer and its learning-rate scheduler."""
    import torch

    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError("torch.optim.Muon is unavailable; upgrade PyTorch for this arm")
    if not adamw_params:
        raise ValueError("Expected a trainable classification head for the AdamW arm")

    muon = muon_cls(muon_params, lr=muon_lr, weight_decay=args.weight_decay)
    adamw = torch.optim.AdamW(
        adamw_params, lr=args.learning_rate, weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_epsilon,
    )

    class OptimizerPair(torch.optim.Optimizer):
        def __init__(self):
            super().__init__([{"params": muon_params}, {"params": adamw_params}], {})
            self.muon, self.adamw = muon, adamw
            # Scheduler writes directly to these shared dictionaries.
            self.param_groups = muon.param_groups + adamw.param_groups

        def step(self, closure=None):
            loss = self.muon.step(closure=closure)
            self.adamw.step()
            return loss

        def zero_grad(self, set_to_none=True):
            self.muon.zero_grad(set_to_none=set_to_none)
            self.adamw.zero_grad(set_to_none=set_to_none)

        def state_dict(self):
            return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

        def load_state_dict(self, state_dict):
            self.muon.load_state_dict(state_dict["muon"])
            self.adamw.load_state_dict(state_dict["adamw"])
            self.param_groups = self.muon.param_groups + self.adamw.param_groups

    return OptimizerPair()


class BenchmarkRun:
    @classmethod
    def from_environment(cls):
        mode = os.environ[PREFIX]
        if mode not in ("adamw", "muon"):
            raise ValueError(f"Invalid {PREFIX}={mode!r}")
        return cls(
            mode=mode,
            output_dir=Path(os.environ[f"{PREFIX}_OUTPUT"]),
            max_steps=int(os.environ[f"{PREFIX}_STEPS"]),
            batch_size=int(os.environ[f"{PREFIX}_BATCH_SIZE"]),
            muon_lr=float(os.environ[f"{PREFIX}_LR"]),
            skip_steps=int(os.environ[f"{PREFIX}_SKIP_STEPS"]),
        )

    def __init__(self, mode, output_dir, max_steps, batch_size, muon_lr, skip_steps):
        self.mode, self.output_dir = mode, output_dir
        self.max_steps, self.batch_size = max_steps, batch_size
        self.muon_lr, self.skip_steps = muon_lr, skip_steps
        self.callback = None
        self.parameters = None

    def preflight(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("This 8-bit/bf16 training recipe requires a CUDA GPU")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("This benchmark is single-process only (device_map='auto')")
        if self.mode == "muon" and getattr(torch.optim, "Muon", None) is None:
            raise RuntimeError("torch.optim.Muon unavailable; install a PyTorch build exposing it")

    def before_model_load(self):
        from transformers import set_seed
        import torch

        set_seed(42)  # Includes the randomly initialized classification head.
        for device in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device)

    def configure_args(self, args):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        args.output_dir = str(self.output_dir / self.mode)
        args.per_device_train_batch_size = self.batch_size
        args.logging_steps = 1
        # Avoid checkpoint I/O inside step timing. Both arms test their final model.
        args.save_strategy = "no"
        args.load_best_model_at_end = False
        if self.max_steps:
            args.max_steps = self.max_steps

    def trainer_kwargs(self, model, args):
        from transformers import TrainerCallback
        import torch

        self.parameters, muon_params, adamw_params = trainable_parameters(model)
        counts = Counter(row["ndim"] for row in self.parameters)
        print(f"Trainable dimensions: {dict(sorted(counts.items()))}; "
              f"LoRA matrices: {len(muon_params)}; other: {len(adamw_params)}")
        for row in self.parameters:
            if row["optimizer_in_muon_run"] == "adamw":
                print(f"  AdamW: {row['name']} {row['shape']}")

        class StepTimer(TrainerCallback):
            def __init__(self):
                self.started = None
                self.times = []

            @staticmethod
            def synchronize():
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)

            def on_step_begin(self, args, state, control, **kwargs):
                self.synchronize()
                self.started = time.perf_counter()

            def on_step_end(self, args, state, control, **kwargs):
                self.synchronize()
                self.times.append({"step": state.global_step,
                                   "ms": (time.perf_counter() - self.started) * 1000})

        self.callback = StepTimer()
        result = {"callbacks": [self.callback]}
        if self.mode == "muon":
            result["optimizers"] = (
                make_muon_and_adamw(muon_params, adamw_params, args, self.muon_lr), None
            )
        # AdamW: omit optimizers entirely, preserving Trainer's default selection.
        return result

    def write_result(self, trainer, train_result, test_metrics):
        import torch
        import transformers
        import peft

        times = self.callback.times[self.skip_steps:]
        peak = []
        for device in range(torch.cuda.device_count()):
            peak.append({
                "device": torch.cuda.get_device_name(device),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
            })
        report = {
            "optimizer": self.mode, "model": "meta-llama/Llama-3.2-1B-Instruct",
            "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                         "transformers": transformers.__version__, "peft": peft.__version__},
            "settings": {"seed": 42, "batch_size": self.batch_size, "max_steps": self.max_steps,
                         "adamw_lr": trainer.args.learning_rate, "muon_lr": self.muon_lr,
                         "warmup_ratio": trainer.args.warmup_ratio,
                         "timing_warmup_steps_excluded": self.skip_steps},
            "trainable_parameters": self.parameters,
            "train": {"global_steps": trainer.state.global_step,
                      "train_loss": train_result.training_loss,
                      "loss_trace": [{"step": row["step"], "loss": row["loss"]}
                                     for row in trainer.state.log_history if "loss" in row],
                      "step_time_ms": self.callback.times,
                      "timed_steps": len(times),
                      "mean_step_ms_after_warmup": statistics.mean(row["ms"] for row in times)
                      if times else None},
            "test": {"loss": test_metrics.get("eval_loss"),
                     "accuracy": test_metrics.get("eval_accuracy"),
                     "f1": test_metrics.get("eval_f1")},
            "peak_cuda": peak,
        }
        path = self.output_dir / f"{self.mode}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Benchmark report: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_runs"))
    parser.add_argument("--max-steps", type=int, default=0,
                        help="0 runs the original full one-epoch recipe; positive values make a smoke run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--muon-lr", type=float, default=0.02,
                        help="Muon LoRA learning rate (classifier keeps the 2e-4 AdamW rate)")
    parser.add_argument("--skip-steps", type=int, default=5,
                        help="Exclude this many first steps from mean timing only")
    options = parser.parse_args()
    if options.max_steps < 0 or options.batch_size < 1 or options.muon_lr <= 0 or options.skip_steps < 0:
        parser.error("max-steps/skip-steps must be nonnegative; batch-size/muon-lr positive")

    import torch

    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for this 8-bit/bf16 recipe")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ["adamw", "muon"] if getattr(torch.optim, "Muon", None) else ["adamw"]
    if len(modes) == 1:
        print(f"torch {torch.__version__}: torch.optim.Muon unavailable; running AdamW only. "
              "Upgrade PyTorch to benchmark Muon.")
    summary = {"runs": {}, "skipped": {}}
    if "muon" not in modes:
        summary["skipped"]["muon"] = (
            f"torch {torch.__version__} does not expose torch.optim.Muon"
        )
    for mode in modes:
        env = os.environ.copy()
        env.update({
            PREFIX: mode, f"{PREFIX}_OUTPUT": str(options.output_dir.resolve()),
            f"{PREFIX}_STEPS": str(options.max_steps),
            f"{PREFIX}_BATCH_SIZE": str(options.batch_size),
            f"{PREFIX}_LR": str(options.muon_lr),
            f"{PREFIX}_SKIP_STEPS": str(options.skip_steps),
        })
        subprocess.run([sys.executable, str(Path(__file__).with_name("training.py"))],
                       env=env, check=True)
    for mode in modes:
        report = json.loads((options.output_dir / f"{mode}.json").read_text())
        summary["runs"][mode] = {
            "steps": report["train"]["global_steps"],
            "mean_step_ms_after_warmup": report["train"]["mean_step_ms_after_warmup"],
            "test": report["test"], "peak_cuda": report["peak_cuda"],
        }
        print(f"{mode}: steps={report['train']['global_steps']}, "
              f"mean_step_ms={report['train']['mean_step_ms_after_warmup']}, "
              f"accuracy={report['test']['accuracy']}, F1={report['test']['f1']}, "
              f"peak_allocated_MiB={[d['peak_allocated_mib'] for d in report['peak_cuda']]}")
    (options.output_dir / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
