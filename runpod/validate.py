"""Single-GPU Trainer validation. --help and config validation need only stdlib."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path):
    c = json.loads(Path(path).read_text())
    positive = ("max_steps", "train_samples", "eval_samples", "max_length", "batch_size",
                "eval_batch_size", "gradient_accumulation_steps", "expected_lora_matrices")
    for key in positive:
        if type(c.get(key)) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("skip_steps", "checkpoint_step"):
        if type(c.get(key)) is not int or not 0 <= c[key] < c["max_steps"]:
            raise ValueError(f"{key} must be in [0, max_steps)")
    for key in ("learning_rate", "muon_lr", "resume_atol", "resume_rtol"):
        if not isinstance(c.get(key), (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("warmup_ratio", "weight_decay"):
        if not isinstance(c.get(key), (int, float)) or not math.isfinite(c[key]) or not 0 <= c[key] <= 1:
            raise ValueError(f"{key} must be in [0, 1]")
    if not c.get("seeds") or any(type(s) is not int or s < 0 for s in c["seeds"]):
        raise ValueError("seeds must be a nonempty list of nonnegative integers")
    if len(set(c["seeds"])) != len(c["seeds"]):
        raise ValueError("seeds must be unique")
    effective = c["batch_size"] * c["gradient_accumulation_steps"]
    if c["train_samples"] < effective or c["train_samples"] % effective:
        raise ValueError("train_samples must be a multiple of effective batch size")
    if c["eval_samples"] % c["eval_batch_size"]:
        raise ValueError("eval_samples must be a multiple of eval_batch_size (drop_last=True)")
    if c["checkpoint_step"] and c["checkpoint_step"] * 2 != c["max_steps"]:
        raise ValueError("smoke checkpoint_step must be max_steps / 2 to retain the resume checkpoint")
    for key in ("model", "dataset", "model_revision", "dataset_revision"):
        if not isinstance(c.get(key), str) or not c[key]:
            raise ValueError(f"{key} must be a nonempty string")
    return c


def timing_summary(rows, skip_steps, effective_batch):
    values = [r["ms"] for r in rows if r["step"] > skip_steps]
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("No valid measured optimizer steps")
    ordered = sorted(values)
    return {"timed_steps": len(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "p95_ms": ordered[max(0, math.ceil(len(values) * 0.95) - 1)],
            "examples_per_second": effective_batch * 1000 / statistics.mean(values)}


def command_output(args):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def preflight():
    import importlib.metadata
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU (CUDA_VISIBLE_DEVICES=0)")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Single-process only; do not use torchrun/DDP/FSDP/DeepSpeed")
    if torch.cuda.get_device_capability()[0] < 8 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Native bf16 requires an Ampere-or-newer GPU for this template")
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("Native torch.optim.Muon required; run bootstrap.sh")
    # Exercise CUDA kernels before downloading gated model/data.
    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(x @ x).all().item():
        raise RuntimeError("bf16 CUDA matrix multiplication failed")
    if (ROOT / ".git").exists():
        revision = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        dirty = bool(command_output(["git", "-C", str(ROOT), "status", "--porcelain"]))
    else:
        revision = os.environ.get("MUON_SOURCE_COMMIT", "unknown")
        dirty = None  # Container source tree is hashed below; git metadata is not copied.
    source_hash = hashlib.sha256()
    for source in sorted([ROOT / "benchmark_muon.py", *Path(__file__).parent.rglob("*.py")]):
        source_hash.update(str(source.relative_to(ROOT)).encode())
        source_hash.update(source.read_bytes())
    return {"python": sys.version, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "accelerate", "peft", "bitsandbytes",
                          "datasets", "numpy", "scikit-learn", "huggingface-hub")},
            "nvidia_smi": command_output(["nvidia-smi"]),
            "git_commit": revision, "git_dirty": dirty, "source_sha256": source_hash.hexdigest()}


def prepare_data(c, output):
    from datasets import DatasetDict, load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    api = HfApi()
    c = dict(c)
    c["model_revision"] = api.model_info(c["model"], revision=c["model_revision"]).sha
    c["dataset_revision"] = api.dataset_info(c["dataset"], revision=c["dataset_revision"]).sha
    tok = AutoTokenizer.from_pretrained(c["model"], revision=c["model_revision"])
    tok.pad_token = tok.eos_token
    ds = load_dataset(c["dataset"], revision=c["dataset_revision"])
    # Match training.py split; subset selection is shared across ALL seeds and arms.
    split = ds["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="label")
    chosen = DatasetDict()
    for name, source, count in (("train", split["train"], c["train_samples"]),
                                ("test", ds["test"], c["eval_samples"])):
        if count > len(source):
            raise ValueError(f"{name}_samples exceeds available examples")
        chosen[name] = source.shuffle(seed=42).select(range(count)).map(
            lambda batch: tok(batch["text"], truncation=True, max_length=c["max_length"]),
            batched=True, remove_columns=["text"])
    chosen.save_to_disk(str(output / "data"))
    tok.save_pretrained(str(output / "tokenizer"))
    c["data_fingerprints"] = {name: ds._fingerprint for name, ds in chosen.items()}
    c["data_sha256"] = {
        name: hashlib.sha256(json.dumps(ds.to_dict(), sort_keys=True).encode()).hexdigest()
        for name, ds in chosen.items()}
    write_json(output / "resolved-config.json", c)
    return c


def worker(c, output, mode, seed, resume=None):
    import numpy as np
    import torch
    import bitsandbytes as bnb
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig, DataCollatorWithPadding, Trainer,
                              TrainerCallback, TrainingArguments, set_seed)
    from benchmark_muon import BenchmarkRun, trainable_parameters

    environment = preflight()
    output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    data_root = Path(c["data_root"])
    ds = load_from_disk(str(data_root / "data"))
    tok = AutoTokenizer.from_pretrained(str(data_root / "tokenizer"))
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForSequenceClassification.from_pretrained(
        c["model"], revision=c["model_revision"], num_labels=4,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map={"": 0}, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.1, bias="none", task_type="SEQ_CLS",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], modules_to_save=["score"]))
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    rows, lora, head = trainable_parameters(model)
    if len(lora) != c["expected_lora_matrices"] or len(head) != 1:
        raise RuntimeError("Unexpected LoRA/classifier parameter count; inspect recipe")
    if any(".score.modules_to_save." not in r["name"] for r in rows
           if r["optimizer_in_muon_run"] == "adamw"):
        raise RuntimeError("Unexpected non-LoRA trainable parameter")
    quantized = [m for m in model.modules() if isinstance(m, bnb.nn.Linear8bitLt)]
    if not quantized or any(m.weight.dtype != torch.int8 for m in quantized):
        raise RuntimeError("Expected actual int8 bitsandbytes base weights")
    if any(p.device.type != "cuda" for p in model.parameters()):
        raise RuntimeError("CPU/disk offload is not allowed for this comparison")
    smoke = c["checkpoint_step"] > 0
    before = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
              if p.requires_grad} if smoke else {}
    load_peak = torch.cuda.max_memory_allocated() / 2**20
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), max_steps=c["max_steps"],
        learning_rate=c["learning_rate"], weight_decay=c["weight_decay"],
        per_device_train_batch_size=c["batch_size"],
        per_device_eval_batch_size=c["eval_batch_size"],
        gradient_accumulation_steps=c["gradient_accumulation_steps"],
        bf16=True, fp16=False, tf32=False, gradient_checkpointing=False,
        warmup_ratio=c["warmup_ratio"], lr_scheduler_type="linear",
        eval_strategy="no", save_strategy="steps" if smoke and not resume else "no",
        save_steps=c["checkpoint_step"] or 1, save_total_limit=2,
        logging_steps=1, logging_nan_inf_filter=False, report_to="none",
        seed=seed, data_seed=seed, dataloader_num_workers=0, dataloader_drop_last=True,
        disable_tqdm=True)
    run = BenchmarkRun(mode, output, c["max_steps"], c["batch_size"], c["muon_lr"], c["skip_steps"])
    kwargs = run.trainer_kwargs(model, args)

    class Audit(TrainerCallback):
        def __init__(self):
            self.lrs, self.autocast, self.start_step = [], False, None
            self.grad_groups = {"lora": False, "head": False}

        def on_train_begin(self, args, state, control, optimizer=None, **kw):
            self.start_step = state.global_step
            if resume and self.start_step != c["checkpoint_step"]:
                raise RuntimeError("Trainer failed to restore global step")
            raw = getattr(optimizer, "optimizer", optimizer)
            if mode == "muon":
                children = raw.muon.param_groups + raw.adamw.param_groups
                if any(a is not b for a, b in zip(raw.param_groups, children)):
                    raise RuntimeError("OptimizerPair child group references are stale")
                if resume and (not raw.muon.state or not raw.adamw.state):
                    raise RuntimeError("Child optimizer state was not restored")
            torch.cuda.reset_peak_memory_stats()

        def on_pre_optimizer_step(self, args, state, control, **kw):
            if smoke:
                for label, params in (("lora", lora), ("head", head)):
                    for p in params:
                        if p.grad is not None:
                            if not torch.isfinite(p.grad).all().item():
                                raise RuntimeError(f"Nonfinite {label} gradients")
                            self.grad_groups[label] |= bool(torch.count_nonzero(p.grad).item())
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen base received gradients")

        def on_step_end(self, args, state, control, optimizer=None, lr_scheduler=None, **kw):
            lrs = [float(g["lr"]) for g in optimizer.param_groups]
            expected = lr_scheduler.get_last_lr()
            if lrs != list(expected) or any(not math.isfinite(x) for x in lrs):
                raise RuntimeError("Trainer scheduler/optimizer learning rates differ")
            if mode == "muon" and len(lrs) != 2:
                raise RuntimeError("Expected separate Muon and AdamW LR groups")
            if mode == "muon" and not math.isclose(
                    lrs[0] * c["learning_rate"], lrs[1] * c["muon_lr"], abs_tol=1e-12):
                raise RuntimeError("Muon/head LR ratio changed")
            self.lrs.append({"step": state.global_step, "lr": lrs})

    audit = Audit()
    # Probe inside the autocast-wrapped forward, not outside Trainer's context.
    def probe(module, inputs):
        audit.autocast |= (torch.is_autocast_enabled("cuda") and
                           torch.get_autocast_dtype("cuda") == torch.bfloat16)
    hook = quantized[0].register_forward_pre_hook(probe)
    kwargs["callbacks"].append(audit)
    def metrics(pred):
        labels = pred.label_ids
        predicted = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(labels, predicted),
                "f1": f1_score(labels, predicted, average="weighted", zero_division=0)}
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      processing_class=tok, data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=metrics, **kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    trained = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    train_peak = {"allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                  "reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    hook.remove()
    if trainer.state.global_step != c["max_steps"] or not audit.autocast:
        raise RuntimeError("Missing requested steps or bf16 autocast evidence")
    from accelerate.optimizer import AcceleratedOptimizer
    if not isinstance(trainer.optimizer, AcceleratedOptimizer) or trainer.accelerator.mixed_precision != "bf16":
        raise RuntimeError("Expected Accelerate optimizer wrapping and bf16 mixed precision")
    if any(not torch.isfinite(p).all().item() for p in lora + head):
        raise RuntimeError("Nonfinite final trainable weights")
    loss = [{"step": r["step"], "loss": r["loss"]} for r in trainer.state.log_history if "loss" in r]
    if not loss or any(not math.isfinite(r["loss"]) for r in loss):
        raise RuntimeError("Missing or nonfinite loss trace")
    if not math.isfinite(trained.training_loss):
        raise RuntimeError("Nonfinite final training loss")
    changed = {}
    if smoke:
        after = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        for label, names in (("lora", [n for n in after if ".lora_" in n]),
                             ("head", [n for n in after if ".score." in n])):
            changed[label] = any(not torch.equal(before[n], after[n]) for n in names)
        if not all(changed.values()) or not all(audit.grad_groups.values()):
            raise RuntimeError("Missing LoRA/head gradients or parameter updates")
        torch.save(after, output / "final-trainables.pt")
        torch.save({"optimizer": trainer.optimizer.state_dict(),
                    "scheduler": trainer.lr_scheduler.state_dict()}, output / "final-state.pt")
    torch.cuda.reset_peak_memory_stats()
    test = trainer.evaluate(ds["test"])
    if any(not math.isfinite(test[k]) for k in ("eval_loss", "eval_accuracy", "eval_f1")):
        raise RuntimeError("Nonfinite evaluation metrics")
    report = {
        "status": "passed", "mode": mode, "seed": seed, "config": c,
        "environment": environment, "global_steps": trainer.state.global_step,
        "start_step": audit.start_step, "resumed_from": str(resume) if resume else None,
        "integration": {"accelerator": type(trainer.accelerator).__name__,
                        "optimizer_wrapper": type(trainer.optimizer).__name__,
                        "optimizer_implementation": type(trainer.optimizer.optimizer).__name__,
                        "trainer_optim_setting": args.optim.value,
                        "mixed_precision": trainer.accelerator.mixed_precision,
                        "bf16_autocast_observed": audit.autocast,
                        "int8_linear_layers": len(quantized), "parameter_updates": changed,
                        "gradient_checks": audit.grad_groups if smoke else "not instrumented"},
        "trainable_parameters": rows, "train_loss": trained.training_loss,
        "loss_trace": loss, "lr_trace": audit.lrs, "step_time_ms": run.callback.times,
        "timing": timing_summary(run.callback.times, c["skip_steps"],
                                  c["batch_size"] * c["gradient_accumulation_steps"]),
        "train_wall_seconds": train_seconds, "test": test,
        "peak_cuda": {"load_allocated_mib": load_peak, "train": train_peak,
                      "eval_allocated_mib": torch.cuda.max_memory_allocated() / 2**20},
        "performance_eligible": not smoke and not resume}
    write_json(output / "result.json", report)


def compare_resume(original, resumed, c):
    import torch
    errors = []
    max_difference = 0.0
    def compare(a, b, path):
        nonlocal max_difference
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape:
                errors.append(path + ": tensor shape/type differs")
                return
            diff = (a.double() - b.double()).abs().max().item() if a.numel() else 0
            max_difference = max(max_difference, diff)
            if not torch.allclose(a, b, atol=c["resume_atol"], rtol=c["resume_rtol"]):
                errors.append(path + ": tensor differs")
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                errors.append(path + ": keys differ")
            else:
                for key in a:
                    compare(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)):
            if not isinstance(b, (list, tuple)) or len(a) != len(b):
                errors.append(path + ": sequence length differs")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, f"{path}.{i}")
        elif a != b:
            errors.append(path + ": scalar differs")
    for file in ("final-trainables.pt", "final-state.pt"):
        compare(torch.load(original / file, map_location="cpu", weights_only=True),
                torch.load(resumed / file, map_location="cpu", weights_only=True), file)
    a = json.loads((original / "result.json").read_text())
    b = json.loads((resumed / "result.json").read_text())
    compare([r for r in a["lr_trace"] if r["step"] > c["checkpoint_step"]], b["lr_trace"], "lr_trace")
    result = {"status": "failed" if errors else "passed", "errors": errors,
              "max_abs_tensor_difference": max_difference,
              "atol": c["resume_atol"], "rtol": c["resume_rtol"]}
    write_json(resumed / "resume-comparison.json", result)
    if errors:
        raise RuntimeError("Resume diverged; see resume-comparison.json (do not hide by loosening tolerances)")
    return result


def summarize(output, records):
    summary = {"status": "passed", "runs": records, "aggregates": {}}
    columns = ["mode", "seed", "mean_step_ms", "examples_per_second", "accuracy", "f1", "train_peak_mib"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row[k] for k in columns})
    for mode in ("adamw", "muon"):
        eligible = [r for r in records if r["mode"] == mode and r["performance_eligible"]]
        if eligible:
            summary["aggregates"][mode] = {
                key: {"mean": statistics.mean(r[key] for r in eligible),
                      "stdev": statistics.stdev(r[key] for r in eligible) if len(eligible) > 1 else None}
                for key in columns[2:]}
    write_json(output / "summary.json", summary)
    lines = ["# GPU validation results", "", "| Optimizer | Seed | ms/step | examples/s | accuracy | weighted F1 | train MiB |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        lines.append("| {mode} | {seed} | {mean_step_ms:.2f} | {examples_per_second:.2f} | {accuracy:.4f} | {f1:.4f} | {train_peak_mib:.1f} |".format(**r))
    lines += ["", "Smoke runs include gradient probes/checkpoint I/O and are NOT performance measurements.",
              "Step timing is synchronized optimizer-step compute (including gradient accumulation), not end-to-end data throughput.",
              "See train_wall_seconds for training-loop wall time. Evaluation uses a fixed test subset; this is not a tuned optimizer ranking."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def suite(config, output):
    c = read_config(config)
    output.mkdir(parents=True, exist_ok=False)  # Never mix with stale reports.
    write_json(output / "status.json", {"status": "running"})
    records = []
    telemetry = None
    telemetry_stream = None
    try:
        write_json(output / "environment.json", preflight())
        telemetry_stream = (output / "gpu-telemetry.csv").open("w")
        telemetry = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,name,utilization.gpu,memory.used,temperature.gpu,power.draw",
            "--format=csv", "--loop=1"], stdout=telemetry_stream, stderr=subprocess.STDOUT)
        (output / "pip-freeze.txt").write_text(command_output([sys.executable, "-m", "pip", "freeze"]) + "\n")
        c = prepare_data(c, output)
        c["data_root"] = str(output.resolve())
        write_json(output / "resolved-config.json", c)
        for index, seed in enumerate(c["seeds"]):
            # Alternate order to reduce systematic warm/cold GPU ordering bias.
            for mode in (("adamw", "muon") if index % 2 == 0 else ("muon", "adamw")):
                run_dir = output / f"{mode}-{seed}"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                           "--config", str(output / "resolved-config.json"), "--mode", mode,
                           "--seed", str(seed), "--output", str(run_dir)]
                with (output / f"{mode}-{seed}.log").open("w") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                report = json.loads((run_dir / "result.json").read_text())
                if c["checkpoint_step"]:
                    resumed = output / f"{mode}-{seed}-resumed"
                    resume = run_dir / "checkpoints" / f"checkpoint-{c['checkpoint_step']}"
                    resume_command = command[:-1] + [str(resumed), "--resume", str(resume)]
                    with (output / f"{mode}-{seed}-resumed.log").open("w") as log:
                        subprocess.run(resume_command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    compare_resume(run_dir, resumed, c)
                records.append({"mode": mode, "seed": seed, "path": str(run_dir),
                                "mean_step_ms": report["timing"]["mean_ms"],
                                "examples_per_second": report["timing"]["examples_per_second"],
                                "accuracy": report["test"]["eval_accuracy"],
                                "f1": report["test"]["eval_f1"],
                                "train_peak_mib": report["peak_cuda"]["train"]["allocated_mib"],
                                "performance_eligible": report["performance_eligible"]})
                print(f"PASS {mode} seed={seed}; results: {run_dir}", flush=True)
        summarize(output, records)
        write_json(output / "status.json", {"status": "passed"})
    except Exception as error:
        write_json(output / "status.json", {"status": "failed", "error": str(error), "completed_runs": records})
        raise
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:
                telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:
                telemetry.kill()
                telemetry.wait()
        if telemetry_stream is not None:
            telemetry_stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "runpod/configs/smoke.json")
    parser.add_argument("--output", type=Path, default=Path("/workspace/results/muon-smoke"))
    parser.add_argument("--check-config", action="store_true", help="Validate JSON without importing ML libraries")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("adamw", "muon"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    c = read_config(args.config)
    if args.check_config:
        print(json.dumps(c, indent=2))
    elif args.worker:
        if args.mode is None or args.seed is None:
            parser.error("worker requires --mode and --seed")
        worker(c, args.output.resolve(), args.mode, args.seed, args.resume)
    else:
        suite(args.config, args.output.resolve())


if __name__ == "__main__":
    main()
