"""Local single-GPU smoke launcher; configuration checks require only stdlib."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from runpod.validate import read_config, write_json


def absolute(value):
    # Do not resolve executable symlinks: a venv's bin/python often points at
    # the base interpreter, but must be invoked through the venv path.
    return Path(value).expanduser().absolute()


def plan(args):
    config = absolute(args.config)
    recipe = read_config(config)
    if not recipe["checkpoint_step"]:
        raise ValueError("Local smoke requires checkpoint/resume validation; use a smoke config")
    venv = absolute(args.venv)
    python = absolute(args.python) if args.python else venv / "bin/python"
    env = os.environ.copy()
    env.update({
        "HF_HOME": str(absolute(args.hf_home)),
        "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0"),
        "RUNPOD_PYTHON": str(python),
        # Prevent an inherited PYTHONPATH/user site from changing the isolated stack.
        "PYTHONPATH": "", "PYTHONNOUSERSITE": "1",
    })
    env.pop("PYTHONHOME", None)
    # runpod/run.sh defaults an empty value to GPU 0; do not silently unmask a
    # device deliberately hidden by a workstation's inherited environment.
    if not env["CUDA_VISIBLE_DEVICES"].strip():
        raise ValueError("CUDA_VISIBLE_DEVICES is empty; explicitly select one GPU")
    return {
        "config": str(config), "recipe": recipe, "python": str(python),
        "venv": str(venv), "hf_home": env["HF_HOME"],
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "output_root": str(absolute(args.output_root)),
        "output": str(absolute(args.output)) if args.output else None,
    }, env


def validate_evidence(suite, recipe):
    """Check the shared suite's evidence contract without importing torch/loading tensors."""
    def require(condition, path, detail):
        if not condition:
            raise ValueError(f"Incomplete GPU evidence: {path}: {detail}")

    def nonempty(path):
        require(path.is_file() and path.stat().st_size > 0, path, "missing or empty file")

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value {value}")

    def read(path):
        nonempty(path)
        try:
            value = json.loads(path.read_text(), parse_constant=reject_constant)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"Incomplete GPU evidence: {path}: invalid JSON: {error}") from error
        require(isinstance(value, dict), path, "expected JSON object")
        return value

    def number(value):
        return type(value) in (int, float) and math.isfinite(value)

    def same_path(value, expected):
        return isinstance(value, str) and Path(value).is_absolute() and Path(value).resolve() == expected.resolve()

    config_path = suite / "resolved-config.json"
    resolved = read(config_path)
    # prepare_data resolves revisions and regenerates data metadata for this run.
    generated = {"model_revision", "dataset_revision", "data_root", "data_fingerprints", "data_sha256"}
    require(all(resolved.get(k) == v for k, v in recipe.items() if k not in generated),
            config_path, "does not match input smoke config")
    require(all(isinstance(resolved.get(k), str) and resolved[k]
                for k in ("model_revision", "dataset_revision")) and
            same_path(resolved.get("data_root"), suite), config_path, "missing resolved revisions/data root")
    summary_path = suite / "summary.json"
    summary = read(summary_path)
    for name in ("summary.md", "summary.csv"):
        nonempty(suite / name)
    records = summary.get("runs")
    require(summary.get("status") == "passed" and isinstance(records, list),
            summary_path, "missing passed summary/runs")
    expected = {(mode, seed) for mode in ("adamw", "muon") for seed in recipe["seeds"]}
    seen = set()
    for row in records:
        require(isinstance(row, dict) and isinstance(row.get("mode"), str) and
                type(row.get("seed")) is int, summary_path, "invalid run identity")
        key = (row["mode"], row["seed"])
        require(key in expected and key not in seen, summary_path, f"unexpected/duplicate run {key}")
        seen.add(key)
    require(seen == expected, summary_path, f"missing runs: {sorted(expected - seen)}")
    for row in records:
        mode, seed = row["mode"], row["seed"]
        original = suite / f"{mode}-{seed}"
        checkpoint = original / "checkpoints" / f"checkpoint-{recipe['checkpoint_step']}"
        require(same_path(row.get("path"), original) and row.get("performance_eligible") is False,
                summary_path, f"invalid smoke path/eligibility for {mode}-{seed}")
        state_path = checkpoint / "trainer_state.json"
        require(read(state_path).get("global_step") == recipe["checkpoint_step"],
                state_path, "wrong checkpoint step")
        for name in ("optimizer.pt", "scheduler.pt"):
            nonempty(checkpoint / name)
        for resumed in (False, True):
            run = suite / f"{mode}-{seed}{'-resumed' if resumed else ''}"
            path = run / "result.json"
            report = read(path)
            require(report.get("status") == "passed" and report.get("mode") == mode and
                    type(report.get("seed")) is int and report["seed"] == seed and
                    report.get("config") == resolved and
                    report.get("global_steps") == recipe["max_steps"] and
                    report.get("start_step") == (recipe["checkpoint_step"] if resumed else 0) and
                    "resumed_from" in report and
                    (same_path(report["resumed_from"], checkpoint) if resumed else report["resumed_from"] is None) and
                    report.get("performance_eligible") is False, path, "run identity/config/steps/resume mismatch")
            try:
                metrics = {"mean_step_ms": report["timing"]["mean_ms"],
                           "examples_per_second": report["timing"]["examples_per_second"],
                           "accuracy": report["test"]["eval_accuracy"], "f1": report["test"]["eval_f1"],
                           "train_peak_mib": report["peak_cuda"]["train"]["allocated_mib"]}
                values = [*metrics.values(), report["train_loss"], report["test"]["eval_loss"]]
            except (KeyError, TypeError) as error:
                raise ValueError(f"Incomplete GPU evidence: {path}: missing metrics") from error
            require(all(number(v) for v in values), path, "nonfinite/invalid metrics")
            if not resumed:
                require(all(number(row.get(k)) and row[k] == v for k, v in metrics.items()),
                        summary_path, f"metrics differ from {path}")
            for name in ("final-trainables.pt", "final-state.pt"):
                nonempty(run / name)
            if resumed:
                comparison_path = run / "resume-comparison.json"
                comparison = read(comparison_path)
                require(comparison.get("status") == "passed" and comparison.get("errors") == [] and
                        comparison.get("atol") == recipe["resume_atol"] and
                        comparison.get("rtol") == recipe["resume_rtol"] and
                        number(comparison.get("max_abs_tensor_difference")) and
                        comparison["max_abs_tensor_difference"] >= 0,
                        comparison_path, "missing successful comparison with configured tolerances")


def launch(settings, env):
    if settings["output"]:
        output = Path(settings["output"])
        output.mkdir(parents=True, exist_ok=False)
    else:
        parent = Path(settings["output_root"])
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path(tempfile.mkdtemp(prefix=f"smoke-{stamp}-", dir=parent))
    # Snapshot the validated config so later edits cannot change this run's input.
    write_json(output / "input-config.json", settings["recipe"])
    command = ["bash", str(ROOT / "runpod/run.sh"),
               "--config", str(output / "input-config.json"),
               "--output", str(output / "suite")]
    write_json(output / "launcher.json", {
        **settings, "output": str(output), "command": command,
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    status = {"status": "running", "gpu_validation_passed": False}
    write_json(output / "launcher-status.json", status)
    print(f"Results: {output}\nConsole log: {output / 'console.log'}", flush=True)
    start = time.perf_counter()
    code = 1
    try:
        if not os.access(settings["python"], os.X_OK):
            raise RuntimeError("GPU Python is missing/not executable; run local_gpu/bootstrap.sh "
                               "or set LOCAL_GPU_PYTHON/--python")
        Path(settings["hf_home"]).mkdir(parents=True, exist_ok=True)
        with (output / "console.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False)
        code = result.returncode
        suite_status = output / "suite/status.json"
        evidence = json.loads(suite_status.read_text()) if suite_status.exists() else {}
        if code or evidence.get("status") != "passed":
            raise RuntimeError(evidence.get("error") or
                               f"GPU suite did not pass (exit {code}); see console.log and suite/*.log")
        validate_evidence(output / "suite", settings["recipe"])
        status.update(status="passed", gpu_validation_passed=True)
    except (Exception, KeyboardInterrupt) as error:
        code = code or 1
        status.update(status="failed", error=str(error) or type(error).__name__)
        print(f"FAILED: {status['error']}", file=sys.stderr)
    finally:
        status.update(elapsed_seconds=time.perf_counter() - start, exit_code=code)
        write_json(output / "launcher-status.json", status)
    if status["gpu_validation_passed"]:
        print(f"GPU smoke passed: {output / 'suite/summary.md'}")
    return code if code >= 0 else 128 - code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "runpod/configs/smoke.json"))
    parser.add_argument("--venv", default=os.environ.get("LOCAL_GPU_VENV", str(ROOT / ".venv-local-gpu")))
    parser.add_argument("--python", default=os.environ.get("LOCAL_GPU_PYTHON"),
                        help="Explicit GPU Python executable path; overrides --venv")
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME", str(
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "huggingface")))
    parser.add_argument("--output-root", default=os.environ.get("LOCAL_GPU_OUTPUT_ROOT", str(ROOT / "local_gpu_runs")))
    parser.add_argument("--output", help="Exact new run directory (must not exist); overrides --output-root")
    parser.add_argument("--check-config", action="store_true",
                        help="CPU only: validate recipe and print paths; no imports/downloads/training or writes")
    args = parser.parse_args(argv)
    try:
        settings, env = plan(args)
        if args.check_config:
            print(json.dumps({"status": "configuration_valid", "gpu_validation_passed": False,
                              **settings}, indent=2))
            return 0
        return launch(settings, env)
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    sys.exit(main())
