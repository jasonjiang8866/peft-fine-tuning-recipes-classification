"""Local single-GPU smoke launcher; configuration checks require only stdlib."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
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
