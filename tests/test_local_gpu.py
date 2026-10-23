"""Offline launcher/configuration coverage, not CUDA training evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_gpu.smoke import ROOT, launch, main
from runpod.validate import read_config, write_json


class LocalGpuTests(unittest.TestCase):
    def test_check_config_from_outside_repo_without_site_packages_or_writes(self):
        with tempfile.TemporaryDirectory(prefix="local gpu ") as tmp:
            root = Path(tmp)
            env = {**os.environ, "LOCAL_GPU_VENV": str(root / "venv with spaces"),
                   "HF_HOME": str(root / "cache"), "LOCAL_GPU_OUTPUT_ROOT": str(root / "results")}
            env.pop("LOCAL_GPU_PYTHON", None)
            run = subprocess.run([sys.executable, "-S", str(ROOT / "local_gpu/smoke.py"),
                                  "--check-config"], cwd=root, env=env,
                                 capture_output=True, text=True, check=True)
            report = json.loads(run.stdout)
            self.assertFalse(report["gpu_validation_passed"])
            self.assertEqual(report["recipe"], read_config(ROOT / "runpod/configs/smoke.json"))
            self.assertEqual(report["python"], str(root / "venv with spaces/bin/python"))
            self.assertEqual(report["hf_home"], str(root / "cache"))
            self.assertEqual(report["output_root"], str(root / "results"))
            self.assertEqual(list(root.iterdir()), [])

    def test_explicit_python_keeps_venv_symlink_and_overrides_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "python"
            python.symlink_to(sys.executable)
            run = subprocess.run([sys.executable, "-S", str(ROOT / "local_gpu/smoke.py"),
                                  "--check-config", "--python", str(python)],
                                 env={**os.environ, "LOCAL_GPU_PYTHON": "/invalid/python"},
                                 capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(run.stdout)["python"], str(python))

    def test_no_resume_config_is_rejected_before_launch(self):
        with patch("local_gpu.smoke.launch") as child:
            with self.assertRaises(SystemExit) as error:
                main(["--config", str(ROOT / "runpod/configs/performance.json")])
            self.assertEqual(error.exception.code, 2)
            child.assert_not_called()

    def test_empty_gpu_mask_does_not_silently_expose_gpu_zero(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch("local_gpu.smoke.launch") as child:
            with self.assertRaises(SystemExit) as error:
                main(["--check-config"])
            self.assertEqual(error.exception.code, 2)
            child.assert_not_called()

    def settings(self, root):
        return {"config": str(ROOT / "runpod/configs/smoke.json"),
                "recipe": read_config(ROOT / "runpod/configs/smoke.json"),
                "python": sys.executable, "venv": str(root / "venv"),
                "hf_home": str(root / "cache"), "cuda_visible_devices": "0",
                "output_root": str(root / "results"), "output": str(root / "result")}

    def test_delegates_to_existing_suite_and_preserves_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as tmp:
                settings = self.settings(Path(tmp))
                output = Path(settings["output"])
                env = {"HF_TOKEN": "test-secret-not-for-reports"}

                def child(command, **kwargs):
                    self.assertEqual(command[:2], ["bash", str(ROOT / "runpod/run.sh")])
                    self.assertEqual(command[3], str(output / "input-config.json"))
                    self.assertEqual(command[5], str(output / "suite"))
                    self.assertEqual(json.loads(Path(command[3]).read_text()), settings["recipe"])
                    self.assertEqual(kwargs["env"], env)
                    kwargs["stdout"].write("diagnostic from child\n")
                    write_json(output / "suite/status.json", {
                        "status": "failed" if fail else "passed", "error": "CUDA OOM" if fail else None})
                    return subprocess.CompletedProcess(command, 7 if fail else 0)

                with patch("local_gpu.smoke.subprocess.run", side_effect=child):
                    self.assertEqual(launch(settings, env), 7 if fail else 0)
                status = json.loads((output / "launcher-status.json").read_text())
                self.assertEqual(status["gpu_validation_passed"], not fail)
                self.assertGreaterEqual(status["elapsed_seconds"], 0)
                self.assertIn("diagnostic", (output / "console.log").read_text())
                self.assertNotIn("test-secret", (output / "launcher.json").read_text())
                if fail:
                    self.assertEqual(status["error"], "CUDA OOM")
                with self.assertRaises(FileExistsError):
                    launch(settings, env)

    def test_missing_python_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp))
            settings["python"] = str(Path(tmp) / "missing/python")
            with patch("local_gpu.smoke.subprocess.run") as child:
                self.assertNotEqual(launch(settings, {}), 0)
                child.assert_not_called()
            status = json.loads((Path(settings["output"]) / "launcher-status.json").read_text())
            self.assertFalse(status["gpu_validation_passed"])
            self.assertIn("missing/not executable", status["error"])

    def test_real_child_missing_dependencies_fails_and_records_diagnostics(self):
        with tempfile.TemporaryDirectory(prefix="local gpu ") as tmp:
            root = Path(tmp)
            python = root / "python-without-site"
            # Run the actual shared entry point in a dependency-free interpreter.
            import shlex
            python.write_text("#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' -S "$@"\n')
            python.chmod(0o755)
            output = root / "failure evidence"
            result = subprocess.run([
                sys.executable, "-S", str(ROOT / "local_gpu/smoke.py"),
                "--python", str(python), "--hf-home", str(root / "cache"),
                "--output", str(output)], cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            status = json.loads((output / "launcher-status.json").read_text())
            self.assertFalse(status["gpu_validation_passed"])
            self.assertIn("torch", status["error"])
            self.assertEqual(json.loads((output / "suite/status.json").read_text())["status"], "failed")
            self.assertIn("ModuleNotFoundError", (output / "console.log").read_text())
            self.assertFalse((output / "suite/summary.json").exists())

    def test_zero_exit_without_suite_evidence_is_failure_and_default_runs_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp))
            settings["output"] = None
            with patch("local_gpu.smoke.subprocess.run", return_value=subprocess.CompletedProcess([], 0)):
                self.assertEqual(launch(settings, {}), 1)
                self.assertEqual(launch(settings, {}), 1)
            runs = list(Path(settings["output_root"]).iterdir())
            self.assertEqual(len(runs), 2)
            for output in runs:
                self.assertFalse(json.loads((output / "launcher-status.json").read_text())["gpu_validation_passed"])

    def test_bootstrap_refuses_existing_environment_without_pip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Fake only the version check; any venv/pip invocation must fail.
            python = root / "base-python"
            python.write_text('#!/bin/sh\n[ "$1" = "-c" ] || exit 99\n')
            python.chmod(0o755)
            existing = root / "existing venv"
            existing.mkdir()
            result = subprocess.run(["bash", str(ROOT / "local_gpu/bootstrap.sh")],
                                    env={**os.environ, "LOCAL_GPU_BASE_PYTHON": str(python),
                                         "LOCAL_GPU_VENV": str(existing)}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Refusing existing", result.stderr)
            self.assertEqual(list(existing.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
