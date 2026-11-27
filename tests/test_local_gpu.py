"""Offline launcher/configuration coverage, not CUDA training evidence."""
import json
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from local_gpu.smoke import ROOT, launch, main
from runpod.validate import read_config, summarize, write_json


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

    def suite_evidence(self, output, recipe):
        """Synthetic reports only; tensor placeholders are never loaded as GPU evidence."""
        suite = output / "suite"
        config = {**recipe, "model_revision": "resolved-model-sha",
                  "dataset_revision": "resolved-data-sha", "data_root": str(suite.resolve())}
        write_json(suite / "resolved-config.json", config)
        records = []
        for seed in recipe["seeds"]:
            for mode in ("adamw", "muon"):
                original = suite / f"{mode}-{seed}"
                checkpoint = original / "checkpoints" / f"checkpoint-{recipe['checkpoint_step']}"
                write_json(checkpoint / "trainer_state.json", {"global_step": recipe["checkpoint_step"]})
                for name in ("optimizer.pt", "scheduler.pt"):
                    (checkpoint / name).write_bytes(b"synthetic placeholder")
                for resumed in (False, True):
                    run = suite / f"{mode}-{seed}{'-resumed' if resumed else ''}"
                    report = {"status": "passed", "mode": mode, "seed": seed, "config": config,
                              "global_steps": recipe["max_steps"],
                              "start_step": recipe["checkpoint_step"] if resumed else 0,
                              "resumed_from": str(checkpoint) if resumed else None,
                              "performance_eligible": False, "train_loss": 0.9,
                              "timing": {"mean_ms": 10.0, "examples_per_second": 400.0},
                              "test": {"eval_loss": 0.8, "eval_accuracy": 0.5, "eval_f1": 0.4},
                              "peak_cuda": {"train": {"allocated_mib": 1000.0}}}
                    write_json(run / "result.json", report)
                    for name in ("final-trainables.pt", "final-state.pt"):
                        (run / name).write_bytes(b"synthetic placeholder")
                    if resumed:
                        write_json(run / "resume-comparison.json", {
                            "status": "passed", "errors": [], "max_abs_tensor_difference": 0.0,
                            "atol": recipe["resume_atol"], "rtol": recipe["resume_rtol"]})
                records.append({"mode": mode, "seed": seed, "path": str(original),
                                "mean_step_ms": 10.0, "examples_per_second": 400.0,
                                "accuracy": 0.5, "f1": 0.4, "train_peak_mib": 1000.0,
                                "performance_eligible": False})
        # Use the real stdlib summarizer to keep the fixture aligned with the producer.
        summarize(suite, records)
        write_json(suite / "status.json", {"status": "passed"})

    def test_passed_status_requires_complete_evidence(self):
        for missing in ("summary.json", "summary.md", "summary.csv", "adamw-42/result.json",
                        "muon-42/result.json", "muon-42-resumed/result.json",
                        "muon-42-resumed/resume-comparison.json",
                        "adamw-42/final-trainables.pt", "muon-42-resumed/final-state.pt",
                        "adamw-42/checkpoints/checkpoint-6/optimizer.pt",
                        "adamw-42/checkpoints/checkpoint-6/trainer_state.json"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                settings = self.settings(Path(tmp))
                output = Path(settings["output"])

                def child(command, **kwargs):
                    self.suite_evidence(output, settings["recipe"])
                    (output / "suite" / missing).unlink()
                    return subprocess.CompletedProcess(command, 0)

                with patch("local_gpu.smoke.subprocess.run", side_effect=child), \
                        patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    self.assertEqual(launch(settings, {}), 1)
                status = json.loads((output / "launcher-status.json").read_text())
                self.assertFalse(status["gpu_validation_passed"])
                self.assertEqual(status["status"], "failed")
                self.assertIn(missing, status["error"])
                self.assertNotIn("GPU smoke passed", stdout.getvalue())

    def test_passed_status_rejects_inconsistent_or_malformed_evidence(self):
        cases = [
            ("summary.json", lambda d: d.update(status="failed")),
            ("summary.json", lambda d: d["runs"].pop()),
            ("summary.json", lambda d: d["runs"].append(copy.deepcopy(d["runs"][0]))),
            ("summary.json", lambda d: d["runs"][0].update(seed=99)),
            ("summary.json", lambda d: d["runs"][0].update(path="/stale/adamw-42")),
            ("summary.json", lambda d: d["runs"][0].update(f1=0.99)),
            ("summary.json", lambda d: d["runs"][0].pop("accuracy")),
            ("summary.json", lambda d: d.update(runs={})),
            ("resolved-config.json", lambda d: d.update(max_steps=24)),
            ("muon-42/result.json", lambda d: d.update(status="failed")),
            ("muon-42/result.json", lambda d: d.update(mode="adamw")),
            ("muon-42/result.json", lambda d: d.update(seed=7)),
            ("muon-42/result.json", lambda d: d.update(global_steps=6)),
            ("muon-42/result.json", lambda d: d["config"].update(batch_size=8)),
            ("muon-42/result.json", lambda d: d.pop("timing")),
            ("muon-42/result.json", lambda d: d["test"].update(eval_f1=None)),
            ("muon-42-resumed/result.json", lambda d: d.update(start_step=0)),
            ("muon-42-resumed/result.json", lambda d: d.update(resumed_from=None)),
            ("muon-42-resumed/result.json", lambda d: d.update(performance_eligible=True)),
            ("muon-42-resumed/resume-comparison.json", lambda d: d.update(status="failed")),
            ("muon-42-resumed/resume-comparison.json", lambda d: d.update(errors=["tensor differs"])),
            ("muon-42-resumed/resume-comparison.json", lambda d: d.pop("errors")),
            ("muon-42-resumed/resume-comparison.json", lambda d: d.update(atol=1.0)),
            ("muon-42-resumed/resume-comparison.json", lambda d: d.update(rtol=1.0)),
            ("muon-42-resumed/resume-comparison.json", lambda d: d.pop("max_abs_tensor_difference")),
            ("muon-42-resumed/resume-comparison.json", lambda d: d.update(max_abs_tensor_difference=-1)),
            ("adamw-42/checkpoints/checkpoint-6/trainer_state.json", lambda d: d.update(global_step=0)),
        ]
        cases += [(path, raw) for path in ("summary.json", "muon-42/result.json",
                                          "muon-42-resumed/resume-comparison.json")
                  for raw in ("", "{", "[]", '{"status": "passed", "value": NaN}')]
        for index, (path, mutation) in enumerate(cases):
            with self.subTest(case=index, path=path), tempfile.TemporaryDirectory() as tmp:
                settings = self.settings(Path(tmp))
                output = Path(settings["output"])

                def child(command, **kwargs):
                    self.suite_evidence(output, settings["recipe"])
                    target = output / "suite" / path
                    if isinstance(mutation, str):
                        target.write_text(mutation)
                    else:
                        data = json.loads(target.read_text())
                        mutation(data)
                        write_json(target, data)
                    return subprocess.CompletedProcess(command, 0)

                with patch("local_gpu.smoke.subprocess.run", side_effect=child), \
                        patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    self.assertEqual(launch(settings, {}), 1)
                status = json.loads((output / "launcher-status.json").read_text())
                self.assertFalse(status["gpu_validation_passed"])
                self.assertIn(path, status["error"])
                self.assertNotIn("GPU smoke passed", stdout.getvalue())

    def test_custom_seeds_and_checkpoint_steps_require_every_arm_and_resume(self):
        for missing in (None, "muon-17/result.json", "adamw-9-resumed/resume-comparison.json"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as tmp:
                settings = self.settings(Path(tmp))
                settings["recipe"].update(seeds=[9, 17], max_steps=8, checkpoint_step=4)
                output = Path(settings["output"])

                def child(command, **kwargs):
                    self.suite_evidence(output, settings["recipe"])
                    if missing:
                        (output / "suite" / missing).unlink()
                    return subprocess.CompletedProcess(command, 0)

                with patch("local_gpu.smoke.subprocess.run", side_effect=child):
                    self.assertEqual(launch(settings, {}), 1 if missing else 0)
                status = json.loads((output / "launcher-status.json").read_text())
                self.assertEqual(status["gpu_validation_passed"], missing is None)

    def test_status_only_pass_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = self.settings(Path(tmp))
            output = Path(settings["output"])

            def child(command, **kwargs):
                write_json(output / "suite/status.json", {"status": "passed"})
                return subprocess.CompletedProcess(command, 0)

            with patch("local_gpu.smoke.subprocess.run", side_effect=child):
                self.assertEqual(launch(settings, {}), 1)
            self.assertFalse(json.loads((output / "launcher-status.json").read_text())["gpu_validation_passed"])

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
                    if not fail:
                        self.suite_evidence(output, settings["recipe"])
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
