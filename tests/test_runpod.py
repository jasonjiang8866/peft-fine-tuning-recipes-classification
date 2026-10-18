"""Standard-library validation for RunPod configuration, timing and reporting."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from runpod.validate import ROOT, read_config, suite, summarize, timing_summary, write_json


class RunPodTests(unittest.TestCase):
    def test_configs_and_help_without_dependencies(self):
        for name in ("smoke", "performance"):
            c = read_config(ROOT / f"runpod/configs/{name}.json")
            self.assertEqual(c["expected_lora_matrices"], 128)
            result = subprocess.run([sys.executable, "-S", str(ROOT / "runpod/validate.py"),
                                     "--config", str(ROOT / f"runpod/configs/{name}.json"),
                                     "--check-config"], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), c)
        subprocess.run([sys.executable, "-S", str(ROOT / "runpod/validate.py"), "--help"],
                       capture_output=True, check=True)

    def test_invalid_config_rejected(self):
        valid = read_config(ROOT / "runpod/configs/smoke.json")
        invalid = {"batch_size": 0, "skip_steps": 12, "checkpoint_step": 12,
                   "max_steps": -1, "muon_lr": float("nan"), "seeds": [42, 42],
                   "train_samples": 255, "eval_samples": 127, "max_length": True}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for key, value in invalid.items():
                with self.subTest(key=key):
                    c = copy.deepcopy(valid)
                    c[key] = value
                    path.write_text(json.dumps(c))
                    with self.assertRaises(ValueError):
                        read_config(path)

    def test_timing_uses_global_step_and_effective_batch(self):
        result = timing_summary([{"step": 1, "ms": 10000}, {"step": 2, "ms": 10},
                                 {"step": 3, "ms": 30}], 1, 4)
        self.assertEqual(result, {"timed_steps": 2, "mean_ms": 20, "median_ms": 20,
                                  "p95_ms": 30, "examples_per_second": 200})
        for rows in ([], [{"step": 1, "ms": float("inf")}], [{"step": 1, "ms": 0}]):
            with self.assertRaises(ValueError):
                timing_summary(rows, 0, 4)

    def test_smoke_never_enters_performance_aggregates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            records = [{"mode": mode, "seed": 42, "mean_step_ms": 10,
                        "examples_per_second": 100, "accuracy": 0.5, "f1": 0.4,
                        "train_peak_mib": 1024, "performance_eligible": False}
                       for mode in ("adamw", "muon")]
            summarize(output, records)
            self.assertEqual(json.loads((output / "summary.json").read_text())["aggregates"], {})
            records[0]["performance_eligible"] = True
            summarize(output, records)
            self.assertEqual(set(json.loads((output / "summary.json").read_text())["aggregates"]), {"adamw"})
            self.assertTrue((output / "summary.csv").is_file())
            self.assertIn("NOT performance", (output / "summary.md").read_text())

    def test_json_refuses_nonfinite_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            with self.assertRaises(ValueError):
                write_json(path, {"loss": float("nan")})
            self.assertFalse(path.exists())

    def test_preflight_failure_is_recorded_and_stale_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "failed"
            with patch("runpod.validate.preflight", side_effect=RuntimeError("no CUDA")):
                with self.assertRaisesRegex(RuntimeError, "no CUDA"):
                    suite(ROOT / "runpod/configs/smoke.json", output)
            self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")
            self.assertFalse((output / "summary.json").exists())
            with self.assertRaises(FileExistsError):
                suite(ROOT / "runpod/configs/smoke.json", output)


if __name__ == "__main__":
    unittest.main()
