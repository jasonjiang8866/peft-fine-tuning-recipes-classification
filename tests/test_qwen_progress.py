"""Regressions for live, partial evidence and the remaining campaign allowance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from runpod.validate import ROOT, write_json

with patch.object(sys, "path", [str(ROOT / "runpod/qwen"), *sys.path]):
    from runpod.qwen import controller, worker
    from profiles import profile


class QwenProgressTests(unittest.TestCase):
    def test_running_snapshot_survives_partial_utf8_and_captures_completed_arm(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(worker, "root", Path(temp)), patch.dict(
                os.environ, MUON_SOURCE_COMMIT="commit", RUNPOD_POD_ID="pod"):
            root = Path(temp)
            write_json(root / "performance/adamw-42/progress.json", {"step": 20})
            (root / "performance.log").write_bytes("€".encode() + b"x" * 4999)
            write_json(root / "performance/adamw-42/checkpoints/private.json", {})
            def snapshot():
                return json.loads(worker.collect("performance", "running"))
            first = snapshot()
            self.assertEqual(set(first["files"]), {"performance.log", "performance/adamw-42/progress.json"})
            record = first["files"]["performance.log"]
            self.assertIn("\ufffd", record["text"])
            self.assertEqual(hashlib.sha256(record["text"].encode()).hexdigest(), record["sha256"])
            write_json(root / "performance/adamw-42/result.json", {"global_steps": 200})
            second = snapshot()
            self.assertEqual(second["status"], "running")
            self.assertIn("performance/adamw-42/result.json", second["files"])

    def test_one_pair_is_not_a_three_seed_result(self):
        self.assertEqual(profile("smoke", "all"), profile("smoke", "seed-42"))
        cfg = profile("performance", "all")
        self.assertEqual(cfg["seeds"], [42, 43, 44])
        self.assertEqual(profile("performance", "seed-42"), dict(cfg, seeds=[42]))
        with self.assertRaises(ValueError):
            profile("performance", "invalid")

    def test_remaining_budget_reserves_cleanup_and_extras(self):
        for rate, expected in ((.89, 105), (1.39, 95), (2., 64)):
            minutes = controller.allocation_minutes(3.33, rate)
            self.assertEqual(minutes, expected)
            self.assertLessEqual(1 + (minutes + 5) * rate / 60, 3.33)
        for remaining, rate in ((1.01, 2), (5.01, .89), (float("nan"), 1), (3.33, 2.01)):
            with self.assertRaises(ValueError):
                controller.allocation_minutes(remaining, rate)
