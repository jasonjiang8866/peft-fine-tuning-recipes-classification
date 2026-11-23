"""Regressions for live, partial evidence and the remaining campaign allowance."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from runpod import validate
from runpod.validate import ROOT, write_json

with patch.object(sys, "path", [str(ROOT / "runpod/qwen"), *sys.path]):
    from runpod.qwen import controller, worker
    from profiles import SCREEN_RATES, profile, select_rate, sweep_profile


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
        self.assertEqual(profile("performance", "seeds-43-44"), dict(cfg, seeds=[43, 44], muon_first=True))
        with self.assertRaises(ValueError):
            profile("performance", "invalid")

    def test_sweep_locks_validation_winner_before_confirmation(self):
        rows = {}
        for stage, rate in SCREEN_RATES.items():
            cfg = sweep_profile(stage)
            self.assertEqual(cfg["eval_splits"], ["validation"])
            modes = cfg.get("modes", ["adamw", "muon"])
            self.assertEqual(modes, ["adamw", "muon"] if rate == .02 else ["muon"])
            self.assertEqual((cfg["seeds"], cfg["max_steps"]), ([42], 200))
            rows[f"{stage}/muon-42/result.json"] = {
                "status": "passed", "config": cfg, "validation": {"eval_f1": .8, "eval_loss": .5}}
        self.assertEqual(select_rate(rows.__getitem__), .002)
        medium = rows["screen-006/muon-42/result.json"]
        medium["validation"]["eval_loss"] = .4
        self.assertEqual(select_rate(rows.__getitem__), .006)
        control = rows["screen-020/muon-42/result.json"]
        control["validation"]["eval_f1"] = .81
        chosen = select_rate(rows.__getitem__)
        self.assertEqual(chosen, .02)
        confirm = sweep_profile("confirmation", chosen)
        self.assertEqual((confirm["seeds"], confirm["eval_splits"], confirm["muon_lr"]),
                         ([43, 44], ["validation", "test"], chosen))
        control["test"] = {}
        with self.assertRaises(ValueError):
            select_rate(rows.__getitem__)

    def test_2b_model_pin_scope_order_and_fixed_recipe(self):
        from profiles import TWO_B, stages_for
        self.assertEqual(stages_for("2b-smoke"), ("smoke",))
        full = profile("performance", "2b-all")
        smoke = profile("smoke", "2b-smoke")
        self.assertEqual(full["model"], TWO_B)
        self.assertEqual(full["model_revision"], "15852e8c16360a2fea060d615a32b45270f8a8fc")
        self.assertEqual(full["seeds"], [42, 43, 44])
        self.assertEqual((smoke["max_steps"], smoke["checkpoint_step"], full["max_steps"]), (12, 6, 200))
        for cfg in (smoke, full):
            self.assertEqual((cfg["batch_size"], cfg["max_length"], cfg["gradient_accumulation_steps"]),
                             (8, 512, 1))
            self.assertEqual((cfg["muon_lr"], cfg["learning_rate"], cfg["expected_lora_matrices"]),
                             (.002, .0002, 48))
        self.assertEqual(full["eval_splits"], ["validation", "test"])
        self.assertTrue(full["save_predictions"])
        for seed in full["seeds"]:
            scope = f"2b-seed-{seed}"
            self.assertEqual(stages_for(scope), ("smoke", "performance"))
            self.assertEqual(profile("performance", scope), dict(full, seeds=[seed], muon_first=seed == 43))
        for scope, seeds in (("2b-seeds-45-47", [45, 46, 47]), ("2b-seeds-48-50", [48, 49, 50])):
            self.assertEqual(profile("performance", scope), dict(full, seeds=seeds, muon_first=seeds[0] == 45))
            self.assertEqual(profile("smoke", scope), smoke)
        self.assertNotIn("save_predictions", profile("performance", "all"))
        self.assertEqual(profile("performance", "all")["muon_lr"], .02)

    def test_paired_prediction_evidence_checks_identity_and_metrics(self):
        examples = [[7, 0], [5, 0], [8, 1], [2, 2]]
        predictions = [[7, 0, 0], [5, 0, 1], [8, 1, 1], [2, 2, 2]]
        metrics = {"eval_accuracy": .75, "eval_f1": .75}
        controller.verify_predictions(examples, predictions, metrics, 4)
        for bad in (predictions[::-1], predictions[:-1], [*predictions[:-1], [2, 2, 4]],
                    [*predictions[:-1], [2, 2, 1]]):
            with self.assertRaises(ValueError):
                controller.verify_predictions(examples, bad, metrics, 4)

    def test_split_suite_keeps_the_original_arm_order(self):
        def order(scope):
            seen = []
            def finish_arm(cmd, **kwargs):
                mode = cmd[cmd.index("--mode") + 1]
                seed = int(cmd[cmd.index("--seed") + 1])
                seen.append((seed, mode))
                write_json(Path(cmd[-1]) / "result.json",
                           {"timing": {"mean_ms": 1, "examples_per_second": 1},
                            "test": {"eval_accuracy": 1, "eval_f1": 1},
                            "validation": {"eval_accuracy": 1, "eval_f1": 1},
                            "peak_cuda": {"train": {"allocated_mib": 1}}, "performance_eligible": True})
            with tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / "config.json"
                write_json(config, profile("performance", scope))
                with patch.object(validate, "preflight", return_value={}), patch.object(
                        validate, "command_output", return_value=""), patch.object(
                        validate, "prepare_data", side_effect=lambda cfg, out: cfg), patch.object(
                        validate.subprocess, "Popen"), patch.object(validate.subprocess, "run", side_effect=finish_arm):
                    validate.suite(config, Path(tmp) / "suite")
            return seen

        full = [(42, "adamw"), (42, "muon"), (43, "muon"), (43, "adamw"), (44, "adamw"), (44, "muon")]
        self.assertEqual(order("all"), full)
        self.assertEqual(order("seed-42") + order("seeds-43-44"), full)
        self.assertEqual(order("2b-seeds-45-47") + order("2b-seeds-48-50"),
                         [(s, mode) for s in range(45, 51)
                          for mode in (("muon", "adamw") if s % 2 else ("adamw", "muon"))])

    def test_remaining_budget_reserves_cleanup_and_extras(self):
        for rate, expected in ((.89, 105), (1.39, 95), (2., 64)):
            minutes = controller.allocation_minutes(3.33, rate)
            self.assertEqual(minutes, expected)
            self.assertLessEqual(1 + (minutes + 5) * rate / 60, 3.33)
        for remaining, rate in ((1.01, 2), (5.01, .89), (float("nan"), 1), (3.33, 2.01)):
            with self.assertRaises(ValueError):
                controller.allocation_minutes(remaining, rate)
