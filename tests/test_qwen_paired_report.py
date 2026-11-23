"""CPU checks for combining the two confirmation batches."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

with patch.object(sys, "path", [str(Path(__file__).resolve().parents[1] / "runpod/qwen"), *sys.path]):
    import paired_report as paired


class PairedReportTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        # The controller checks full GPU evidence; these tests exercise the report's extra checks.
        verifier = patch.object(paired, "verify")
        self.verify = verifier.start()
        self.addCleanup(verifier.stop)

    def batch(self, scope, commit="a", hashes=None, missing_arm=False):
        files = {}

        def add(path, value):
            files["performance/" + path] = {"text": json.dumps(value)}

        add("resolved-config.json", {"data_sha256": paired.DATA_SHA256 if hashes is None else hashes})
        for split, digest in paired.MANIFEST_SHA256.items():
            files[f"performance/{split}-examples.json"] = {"sha256": digest}
        gpu = "NVIDIA GeForce RTX 4090"
        runs = []
        for seed in paired.TWO_B_SEEDS[scope]:
            for mode in (("muon", "adamw") if seed % 2 else ("adamw", "muon")):
                if missing_arm and seed == 46 and mode == "muon":
                    continue
                scores = {"eval_accuracy": .75, "eval_f1": .75 + (.01 if mode == "muon" else 0)}
                add(f"{mode}-{seed}/result.json", {
                    "environment": {"gpu": gpu, "git_commit": commit},
                    "validation": scores, "test": scores,
                    "timing": {"mean_ms": 101 if mode == "muon" else 100},
                    "peak_cuda": {"train": {"allocated_mib": 1024}},
                })
                runs.append({"seed": seed, "mode": mode})
        add("summary.json", {"runs": runs})
        path = self.root / (scope + ".zip")
        with ZipFile(path, "w") as archive:
            archive.writestr("allocation.json", json.dumps(dict(scope=scope, gpu=gpu, image=paired.IMAGE,
                                                                sha=commit, pod_id="pod")))
            archive.writestr("worker.json", json.dumps({"status": "passed", "files": files}))
            archive.writestr("cleanup.json", json.dumps({"absent": True}))
        return path

    def test_confirmation_in_reverse_upload_order(self):
        a, b = [self.batch(scope) for scope in paired.CONFIRMATION_SEEDS]
        result = paired.report("confirmation", [b, a])
        self.assertEqual(self.verify.call_count, 2)
        self.assertEqual([p["seed"] for p in result["pairs"]], list(range(45, 51)))
        self.assertEqual(result["muon_wins"], 6)
        for v in (result["validation_delta_pp"], result["muon_step_overhead_pct"],
                  *result["leave_one_seed_out_range_pp"]):
            self.assertAlmostEqual(v, 1.)

    def test_rejects_incomplete_duplicate_and_mismatched_batches(self):
        first, second = paired.CONFIRMATION_SEEDS
        a = self.batch(first)
        for paths, message in (([a], "Missing scopes"), ([a, a], "duplicate scope"),
                               ([a, self.batch(second, commit="b")], "different training commits")):
            with self.assertRaisesRegex(ValueError, message):
                paired.report("confirmation", paths)
        for options, message in (({"hashes": {}}, "dataset differs"), ({"missing_arm": True}, "arm")):
            with self.assertRaisesRegex(ValueError, message):
                paired.report("confirmation", [self.batch(first, **options)])
