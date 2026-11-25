"""Reproduce fixed-step and equal-time Qwen3.5-2B comparisons on CPU."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
from zipfile import BadZipFile, ZipFile

from controller import IMAGE, verify
from profiles import CONFIRMATION_SEEDS, EQUAL_TIME, TWO_B_SEEDS

# Identities from the accepted 2b-all artifact (Actions run 45059026463).
DATA_SHA256 = {
    "train": "323e31bde43a9cdb13c898cfb751808a65ac42a1c36a36ba192f4be8a9c09f4b",
    "validation": "64052b2edc54c989a562f622596d34322c48fbb732ee3925e609603bfeec82de",
    "test": "e6fd778078254e6df5ab9cd852a88a5d893f80f44da0b58c94ba638095310d9b",
}
MANIFEST_SHA256 = {
    "validation": "8b4d9d06c0397e321df76891642c2f51f05d161a38f555286e442e407470d25d",
    "test": "04dd9622ffff9ebf60543e0b75b820faeb3c2aa02fedb68cbdea6147c5f9537e",
}
MODES = ("adamw", "muon")


def read_batch(path, expected_scopes):
    with ZipFile(path) as archive:
        allocation = json.loads(archive.read("allocation.json"))
        worker = json.loads(archive.read("worker.json"))
        cleanup = json.loads(archive.read("cleanup.json"))
    scope = allocation["scope"]
    if scope not in expected_scopes:
        raise ValueError(f"{path}: unexpected or duplicate scope {scope}")
    if worker["status"] != "passed" or cleanup["absent"] is not True:
        raise ValueError(f"{path}: incomplete run or cleanup")
    if allocation["gpu"] != "NVIDIA GeForce RTX 4090" or allocation["image"] != IMAGE:
        raise ValueError(f"{path}: unexpected GPU or image")
    verify(worker, allocation["sha"], allocation["pod_id"], scope)
    files = worker["files"]

    def read(name):
        return json.loads(files["performance/" + name]["text"])

    if read("resolved-config.json")["data_sha256"] != DATA_SHA256:
        raise ValueError(f"{path}: dataset differs from the 2B baseline")
    for split, digest in MANIFEST_SHA256.items():
        if files[f"performance/{split}-examples.json"]["sha256"] != digest:
            raise ValueError(f"{path}: {split} row identities differ from the baseline")

    expected_order = [(seed, mode) for seed in TWO_B_SEEDS[scope]
                      for mode in (reversed(MODES) if seed % 2 else MODES)]
    actual_order = [(r["seed"], r["mode"]) for r in read("summary.json")["runs"]]
    if actual_order != expected_order:
        raise ValueError(f"{path}: missing, duplicate or out-of-order performance arm")

    pairs = []
    for seed in TWO_B_SEEDS[scope]:
        pair = {"seed": seed}
        for mode in MODES:
            row = read(f"{mode}-{seed}/result.json")
            env = row["environment"]
            if env["git_commit"] != allocation["sha"] or env["gpu"] != allocation["gpu"]:
                raise ValueError(f"{mode}-{seed}: training provenance mismatch")
            timing = row["timing"]["mean_ms"]
            peak = row["peak_cuda"]["train"]["allocated_mib"]
            if not all(math.isfinite(v) and v > 0 for v in (timing, peak)):
                raise ValueError(f"{mode}-{seed}: invalid timing or peak memory")
            pair[mode] = {
                "validation_f1": row["validation"]["eval_f1"],
                "test_f1": row["test"]["eval_f1"],
                "test_accuracy": row["test"]["eval_accuracy"],
                "step_ms": timing,
                "train_peak_gib": peak / 1024,
            }
            if scope == EQUAL_TIME:
                budget = row["step_budget"]
                pair[mode].update(global_steps=row["global_steps"], step_seconds=budget["elapsed_seconds"],
                                  train_wall_seconds=row["train_wall_seconds"])
                if budget["outcome"] != "reached":
                    pair.setdefault("below_target", []).append(mode)
        if "below_target" not in pair:
            pair["validation_delta_pp"] = 100 * (pair["muon"]["validation_f1"] - pair["adamw"]["validation_f1"])
        pairs.append(pair)
    return scope, allocation["sha"], pairs


def report(cohort, paths):
    scopes = {"baseline": {"2b-all"}, "confirmation": set(CONFIRMATION_SEEDS), "equal-time": {EQUAL_TIME}}
    remaining = scopes[cohort]
    commits = set()
    pairs = []
    for path in paths:
        scope, commit, batch = read_batch(path, remaining)
        remaining.remove(scope)
        commits.add(commit)
        pairs.extend(batch)
    if remaining:
        raise ValueError(f"Missing scopes: {sorted(remaining)}")
    if len(commits) != 1:
        raise ValueError("Batches used different training commits")
    pairs.sort(key=lambda pair: pair["seed"])
    result = {"cohort": cohort, "commit": commits.pop(), "pairs": pairs}
    if cohort == "equal-time":
        complete = all("below_target" not in pair for pair in pairs)
        result.update(target_seconds=180, comparison_status="complete" if complete else "incomplete")
        if not complete:
            return result
    deltas = [p["validation_delta_pp"] for p in pairs]
    leave_one_out = [mean(d for j, d in enumerate(deltas) if j != i) for i in range(len(deltas))]
    averages = {mode: {k: mean(p[mode][k] for p in pairs) for k in pairs[0][mode]} for mode in MODES}
    if cohort == "equal-time":
        result["quality_first_muon"] = mean(deltas) > 0 and min(leave_one_out) > 0
    return {
        **result, "means": averages,
        "validation_delta_pp": mean(deltas),
        "muon_wins": sum(d > 0 for d in deltas), "ties": deltas.count(0),
        "adamw_wins": sum(d < 0 for d in deltas),
        "leave_one_seed_out_range_pp": [min(leave_one_out), max(leave_one_out)],
        "muon_step_overhead_pct": 100 * (averages["muon"]["step_ms"] / averages["adamw"]["step_ms"] - 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=["baseline", "confirmation", "equal-time"])
    parser.add_argument("archives", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        print(json.dumps(report(args.cohort, args.archives), indent=2, allow_nan=False))
    except (OSError, BadZipFile, KeyError, ValueError) as error:
        parser.exit(1, f"Invalid benchmark evidence: {error}\n")
