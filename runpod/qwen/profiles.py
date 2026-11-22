"""Select the paid run's performance scope from the pinned benchmark recipe."""
import json
import math
from pathlib import Path


def profile(stage, scope):
    if scope not in {"all", "seed-42", "seeds-43-44"}:
        raise ValueError("Unknown Qwen experiment scope")
    cfg = json.loads(Path(__file__).with_name(stage + ".json").read_text())
    if stage == "performance" and scope == "seed-42":
        cfg["seeds"] = [42]
    if stage == "performance" and scope == "seeds-43-44":
        cfg["seeds"] = [43, 44]
        cfg["muon_first"] = True  # Preserve the order of the full three-seed matrix.
    return cfg


# Insertion order fixes the screen order and the last tie-break is lower LR.
SCREEN_RATES = {"screen-020": 0.02, "screen-006": 0.006, "screen-002": 0.002}
SWEEP = "lr-sweep"


def sweep_profile(stage, selected=None):
    if stage == "smoke":
        return dict(profile("smoke", "all"), eval_splits=["validation"])
    if stage == "confirmation" and selected in SCREEN_RATES.values():
        return dict(profile("performance", "seeds-43-44"), muon_lr=selected,
                    eval_splits=["validation", "test"])
    cfg = dict(profile("performance", "seed-42"), muon_lr=SCREEN_RATES[stage],
               eval_splits=["validation"])
    if stage != "screen-020":
        cfg["modes"] = ["muon"]
    return cfg


def select_rate(read):
    scores = []
    for stage, rate in SCREEN_RATES.items():
        result = read(f"{stage}/muon-42/result.json")
        f1 = result["validation"]["eval_f1"]
        loss = result["validation"]["eval_loss"]
        if (result["status"] != "passed" or result["config"]["muon_lr"] != rate
                or "test" in result or not math.isfinite(f1) or not math.isfinite(loss)):
            raise ValueError("Invalid learning-rate screen evidence")
        scores.append((-f1, loss, rate))
    return min(scores)[2]
