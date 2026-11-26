"""Select the paid run's performance scope from the pinned benchmark recipe."""
import json
import math
from pathlib import Path


TWO_B = "Qwen/Qwen3.5-2B"
QWEN_TEXT_MODELS = {"Qwen/Qwen3.5-0.8B", TWO_B}
EQUAL_TIME = "2b-equal-time-180s"
CONFIRMATION_SEEDS = {"2b-seeds-45-47": [45, 46, 47], "2b-seeds-48-50": [48, 49, 50]}
TWO_B_SEEDS = {"2b-smoke": [42], "2b-seed-42": [42], "2b-seed-43": [43],
               "2b-seed-44": [44], "2b-all": [42, 43, 44], **CONFIRMATION_SEEDS,
               EQUAL_TIME: [51, 52, 53, 54]}


def stages_for(scope):
    if scope == "2b-smoke":
        return ("smoke",)
    if scope == SWEEP:
        return ("smoke", *SCREEN_RATES, "confirmation")
    return ("smoke", "performance")


def profile(stage, scope):
    if scope in TWO_B_SEEDS:
        cfg = profile(stage, "all")
        cfg.update(model=TWO_B, model_revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
                   muon_lr=0.002, eval_splits=["validation"])
        if stage == "smoke":
            cfg.update(batch_size=8, eval_batch_size=8, gradient_accumulation_steps=1, max_length=512)
        else:
            seeds = TWO_B_SEEDS[scope]
            cfg.update(seeds=seeds, muon_first=bool(seeds[0] % 2), eval_splits=["validation", "test"],
                       save_predictions=True)
            if scope == EQUAL_TIME:
                cfg.update(step_budget_seconds=180, skip_steps=0)
        return cfg
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
