"""Reconstruct an equal-time endpoint from synchronized optimizer-step timings."""
import math


def step_budget_result(rows, steps, max_steps, seconds):
    """Accept only the first complete step at the target or the unfilled step limit."""
    if type(steps) is not int or not 1 <= steps <= max_steps or len(rows) != steps:
        raise ValueError("Incomplete equal-time step trace")
    elapsed_ms = 0.0
    target_ms = seconds * 1000
    for index, row in enumerate(rows, 1):
        ms = row["ms"]
        if (type(row["step"]) is not int or row["step"] != index
                or type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0):
            raise ValueError("Invalid equal-time step timing")
        if elapsed_ms >= target_ms:
            raise ValueError("Training continued after the time target")
        elapsed_ms += ms
    reached = elapsed_ms >= target_ms
    if not reached and steps != max_steps:
        raise ValueError("Training stopped before either limit")
    return {"target_seconds": seconds, "elapsed_seconds": elapsed_ms / 1000,
            "outcome": "reached" if reached else "step_limit"}


def verify_timed_arm(row, config):
    steps = row["global_steps"]
    cap = config["max_steps"]
    result = step_budget_result(row["step_time_ms"], steps, cap, config["step_budget_seconds"])
    wall = row["train_wall_seconds"]
    if (row["step_budget"] != result or not math.isfinite(wall) or wall < result["elapsed_seconds"]
            or not math.isclose(row["timing"]["mean_ms"], result["elapsed_seconds"] * 1000 / steps)):
        raise ValueError("Equal-time summary disagrees with measured times")

    # Trainer keeps the original linear warmup/decay schedule even when stopped early.
    warmup = math.ceil(cap * config["warmup_ratio"])
    trace = row["lr_trace"]
    if len(trace) != steps:
        raise ValueError("Incomplete equal-time learning-rate trace")
    for step, sample in enumerate(trace, 1):
        factor = step / warmup if step < warmup else (cap - step) / max(1, cap - warmup)
        rates = sample["lr"]
        bases = [config["learning_rate"]] * len(rates)
        if row["mode"] == "muon":
            bases = [config["muon_lr"], config["learning_rate"]]
        if (sample["step"] != step or not rates or len(rates) != len(bases)
                or any(not math.isclose(lr, base * factor, abs_tol=1e-12)
                       for lr, base in zip(rates, bases))):
            raise ValueError("Unexpected equal-time learning-rate schedule")
    return result["outcome"] == "reached"
