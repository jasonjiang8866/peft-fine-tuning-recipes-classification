"""Select the paid run's performance scope from the pinned benchmark recipe."""
import json
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
