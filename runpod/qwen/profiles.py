"""Select the paid run's performance scope from the pinned benchmark recipe."""
import json
from pathlib import Path


def profile(stage, scope):
    if scope not in {"all", "seed-42"}:
        raise ValueError("Unknown Qwen experiment scope")
    cfg = json.loads(Path(__file__).with_name(stage + ".json").read_text())
    if stage == "performance" and scope == "seed-42":
        cfg["seeds"] = [42]
    return cfg
