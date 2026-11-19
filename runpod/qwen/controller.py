"""One manual allocation for the USD 5 Qwen optimizer comparison."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys
import time
from urllib.parse import quote

from profiles import profile
from provider import REPOSITORY, Runpod, budget, finite, identity, request

OUT = Path("qwen-evidence")
IMAGE = json.loads(Path("runpod/pod-template.json").read_text())["container_image"]
GPUS = ["NVIDIA GeForce RTX " + model for model in ("4090", "3090", "5090")]
RATE_CAP = 2.0
EXTRAS_USD = 1.0
CLEANUP_MINUTES = 5
OWNER = "qwen35-" + os.environ.get("GITHUB_RUN_ID", "")


def save(name, data):
    OUT.mkdir(exist_ok=True)
    p = OUT / name
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(p)


def select(api):
    for gpu in GPUS:
        catalog = api.call("/catalog/gpus/" + quote(gpu, safe="") +
                           "?include=AVAILABILITY&product=POD&cloud=SECURE&count=1&minCudaVersion=12.8")
        rate = catalog.get("price", {}).get("secure")
        regions = [r["id"] for r in catalog.get("dataCenters", [])
                   if r.get("availability") in {"LOW", "MEDIUM", "HIGH"}]
        if finite(rate, .001) and rate <= RATE_CAP and regions:
            return gpu, rate, regions
    raise RuntimeError("No approved card at or below $2/hour")


def cleanup(api):
    path = OUT / "allocation.json"
    if not path.exists():
        return
    state = json.loads(path.read_text())
    seen = False
    for check in range(12):
        pods = [p for p in api.listing("/pods", "pods") if p["name"] == OWNER]
        if not pods and (seen or state.get("pod_id") or check >= 8):
            save("cleanup.json", {"absent": True, "name": OWNER, "ended": time.time(),
                                  "estimated_gpu_usd": (time.time()-state["started"])*state["rate"]/3600})
            return
        for p in pods:
            if (len(pods) != 1 or p.get("env", {}).get("BENCH_OWNER") != REPOSITORY
                    or p["env"].get("BENCH_RUN") != OWNER or p.get("image") != IMAGE
                    or (state.get("pod_id") and p["id"] != state["pod_id"])
                    or not re.fullmatch("[A-Za-z0-9_-]+", p["id"])):
                raise RuntimeError("Pod ownership mismatch")
            if not seen:
                api.call("/pods/" + p["id"], "DELETE")
                seen = True
        time.sleep(5)
    raise RuntimeError("Pod absence unverified")


def allocation_minutes(remaining, rate):
    if not finite(remaining, .01) or remaining > 5 or not finite(rate, .001) or rate > RATE_CAP:
        raise ValueError("Invalid experiment allowance or hourly rate")
    minutes = min(105, math.floor((remaining - EXTRAS_USD) * 60 / rate) - CLEANUP_MINUTES)
    if minutes < 1:
        raise ValueError("Remaining allowance cannot fund allocation and cleanup")
    return minutes


def verify(result, sha, pod, scope="all"):
    if result["commit"] != sha or result["pod"] != pod:
        raise ValueError("Worker provenance mismatch")
    files = result["files"]
    for item in files.values():
        if hashlib.sha256(item["text"].encode()).hexdigest() != item["sha256"]:
            raise ValueError("Result checksum mismatch")
    def read(key):
        return json.loads(files[key]["text"])
    for stage in ("smoke", "performance"):
        cfg = profile(stage, scope)
        if read(f"{stage}/status.json")["status"] != "passed":
            raise ValueError("Incomplete suite")
        expected_hashes = read(f"{stage}/resolved-config.json")["data_sha256"]
        if not all(re.fullmatch("[0-9a-f]{64}", expected_hashes[key]) for key in ("train", "test")):
            raise ValueError("Missing dataset hash")
        for seed in cfg["seeds"]:
            for mode in ("adamw", "muon"):
                prefix = f"{stage}/{mode}-{seed}"
                row = read(prefix + "/result.json")
                if (row["mode"] != mode or row["seed"] != seed or row["status"] != "passed"
                        or any(cfg[k] != row["config"][k] for k in cfg)
                        or row["global_steps"] != cfg["max_steps"]
                        or row["config"]["data_sha256"] != expected_hashes
                        or row["performance_eligible"] != (stage == "performance")
                        or not row["integration"]["bf16_autocast_observed"]
                        or row["integration"]["int8_linear_layers"] < 1
                        or row["timing"]["timed_steps"] != cfg["max_steps"] - cfg["skip_steps"]):
                    raise ValueError("Incomplete training evidence: " + prefix)
                if stage == "smoke" and read(prefix + "-resumed/resume-comparison.json")["status"] != "passed":
                    raise ValueError("Resume comparison failed")


def run(api):
    if os.environ["GITHUB_RUN_ATTEMPT"] != "1":
        raise RuntimeError("Reconcile prior spend before retrying")
    sha = os.environ["GITHUB_SHA"]
    if not re.fullmatch("[0-9a-f]{40}", sha) or os.environ["GITHUB_REF"] != "refs/heads/main":
        raise ValueError("Expected reviewed main source")
    if any(p.get("env", {}).get("BENCH_OWNER") == REPOSITORY for p in api.listing("/pods","pods")):
        raise RuntimeError("An owned Pod already exists")
    scope = os.environ.get("QWEN_SCOPE", "all")
    profile("performance", scope)
    remaining = float(os.environ["QWEN_REMAINING_USD"])
    gpu, rate, regions = select(api)
    minutes = allocation_minutes(remaining, rate)
    save("budget.json", budget(api, reserve=remaining))
    state = dict(started=time.time(), gpu=gpu, rate=rate, regions=regions, image=IMAGE, sha=sha,
                 scope=scope, remaining_usd=remaining, allocation_minutes=minutes)
    save("allocation.json", state)
    token = secrets.token_urlsafe(32)
    command = (f"git init -q /tmp/qwen && cd /tmp/qwen && git fetch -q --depth 1 "
               f"https://github.com/{REPOSITORY}.git {sha} && git checkout -q FETCH_HEAD && "
               f"test \"$(git rev-parse HEAD)\" = {sha} && exec /opt/venv/bin/python runpod/qwen/worker.py")
    body = dict(name=OWNER, image=IMAGE, cloud="SECURE", disk=45, startSsh=False,
                ports=["8888/http"], gpu=dict(id=gpu,count=1,minCudaVersion="12.8"),
                dataCenterIds=regions, entrypoint=["/bin/bash","-lc"], cmd=[command],
                env=dict(BENCH_OWNER=REPOSITORY, BENCH_RUN=OWNER, RESULT_TOKEN=token,
                         RUNPOD_PYTHON="/opt/venv/bin/python", MUON_SOURCE_COMMIT=sha, QWEN_SCOPE=scope))
    pod = api.call("/pods", "POST", body)["id"]
    if not re.fullmatch("[A-Za-z0-9_-]+", pod):
        raise ValueError("Invalid Pod identity")
    state.update(pod_id=pod)
    save("allocation.json", state)
    detail = api.call("/pods/" + pod)
    if detail.get("gpu", {}).get("id") != gpu or not finite(detail.get("cost"), .001) or detail["cost"] > RATE_CAP:
        raise RuntimeError("Allocated GPU or rate outside approved bounds")
    state.update(rate=detail["cost"], region=detail["dataCenterId"])
    save("allocation.json", state)
    state["allocation_minutes"] = allocation_minutes(remaining, state["rate"])
    save("allocation.json", state)
    last_progress = None
    while time.time() < state["started"] + state["allocation_minutes"] * 60:
        try:
            result = request(f"https://{pod}-8888.proxy.runpod.net/result", token=token, limit=20_000_000)
        except RuntimeError:
            time.sleep(15)
            continue
        if "files" in result:
            save("worker.json", result)
            progress = [result["stage"]]
            for path, record in sorted(result["files"].items()):
                if re.fullmatch(r"performance/(adamw|muon)-[0-9]+/progress.json", path):
                    row = json.loads(record["text"])
                    progress.append((path.split("/")[1], row["phase"], row["step"]))
            if progress != last_progress:
                print("Qwen progress", progress, flush=True)
                last_progress = progress
        if result["status"] == "passed":
            verify(result, sha, pod, scope)
            return
        if result["status"] == "failed":
            raise RuntimeError("Worker failed; inspect worker.json")
        time.sleep(15)
    raise TimeoutError("Budgeted allocation window elapsed; inspect latest worker snapshot")


if __name__ == "__main__":
    identity()
    api = Runpod()
    if sys.argv[1] == "cleanup":
        cleanup(api)
    elif sys.argv[1] == "run":
        run(api)
