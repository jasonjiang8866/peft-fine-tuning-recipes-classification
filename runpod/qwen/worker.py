"""Expose only this run's bounded benchmark evidence to its Actions controller."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from profiles import profile

root = Path("/workspace/qwen-results")
state = ("dependencies", "running")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/result" or not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + os.environ["RESULT_TOKEN"]):
            self.send_error(403)
            return
        body = collect(*state)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def collect(stage, status):
    files = {}
    # Read only public evidence. Avoid walking live checkpoint/tokenizer directories.
    paths = list(root.glob("*.log"))
    for suite in ("smoke", "performance"):
        directory = root / suite
        paths.extend(directory.glob("*.*"))
        for pattern in ("adamw-[0-9]*", "muon-[0-9]*"):
            for arm in directory.glob(pattern):
                if arm.is_dir():
                    paths.extend(arm.glob("*.json"))
    for p in paths:
        name = p.relative_to(root)
        if p.suffix in {".md", ".csv", ".json", ".txt", ".log"}:
            if p.suffix == ".log":
                with p.open("rb") as f:
                    f.seek(max(0, p.stat().st_size - 5000))
                    raw = f.read(5000)
            else:
                raw = p.read_bytes()
            content = raw.decode(errors="replace")
            files[str(name)] = {"text": content,
                                "sha256": hashlib.sha256(content.encode()).hexdigest()}
    return json.dumps(dict(status=status, stage=stage, commit=os.environ["MUON_SOURCE_COMMIT"],
                           pod=os.environ["RUNPOD_POD_ID"], files=files)).encode()


def train():
    global state
    stage = "dependencies"
    try:
        with (root / "setup.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "runpod/qwen/requirements.txt"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, "-m", "pip", "check"], stdout=log, stderr=subprocess.STDOUT, check=True)
        for stage in ("smoke", "performance"):
            cfg = root / (stage + "-config.json")
            cfg.write_text(json.dumps(profile(stage, os.environ.get("QWEN_SCOPE", "all"))))
            state = (stage, "running")
            with (root / (stage + ".log")).open("w") as log:
                subprocess.run(["bash", "runpod/run.sh", "--config", str(cfg),
                                "--output", str(root / stage)], stdout=log, stderr=subprocess.STDOUT, check=True)
        state = (stage, "passed")
    except Exception:
        import traceback
        (root / "error.log").write_text(traceback.format_exc())
        state = (stage, "failed")


if __name__ == "__main__":
    root.mkdir(parents=True, exist_ok=False)
    server = ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    threading.Thread(target=train, daemon=True).start()
    server.serve_forever()
