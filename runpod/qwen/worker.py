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

root = Path("/workspace/qwen-results")
root.mkdir(parents=True, exist_ok=False)
payload = b'{"status":"starting"}'


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/result" or not hmac.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + os.environ["RESULT_TOKEN"]):
            self.send_error(403)
            return
        body = payload
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def collect(stage, status):
    files = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        name = p.relative_to(root)
        if any(k in name.parts for k in ("data", "tokenizer", "checkpoints")):
            continue
        if p.suffix in {".md", ".csv", ".json", ".txt", ".log"}:
            if p.suffix == ".log":
                with p.open("rb") as f:
                    f.seek(max(0, p.stat().st_size - 5000))
                    raw = f.read()
            else:
                raw = p.read_bytes()
            files[str(name)] = {"text": raw.decode(errors="replace"),
                                "sha256": hashlib.sha256(raw).hexdigest()}
    return json.dumps(dict(status=status, stage=stage, commit=os.environ["MUON_SOURCE_COMMIT"],
                           pod=os.environ["RUNPOD_POD_ID"], files=files)).encode()


def train():
    global payload
    stage = "dependencies"
    try:
        with (root / "setup.log").open("w") as log:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "runpod/qwen/requirements.txt"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, "-m", "pip", "check"], stdout=log, stderr=subprocess.STDOUT, check=True)
        for stage in ("smoke", "performance"):
            payload = collect(stage, "running")
            with (root / (stage + ".log")).open("w") as log:
                subprocess.run(["bash", "runpod/run.sh", "--config", f"runpod/qwen/{stage}.json",
                                "--output", str(root / stage)], stdout=log, stderr=subprocess.STDOUT, check=True)
        payload = collect(stage, "passed")
    except Exception:
        import traceback
        (root / "error.log").write_text(traceback.format_exc())
        payload = collect(stage, "failed")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8888), Handler)
    threading.Thread(target=train, daemon=True).start()
    server.serve_forever()
