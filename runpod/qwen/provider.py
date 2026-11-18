"""Bounded RunPod API access. Account credentials never enter the Pod."""
import json
import math
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

API = "https://api.runpod.io/v2"
REPOSITORY = "jasonjiang8866/peft-fine-tuning-recipes-classification"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request(url, *, token, method="GET", body=None, limit=8_000_000):
    req = urllib.request.Request(url, method=method,
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json",
                 "User-Agent": "muon-bench-actions/0.1"},
        data=json.dumps(body).encode() if body is not None else None)
    try:
        with urllib.request.build_opener(NoRedirect).open(req, timeout=30) as response:
            data = response.read(limit + 1)
        if len(data) > limit:
            raise RuntimeError("Response exceeds limit")
        return json.loads(data) if data else None
    except urllib.error.HTTPError as exc:
        # Only fixed diagnostic categories leave this handler, never provider text.
        raw = exc.read(16000).decode(errors="replace").lower()
        categories = [label for word, label in (
            ("capacity", "capacity"), ("stock", "capacity"), ("disk", "disk"),
            ("cpu", "cpu"), ("vcpu", "vcpu"), ("image", "image"),
            ("global", "global-volume"), ("mount", "mount"), ("port", "port"),
            ("entrypoint", "entrypoint"), ("command", "command"), ("argument", "argument"),
            ("not supported", "unsupported"), ("minimum", "minimum"), ("maximum", "maximum"),
            ("unavailable", "unavailable"), ("not found", "not-found")) if word in raw]
        raise RuntimeError(f"Remote HTTP {exc.code}; categories={','.join(sorted(set(categories))) or 'unclassified'}") from None
    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError("Remote request timeout/unavailable; outcome may be unknown") from None


def finite(value, minimum=0):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value >= minimum


class Runpod:
    def __init__(self):
        self.key = os.environ.get("RUNPOD_API_KEY", "")
        if not self.key:
            raise RuntimeError("RUNPOD_API_KEY secret is missing")

    def call(self, path, method="GET", body=None):
        return request(API + path, token=self.key, method=method, body=body)

    def listing(self, path, key):
        items, seen = [], set()
        base = path
        while True:
            response = self.call(path)
            if not isinstance(response.get(key), list):
                raise RuntimeError("Invalid inventory")
            items.extend(response[key])
            page = response.get("pagination", {})
            if page.get("hasNextPage") is False:
                return items
            cursor = page.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise RuntimeError("Incomplete inventory")
            seen.add(cursor)
            path = base + "?" + urllib.parse.urlencode({"cursor": cursor})


def identity():
    if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
        raise RuntimeError("Unexpected Actions repository")
    run, attempt = os.environ.get("GITHUB_RUN_ID", ""), os.environ.get("GITHUB_RUN_ATTEMPT", "")
    if not run.isdigit() or not attempt.isdigit():
        raise RuntimeError("Invalid Actions identity")
    return run + "-" + attempt


def budget(api, reserve=1):
    if not finite(reserve, .01):
        raise ValueError("Invalid budget reserve")
    now = datetime.now(timezone.utc)
    result = request("https://api.runpod.io/graphql", token=api.key, method="POST", body={
        "query": "query SmokeBudget($start: DateTime!, $end: DateTime!) { myself { clientBalance currentSpendPerHr dailyCharges(startDate: $start, endDate: $end) { amount } } }",
        "variables": {"start": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat(), "end": now.isoformat()}})
    if result.get("errors"):
        raise RuntimeError("Budget query failed")
    account = result.get("data", {}).get("myself") or {}
    balance, spend, charges = account.get("clientBalance"), account.get("currentSpendPerHr"), account.get("dailyCharges")
    if not finite(balance) or not finite(spend) or not isinstance(charges, list):
        raise RuntimeError("Missing budget information")
    amounts = [c.get("amount") for c in charges if isinstance(c, dict)]
    if len(amounts) != len(charges) or not all(finite(x, -math.inf) for x in amounts):
        raise RuntimeError("Invalid billing history")
    if balance < reserve or sum(abs(x) for x in amounts) + spend + reserve > 500:
        raise RuntimeError("Insufficient experiment reserve or $500 monthly budget headroom")
    return {"reserve_usd": reserve, "monthly_budget_usd": 500, "balance_sufficient": True,
            "month_headroom_sufficient": True, "month_basis": "UTC", "checked_utc": now.isoformat(),
            "scope": "reported charges; billing may lag; storage/request fees additional"}
