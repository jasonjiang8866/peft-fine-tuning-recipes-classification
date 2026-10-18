"""Publish the already checked local image; never build or accept a source ref here."""
import json
import os
from pathlib import Path
import re
import subprocess


REPOSITORY = "jasonjiang8866/peft-fine-tuning-recipes-classification"
IMAGE = "ghcr.io/" + REPOSITORY
SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def docker(*args, log):
    result = subprocess.run(
        ["docker", *args], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    with log.open("a") as stream:
        stream.write(result.stdout)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    return result.stdout


def publish(environ, checks=Path("docker-checks"), output=Path("ghcr-publication")):
    # Defense in depth; the Actions job-level gate is the permission boundary.
    event = environ.get("GITHUB_EVENT_NAME")
    require(
        environ.get("GITHUB_REPOSITORY") == REPOSITORY
        and environ.get("GITHUB_REF") == "refs/heads/main"
        and (event == "push" or (
            event == "workflow_dispatch" and environ.get("PUBLISH_REQUESTED") == "true"
        )),
        "GHCR publication requires this repository's main push or explicit main dispatch",
    )
    source = environ.get("GITHUB_SHA", "")
    require(re.fullmatch(r"[0-9a-f]{40}", source), "Expected a full Git commit SHA")
    require((checks / "source-commit.txt").read_text().strip() == source,
            "Checked source does not match this workflow commit")
    image_id = (checks / "verified-image-id.txt").read_text().strip()
    require(SHA256.fullmatch(image_id), "Missing verified Docker image ID")
    output.mkdir(parents=True, exist_ok=True)
    log = output / "publication.log"
    metadata = json.loads(docker("image", "inspect", image_id, log=log))[0]
    labels = metadata["Config"].get("Labels") or {}
    require(metadata["Id"] == image_id and metadata["Os"] == "linux"
            and metadata["Architecture"] == "amd64", "Verified image identity/platform mismatch")
    require(labels.get("org.opencontainers.image.revision") == source
            and labels.get("org.opencontainers.image.source") == "https://github.com/" + REPOSITORY,
            "Verified image source/revision labels mismatch")

    tag = IMAGE + ":sha-" + source
    docker("tag", image_id, tag, log=log)
    docker("push", tag, log=log)
    pushed = json.loads(docker("image", "inspect", tag, log=log))[0]
    require(pushed["Id"] == image_id, "Published tag no longer names the checked image")
    references = [ref for ref in (pushed.get("RepoDigests") or [])
                  if ref.startswith(IMAGE + "@") and SHA256.fullmatch(ref[len(IMAGE) + 1:])]
    require(len(references) == 1, "Expected exactly one GHCR repository digest after push")
    reference = references[0]
    digest = reference.split("@", 1)[1]
    # Resolve the immutable reference from GHCR and confirm its image config ID.
    docker("pull", "--platform", "linux/amd64", reference, log=log)
    pulled = json.loads(docker("image", "inspect", reference, log=log))[0]
    require(pulled["Id"] == image_id, "Registry digest does not resolve to the checked image")

    receipt = {
        "source_commit": source, "image_id": image_id, "tag": tag,
        "digest": digest, "image_reference": reference,
        "workflow_run": "https://github.com/{}/actions/runs/{}".format(
            REPOSITORY, environ["GITHUB_RUN_ID"]),
        "run_attempt": environ["GITHUB_RUN_ATTEMPT"],
        "validation": "Docker build and offline checks only; GPU training unverified",
    }
    (output / "image.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "runpod-image.txt").write_text(reference + "\n")
    with Path(environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write("digest={}\nimage-reference={}\ntag={}\n".format(digest, reference, tag))
    with Path(environ["GITHUB_STEP_SUMMARY"]).open("a") as stream:
        stream.write(
            "## GHCR publication\n\n"
            "Source commit: `{}`\n\nSHA tag: `{}`\n\n"
            "RunPod Container Image (immutable):\n\n```text\n{}\n```\n\n"
            "Receipt: `runpod-ghcr-publication` artifact. Package visibility/access must "
            "be configured before RunPod can pull a private image.\n\n"
            "**GPU training remains unverified.**\n".format(source, tag, reference)
        )
    return receipt


if __name__ == "__main__":
    publish(os.environ)
