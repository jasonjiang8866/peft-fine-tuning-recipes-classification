"""No registry access: test publication policy, image identity and failure receipts."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/publish_image.py"
SPEC = importlib.util.spec_from_file_location("ghcr_publish", SCRIPT)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class GhcrPublishTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checks = self.root / "checks"
        self.checks.mkdir()
        self.output = self.root / "publication"
        self.source = "a" * 40
        self.image_id = "sha256:" + "b" * 64
        self.digest = "sha256:" + "c" * 64
        self.reference = publisher.IMAGE + "@" + self.digest
        (self.checks / "source-commit.txt").write_text(self.source + "\n")
        (self.checks / "verified-image-id.txt").write_text(self.image_id + "\n")
        self.env = {
            "GITHUB_REPOSITORY": publisher.REPOSITORY,
            "GITHUB_REF": "refs/heads/main", "GITHUB_EVENT_NAME": "push",
            "GITHUB_SHA": self.source, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_OUTPUT": str(self.root / "outputs"),
            "GITHUB_STEP_SUMMARY": str(self.root / "summary"),
        }
        self.image = {
            "Id": self.image_id, "Os": "linux", "Architecture": "amd64",
            "Config": {"Labels": {
                "org.opencontainers.image.revision": self.source,
                "org.opencontainers.image.source": "https://github.com/" + publisher.REPOSITORY,
            }}, "RepoDigests": [self.reference],
        }
        self.commands = []

    def fake_run(self, command, **kwargs):
        self.commands.append(command)
        text = json.dumps([self.image]) if command[1:3] == ["image", "inspect"] else "OK\n"
        return subprocess.CompletedProcess(command, 0, text)

    def publish(self):
        with mock.patch.object(publisher.subprocess, "run", side_effect=self.fake_run):
            with redirect_stdout(io.StringIO()):
                return publisher.publish(self.env, self.checks, self.output)

    def assert_no_receipt(self):
        self.assertFalse((self.output / "image.json").exists())
        self.assertFalse((self.output / "runpod-image.txt").exists())
        self.assertFalse(Path(self.env["GITHUB_OUTPUT"]).exists())

    def test_main_push_publishes_checked_id_and_resolves_digest(self):
        receipt = self.publish()
        tag = publisher.IMAGE + ":sha-" + self.source
        self.assertEqual(self.commands, [
            ["docker", "image", "inspect", self.image_id],
            ["docker", "tag", self.image_id, tag],
            ["docker", "push", tag],
            ["docker", "image", "inspect", tag],
            ["docker", "pull", "--platform", "linux/amd64", self.reference],
            ["docker", "image", "inspect", self.reference],
        ])
        self.assertEqual(receipt["image_reference"], self.reference)
        self.assertEqual(json.loads((self.output / "image.json").read_text()), receipt)
        self.assertEqual((self.output / "runpod-image.txt").read_text(), self.reference + "\n")
        self.assertIn("image-reference=" + self.reference, Path(self.env["GITHUB_OUTPUT"]).read_text())

    def test_explicit_main_dispatch_is_allowed(self):
        self.env.update(GITHUB_EVENT_NAME="workflow_dispatch", PUBLISH_REQUESTED="true")
        self.assertEqual(self.publish()["digest"], self.digest)

    def test_untrusted_events_and_refs_never_invoke_docker(self):
        for changes in [
            {"GITHUB_EVENT_NAME": "pull_request"},
            {"GITHUB_EVENT_NAME": "pull_request_target"},
            {"GITHUB_EVENT_NAME": "release"},
            {"GITHUB_EVENT_NAME": "workflow_run"},
            {"GITHUB_REPOSITORY": "fork/project"},
            {"GITHUB_REF": "refs/heads/benchmark/optional-muon"},
            {"GITHUB_REF": "refs/tags/v1"},
            {"GITHUB_EVENT_NAME": "workflow_dispatch"},
            {"GITHUB_EVENT_NAME": "workflow_dispatch", "PUBLISH_REQUESTED": "false"},
            {"GITHUB_EVENT_NAME": "workflow_dispatch", "PUBLISH_REQUESTED": "true",
             "GITHUB_REF": "refs/heads/feature"},
        ]:
            with self.subTest(changes=changes):
                env = dict(self.env, **changes)
                with mock.patch.object(publisher.subprocess, "run") as run:
                    with self.assertRaises(RuntimeError):
                        publisher.publish(env, self.checks, self.output)
                    run.assert_not_called()
                self.assert_no_receipt()

    def test_absent_check_success_marker_never_invokes_docker(self):
        (self.checks / "verified-image-id.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.publish()
        self.assertEqual(self.commands, [])
        self.assert_no_receipt()

    def test_different_checked_commit_never_invokes_docker(self):
        self.env["GITHUB_SHA"] = "d" * 40
        with self.assertRaises(RuntimeError):
            self.publish()
        self.assertEqual(self.commands, [])

    def test_wrong_revision_label_never_pushes(self):
        self.image["Config"]["Labels"]["org.opencontainers.image.revision"] = "d" * 40
        with self.assertRaises(RuntimeError):
            self.publish()
        self.assertEqual(len(self.commands), 1)
        self.assert_no_receipt()

    def test_push_failure_has_no_success_receipt_or_outputs(self):
        original = self.fake_run

        def denied(command, **kwargs):
            if command[1] == "push":
                self.commands.append(command)
                return subprocess.CompletedProcess(command, 1, "denied\n")
            return original(command, **kwargs)

        with mock.patch.object(self, "fake_run", side_effect=denied):
            with self.assertRaises(subprocess.CalledProcessError):
                self.publish()
        self.assertEqual(self.commands[-1][1], "push")
        self.assertIn("denied", (self.output / "publication.log").read_text())
        self.assert_no_receipt()

    def test_invalid_registry_digest_has_no_success_receipt(self):
        self.image["RepoDigests"] = [publisher.IMAGE + "@sha256:invalid"]
        with self.assertRaises(RuntimeError):
            self.publish()
        self.assert_no_receipt()

    def test_registry_digest_must_resolve_to_checked_image(self):
        original = self.fake_run

        def different_image(command, **kwargs):
            result = original(command, **kwargs)
            if command[1:3] == ["image", "inspect"] and command[-1] == self.reference:
                wrong = dict(self.image, Id="sha256:" + "d" * 64)
                return subprocess.CompletedProcess(command, 0, json.dumps([wrong]))
            return result

        with mock.patch.object(self, "fake_run", side_effect=different_image):
            with self.assertRaises(RuntimeError):
                self.publish()
        self.assert_no_receipt()


if __name__ == "__main__":
    unittest.main()
