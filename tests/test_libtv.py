from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from clean_cut.errors import MediaProcessError
from clean_cut.libtv import LibTvClient


class FakeLibTvClient(LibTvClient):
    def __init__(self) -> None:
        self.executable = "libtv"
        self.arguments: list[str] | None = None

    def get_node(self, project_uuid: str, node: str) -> dict[str, object]:
        return {
            "nodeKey": "template-key",
            "data": {
                "generatorType": "subtitle_erase",
                "params": {"model": "volcano-subtitle-eraser"},
            },
        }

    def get_project(self, project_uuid: str) -> dict[str, object]:
        return {
            "edges": [
                {"source": "old-source", "target": "template-key"},
                {"source": "unrelated", "target": "another-node"},
            ]
        }

    def _run_json(self, arguments: list[str], *, description: str) -> dict[str, object]:
        self.arguments = arguments
        return {
            "data": {
                "url": ["https://example.invalid/result.mp4"],
                "taskInfo": {"status": 2, "taskId": "task-1"},
            }
        }


class LibTvClientTests(TestCase):
    def test_validates_subtitle_erase_template(self) -> None:
        client = FakeLibTvClient()
        self.assertEqual(
            client.validate_subtitle_template("project", "template"),
            "template-key",
        )

    def test_repoints_template_before_synchronous_run(self) -> None:
        client = FakeLibTvClient()
        result = client.run_subtitle_template("project", "template", "new-source")

        self.assertEqual(result["data"]["taskInfo"]["status"], 2)
        self.assertEqual(
            client.arguments,
            [
                "node",
                "template-key",
                "-p",
                "project",
                "--left-rm",
                "old-source",
                "--left-add",
                "new-source",
                "-s",
                "mode=Subtitle",
                "--run",
            ],
        )

    def test_rejects_non_subtitle_template(self) -> None:
        client = FakeLibTvClient()
        with patch.object(
            client,
            "get_node",
            return_value={"nodeKey": "x", "data": {"generatorType": "default"}},
        ):
            with self.assertRaisesRegex(MediaProcessError, "不是智能去字幕模板"):
                client.validate_subtitle_template("project", "wrong")

    def test_reports_invalid_cli_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "libtv.exe"
            executable.touch()
            client = LibTvClient(executable)
            completed = subprocess.CompletedProcess(
                args=[], returncode=0, stdout="not-json", stderr=""
            )
            with patch("clean_cut.libtv.subprocess.run", return_value=completed):
                with self.assertRaisesRegex(MediaProcessError, "无效 JSON"):
                    client.get_project("project")

    def test_upload_requires_node_key(self) -> None:
        client = FakeLibTvClient()
        with patch.object(client, "_run_json", return_value=json.loads("{}")):
            with self.assertRaisesRegex(MediaProcessError, "缺少 nodeKey"):
                client.upload_video(Path("video.mp4"), "project", "video")
