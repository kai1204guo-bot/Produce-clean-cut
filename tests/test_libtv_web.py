from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clean_cut.libtv_web import LibTvWebBatchRunner

PROJECT_ID = "a282f30b20a04d8aac4e32d20f901f73"
PROJECT_URL = f"https://www.liblib.tv/canvas?spaceId=1&projectId={PROJECT_ID}"


def make_runner() -> LibTvWebBatchRunner:
    return LibTvWebBatchRunner(project_url=PROJECT_URL, profile_dir=Path("profile"))


def test_project_id_is_read_from_canvas_url() -> None:
    assert make_runner().project_id == PROJECT_ID


def test_canvas_url_requires_project_id() -> None:
    with pytest.raises(ValueError, match="projectId"):
        LibTvWebBatchRunner(
            project_url="https://www.liblib.tv/canvas?spaceId=1",
            profile_dir=Path("profile"),
        )


def test_video_node_ids_are_grouped_by_exact_name() -> None:
    payload = {
        "nodes": [
            {"id": "v-source-1", "type": "video", "name": "EP5"},
            {"id": "v-source-2", "type": "video", "name": "EP5"},
            {"id": "v-output", "type": "video", "name": "视频一键去字幕-EP5"},
            {"id": "group", "type": "group", "name": "ignored"},
        ]
    }
    completed = type("Completed", (), {"stdout": json.dumps(payload)})()

    with patch("clean_cut.libtv_web.subprocess.run", return_value=completed) as run:
        node_ids = make_runner()._video_node_ids_by_name()

    assert node_ids == {
        "EP5": ["v-source-1", "v-source-2"],
        "视频一键去字幕-EP5": ["v-output"],
    }
    assert run.call_args.args[0] == ["libtv", "node", "list", "-p", PROJECT_ID]


def test_completed_node_exposes_direct_media_url() -> None:
    details = {
        "data": {
            "url": ["https://example.test/clean.mp4"],
            "taskInfo": {"taskId": "task-1", "status": 2, "progressPercent": 100},
        }
    }

    assert make_runner()._media_url_from_details(details) == "https://example.test/clean.mp4"
    assert make_runner()._output_has_started(details)


def test_placeholder_node_has_not_started() -> None:
    details = {
        "data": {
            "url": [],
            "taskInfo": {"taskId": "", "loading": True, "status": 0},
        }
    }

    assert make_runner()._media_url_from_details(details) == ""
    assert not make_runner()._output_has_started(details)
