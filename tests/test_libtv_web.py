from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean_cut.batch import BatchJob, BatchManifest, JobState
from clean_cut.errors import MediaProcessError
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


def test_create_project_url_uses_requested_workspace() -> None:
    empty_list = type("Completed", (), {"stdout": json.dumps({"projectMetaList": []})})()
    created = type(
        "Completed",
        (),
        {"stdout": json.dumps({"data": {"uuid": "new-project-uuid"}})},
    )()

    with patch(
        "clean_cut.libtv_web.subprocess.run", side_effect=[empty_list, created]
    ) as run:
        url = LibTvWebBatchRunner.create_project_url(
            "BITE CLUB", workspace_id=7887875
        )

    assert url == (
        "https://www.liblib.tv/canvas?"
        "spaceId=7887875&projectId=new-project-uuid"
    )
    assert run.call_args_list[1].args[0] == [
        "libtv",
        "project",
        "create",
        "BITE CLUB",
        "-d",
        "清水版批量制作自动创建",
        "-w",
        "7887875",
    ]


def test_create_project_url_reuses_newest_auto_project() -> None:
    projects = {
        "projectMetaList": [
            {
                "uuid": "older",
                "name": "BITE CLUB",
                "description": "清水版批量制作自动创建",
                "projectSpaceId": 7887875,
                "updatedAtMs": 1,
            },
            {
                "uuid": "newer",
                "name": "BITE CLUB",
                "description": "清水版批量制作自动创建",
                "projectSpaceId": 7887875,
                "updatedAtMs": 2,
            },
        ]
    }
    completed = type("Completed", (), {"stdout": json.dumps(projects)})()

    with patch("clean_cut.libtv_web.subprocess.run", return_value=completed) as run:
        url = LibTvWebBatchRunner.create_project_url(
            "BITE CLUB", workspace_id=7887875
        )

    assert url.endswith("projectId=newer")
    assert run.call_count == 1


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


def test_active_cloud_task_count_ignores_completed_and_source_nodes() -> None:
    runner = make_runner()
    running = {
        "data": {
            "url": [],
            "taskInfo": {"taskId": "task-running", "loading": True, "status": 1},
        }
    }
    completed = {
        "data": {
            "url": ["https://example.test/clean.mp4"],
            "taskInfo": {"taskId": "task-done", "loading": False, "status": 2},
        }
    }

    with (
        patch.object(
            runner,
            "_video_node_ids_by_name",
            return_value={
                "EP1": ["source"],
                "视频一键去字幕-EP1": ["running"],
                "视频一键去字幕-EP2": ["completed"],
            },
        ),
        patch.object(
            runner,
            "_canvas_node_details",
            side_effect=lambda node_id: {"running": running, "completed": completed}[node_id],
        ),
    ):
        assert runner._active_cloud_task_count() == 1


def test_wait_and_download_retries_transient_episode_failure(tmp_path: Path) -> None:
    source = tmp_path / "EP1.mp4"
    job = BatchJob(source=str(source), episode=1)
    manifest = BatchManifest(tmp_path / "state.json", [job])
    runner = LibTvWebBatchRunner(
        project_url=PROJECT_URL,
        profile_dir=tmp_path / "profile",
        max_job_retries=3,
    )

    with (
        patch.object(runner, "_video_node_ids_by_name", return_value={}),
        patch.object(
            runner,
            "_poll_and_download_one",
            side_effect=[OSError("temporary SSL error"), True],
        ) as poll,
        patch("clean_cut.libtv_web.time.sleep"),
    ):
        runner._wait_and_download(object(), object(), manifest, [job], tmp_path)

    assert poll.call_count == 2
    assert runner.last_failed_jobs == []
    assert job.state is JobState.GENERATING


def test_source_upload_ready_uses_cli_media_url(tmp_path: Path) -> None:
    runner = make_runner()
    job = BatchJob(source=str(tmp_path / "EP9.mp4"), episode=9)
    with patch.object(
        runner,
        "_canvas_node_details",
        return_value={"data": {"url": ["https://example.test/EP9.mp4"]}},
    ):
        assert runner._source_upload_ready({"EP9": ["node-9"]}, job)

    assert not runner._source_upload_ready({}, job)


def test_single_upload_fallback_uses_official_cli(tmp_path: Path) -> None:
    runner = make_runner()
    job = BatchJob(source=str(tmp_path / "EP1.mp4"), episode=1)
    completed = type(
        "Completed", (), {"stdout": json.dumps({"nodeKey": "video-node-1"})}
    )()

    with patch("clean_cut.libtv_web.subprocess.run", return_value=completed) as run:
        runner._upload_file_with_cli(job)

    assert run.call_args.args[0] == [
        "libtv",
        "upload",
        "EP1",
        "-f",
        str(job.source_path.resolve()),
        "-p",
        PROJECT_ID,
        "-t",
        "video",
    ]


def test_upload_failure_does_not_stop_remaining_jobs(tmp_path: Path) -> None:
    runner = make_runner()
    first = BatchJob(source=str(tmp_path / "EP1.mp4"), episode=1)
    second = BatchJob(source=str(tmp_path / "EP2.mp4"), episode=2)
    manifest = BatchManifest(tmp_path / "manifest.json", [first, second])
    page = MagicMock()

    with (
        patch.object(
            runner,
            "_video_node_ids_by_name",
            side_effect=[{}, {}, {}, {"EP2": ["node-2"]}],
        ),
        patch.object(runner, "_upload_files"),
        patch.object(runner, "_wait_for_uploads"),
        patch.object(runner, "_reload_canvas"),
        patch.object(
            runner,
            "_upload_file_with_cli",
            side_effect=[MediaProcessError("视频审核未通过"), None],
        ) as upload,
    ):
        runner._upload_all(page, manifest, [first, second])

    assert upload.call_count == 2
    assert first.state is JobState.ERROR
    assert "视频审核未通过" in first.message
    assert second.state is JobState.UPLOADED


def test_non_json_cli_error_keeps_official_message() -> None:
    completed = type(
        "Completed",
        (),
        {"stdout": "视频审核未通过", "stderr": ""},
    )()

    with (
        patch("clean_cut.libtv_web.subprocess.run", return_value=completed),
        pytest.raises(MediaProcessError, match="视频审核未通过"),
    ):
        LibTvWebBatchRunner._run_libtv_json(["upload"], timeout=1)
