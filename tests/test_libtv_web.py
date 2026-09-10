from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from clean_cut.batch import BatchJob, BatchManifest, JobState
from clean_cut.errors import LibTvAuthenticationError, MediaProcessError
from clean_cut.libtv_web import LibTvWebBatchRunner

PROJECT_ID = "a282f30b20a04d8aac4e32d20f901f73"
PROJECT_URL = f"https://www.liblib.tv/canvas?spaceId=1&projectId={PROJECT_ID}"


@pytest.fixture(autouse=True)
def use_plain_libtv_command():
    with patch("clean_cut.libtv_web.locate_executable", return_value=None):
        yield


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


def test_create_project_url_defaults_to_active_account_root() -> None:
    empty_list = type("Completed", (), {"stdout": json.dumps({"projectMetaList": []})})()
    created = type(
        "Completed",
        (),
        {"stdout": json.dumps({"data": {"uuid": "portable-project"}})},
    )()

    with patch(
        "clean_cut.libtv_web.subprocess.run", side_effect=[empty_list, created]
    ) as run:
        url = LibTvWebBatchRunner.create_project_url("OTHER ACCOUNT")

    assert url == "https://www.liblib.tv/canvas?projectId=portable-project"
    assert run.call_args_list[0].args[0][0:4] == [
        "libtv",
        "project",
        "list",
        "-w",
    ]
    assert run.call_args_list[0].args[0][4] == "0"
    assert run.call_args_list[1].args[0][-2:] == ["-w", "0"]


def test_prepare_project_url_keeps_canvas_accessible_to_active_account() -> None:
    account = type("Completed", (), {"stdout": json.dumps({"activeAccount": {}})})()
    project = type("Completed", (), {"stdout": json.dumps({"nodes": []})})()

    with patch(
        "clean_cut.libtv_web.subprocess.run", side_effect=[account, project]
    ) as run:
        result = LibTvWebBatchRunner.prepare_project_url("SHOW", PROJECT_URL)

    assert result == PROJECT_URL
    assert run.call_args_list[0].args[0] == ["libtv", "account", "info"]
    assert run.call_args_list[1].args[0] == ["libtv", "project", PROJECT_ID]


def test_prepare_project_url_replaces_canvas_from_another_account() -> None:
    account = type("Completed", (), {"stdout": json.dumps({"activeAccount": {}})})()
    inaccessible = __import__("subprocess").CalledProcessError(
        1,
        ["libtv", "project", PROJECT_ID],
        stderr="API Request Error: { code: 10001, msg: '用户未授权' }",
    )
    empty_list = type("Completed", (), {"stdout": json.dumps({"projectMetaList": []})})()
    created = type(
        "Completed",
        (),
        {"stdout": json.dumps({"uuid": "new-account-project"})},
    )()

    with patch(
        "clean_cut.libtv_web.subprocess.run",
        side_effect=[account, inaccessible, empty_list, created],
    ):
        result = LibTvWebBatchRunner.prepare_project_url("SHOW", PROJECT_URL)

    assert result == "https://www.liblib.tv/canvas?projectId=new-account-project"


@pytest.mark.parametrize(
    ("member_name", "expected"),
    [
        ("标准版VIP 连续包月", 7),
        ("进阶版VIP", 11),
        ("高级版", 19),
        ("豪华版VIP", None),
        ("至尊版", None),
        ("未知套餐", 7),
    ],
)
def test_cloud_concurrency_follows_membership_with_one_spare_slot(
    member_name: str, expected: int | None
) -> None:
    payload = {
        "activeAccount": {"memberAccount": {"memberName": member_name}}
    }

    assert LibTvWebBatchRunner.cloud_concurrency_for_account(payload) == expected


def test_unlimited_membership_does_not_wait_for_cloud_slot(tmp_path: Path) -> None:
    runner = LibTvWebBatchRunner(
        project_url=PROJECT_URL,
        profile_dir=tmp_path / "profile",
        max_cloud_concurrency=None,
    )
    job = BatchJob(source=str(tmp_path / "EP1.mp4"), episode=1)
    manifest = BatchManifest(tmp_path / "state.json", [job])

    with patch.object(runner, "_active_cloud_task_count") as count:
        runner._wait_for_cloud_slot(object(), object(), manifest, job, [job], tmp_path)

    count.assert_not_called()


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


def test_cli_unauthorized_has_distinct_error() -> None:
    error = __import__("subprocess").CalledProcessError(
        1,
        ["libtv", "project", "list"],
        stderr="API Request Error: { code: 10001, msg: '用户未授权' }",
    )

    with patch("clean_cut.libtv_web.subprocess.run", side_effect=error):
        with pytest.raises(LibTvAuthenticationError, match="登录授权已失效"):
            LibTvWebBatchRunner.create_project_url(
                "BITE CLUB", workspace_id=7887875
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


def test_active_cloud_task_count_syncs_visible_progress(tmp_path: Path) -> None:
    runner = make_runner()
    running_job = BatchJob(source=str(tmp_path / "EP1.mp4"), episode=1)
    completed_job = BatchJob(source=str(tmp_path / "EP2.mp4"), episode=2)
    manifest = BatchManifest(tmp_path / "state.json", [running_job, completed_job])
    details = {
        "running": {
            "data": {
                "url": [],
                "taskInfo": {
                    "taskId": "task-running",
                    "loading": True,
                    "status": 1,
                    "progressPercent": 36,
                },
            }
        },
        "completed": {
            "data": {
                "url": ["https://example.test/clean.mp4"],
                "taskInfo": {
                    "taskId": "task-done",
                    "loading": False,
                    "status": 2,
                    "progressPercent": 100,
                },
            }
        },
    }

    with (
        patch.object(
            runner,
            "_video_node_ids_by_name",
            return_value={
                "视频一键去字幕-EP1": ["running"],
                "视频一键去字幕-EP2": ["completed"],
            },
        ),
        patch.object(
            runner,
            "_canvas_node_details",
            side_effect=lambda node_id: details[node_id],
        ),
    ):
        assert runner._active_cloud_task_count(manifest) == 1

    assert running_job.state is JobState.GENERATING
    assert running_job.progress == 36
    assert completed_job.state is JobState.GENERATING
    assert completed_job.progress == 100
    assert completed_job.message == "云端生成完成，等待下载"


def test_downloads_completed_result_during_submission(tmp_path: Path) -> None:
    runner = make_runner()
    first = BatchJob(source=str(tmp_path / "EP1.mp4"), episode=1)
    second = BatchJob(source=str(tmp_path / "EP2.mp4"), episode=2)
    manifest = BatchManifest(tmp_path / "state.json", [first, second])
    context = object()
    details = {
        "data": {
            "url": ["https://example.test/clean.mp4"],
            "taskInfo": {"taskId": "done", "status": 2, "progressPercent": 100},
        }
    }

    with (
        patch.object(
            runner,
            "_video_node_ids_by_name",
            return_value={
                "视频一键去字幕-EP1": ["output-1"],
                "视频一键去字幕-EP2": ["output-2"],
            },
        ),
        patch.object(runner, "_canvas_node_details", return_value=details),
        patch.object(runner, "_download_one") as download,
    ):
        assert runner._download_one_ready_output(
            context, manifest, [first, second], tmp_path
        )

    download.assert_called_once_with(
        context,
        manifest,
        first,
        tmp_path,
        "https://example.test/clean.mp4",
    )


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


def test_missing_completed_output_is_queued_again(tmp_path: Path) -> None:
    runner = make_runner()
    job = BatchJob(
        source=str(tmp_path / "EP1.mp4"),
        episode=1,
        state=JobState.COMPLETE,
        clean_output=str(tmp_path / "EP1-清水版.mp4"),
    )
    manifest = BatchManifest(tmp_path / "state.json", [job])

    pending = runner._prepare_jobs(manifest, tmp_path, skip_existing=True)

    assert pending == [job]
    assert job.state is JobState.PENDING
    assert job.message == "成品缺失，重新处理"


def test_segment_progress_is_reported_on_parent_row(tmp_path: Path) -> None:
    parent = BatchJob(source=str(tmp_path / "EP1.mp4"), episode=1)
    first = BatchJob(
        source=str(tmp_path / "segment-a.mp4"),
        episode=None,
        parent_source=parent.source,
        segment_index=1,
        segment_count=2,
        state=JobState.COMPLETE,
        progress=100,
    )
    second = BatchJob(
        source=str(tmp_path / "segment-b.mp4"),
        episode=None,
        parent_source=parent.source,
        segment_index=2,
        segment_count=2,
        state=JobState.GENERATING,
        progress=40,
        message="生成中 40%",
    )
    parent_manifest = BatchManifest(tmp_path / "parent.json", [parent])
    callback = MagicMock()
    runner = LibTvWebBatchRunner(
        project_url=PROJECT_URL,
        profile_dir=tmp_path / "profile",
        status_callback=callback,
    )
    runner._parent_manifest = parent_manifest
    runner._parents_by_key = {parent.key: parent}
    runner._children_by_parent = {parent.key: [first, second]}

    runner._sync_parent_status(second)

    assert parent.state is JobState.GENERATING
    assert parent.progress == 70
    assert "分段 2/2" in parent.message
    assert "已完成 1/2" in parent.message
    callback.assert_called_once_with(parent)


def test_completed_segments_are_merged_once_for_parent(tmp_path: Path) -> None:
    source = tmp_path / "EP1.mp4"
    source.touch()
    parent = BatchJob(source=str(source), episode=1)
    children = []
    for index in (1, 2):
        clean = tmp_path / f"segment-{index}-clean.mp4"
        clean.touch()
        children.append(
            BatchJob(
                source=str(tmp_path / f"segment-{index}.mp4"),
                episode=None,
                clean_output=str(clean),
                parent_source=str(source),
                segment_index=index,
                segment_count=2,
                state=JobState.COMPLETE,
                progress=100,
            )
        )
    manifest = BatchManifest(tmp_path / "parent.json", [parent])
    runner = make_runner()
    runner._children_by_parent = {parent.key: children}

    with (
        patch("clean_cut.libtv_web.merge_libtv_segments") as merge,
        patch("clean_cut.libtv_web.discard_segment_sources") as discard,
    ):
        runner._finalize_parent_jobs(manifest, [parent], tmp_path)

    merge.assert_called_once_with(
        source,
        children,
        tmp_path / "EP1-清水版.mp4",
    )
    discard.assert_called_once_with(children)
    assert parent.state is JobState.COMPLETE
    assert parent.progress == 100
