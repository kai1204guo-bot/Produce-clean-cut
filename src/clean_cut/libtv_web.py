from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from clean_cut.batch import BatchJob, BatchManifest, JobState, chunked, episode_stem
from clean_cut.errors import LibTvAuthenticationError, MediaProcessError
from clean_cut.libtv import detect_opening_cover_frames, restore_opening_cover_frames
from clean_cut.segmentation import (
    discard_segment_sources,
    merge_libtv_segments,
    split_for_libtv,
)
from clean_cut.tools import locate_executable

StatusCallback = Callable[[BatchJob], None]


class LibTvWebBatchRunner:
    """Visible Playwright automation for LibTV's web-only subtitle eraser."""

    def __init__(
        self,
        *,
        project_url: str,
        profile_dir: Path,
        batch_size: int = 15,
        max_cloud_concurrency: int = 7,
        max_job_retries: int = 5,
        status_callback: StatusCallback | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        if not re.fullmatch(r"https://www\.liblib\.tv/canvas\?.+", project_url):
            raise ValueError("请输入完整的 LibTV 画布网址。")
        self.project_url = project_url
        project_ids = parse_qs(urlparse(project_url).query).get("projectId", [])
        if len(project_ids) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", project_ids[0]):
            raise ValueError("LibTV 画布网址缺少有效的 projectId。")
        self.project_id = project_ids[0]
        self.profile_dir = profile_dir.resolve()
        self.batch_size = batch_size
        self.max_cloud_concurrency = max_cloud_concurrency
        self.max_job_retries = max(1, max_job_retries)
        self.status_callback = status_callback or (lambda _job: None)
        self.stop_requested = stop_requested or (lambda: False)
        self.last_failed_jobs: list[str] = []
        self._parent_manifest: BatchManifest | None = None
        self._work_manifest: BatchManifest | None = None
        self._parents_by_key: dict[str, BatchJob] = {}
        self._children_by_parent: dict[str, list[BatchJob]] = {}

    @classmethod
    def create_project_url(cls, name: str, *, workspace_id: int) -> str:
        existing_id = cls._find_auto_project(name, workspace_id=workspace_id)
        if existing_id:
            return cls._format_project_url(existing_id, workspace_id=workspace_id)
        payload = cls._run_libtv_json(
            [
                "project",
                "create",
                name,
                "-d",
                "清水版批量制作自动创建",
                "-w",
                str(workspace_id),
            ],
            timeout=120,
        )
        project_id = cls._project_uuid_from_payload(payload)
        if not project_id:
            project_id = cls._find_auto_project(name, workspace_id=workspace_id)
        if not project_id:
            raise MediaProcessError("LibTV 已创建画布，但没有返回有效的画布 UUID。")
        return cls._format_project_url(project_id, workspace_id=workspace_id)

    @staticmethod
    def _format_project_url(project_id: str, *, workspace_id: int) -> str:
        return (
            "https://www.liblib.tv/canvas?"
            f"spaceId={workspace_id}&projectId={project_id}"
        )

    @staticmethod
    def _project_uuid_from_payload(payload: dict) -> str:
        candidates = [payload.get("uuid"), payload.get("projectUuid")]
        for key in ("project", "data"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                candidates.extend((nested.get("uuid"), nested.get("projectUuid")))
        for candidate in candidates:
            if isinstance(candidate, str) and re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
                return candidate
        return ""

    @classmethod
    def _find_auto_project(cls, name: str, *, workspace_id: int) -> str:
        payload = cls._run_libtv_json(
            [
                "project",
                "list",
                "-w",
                str(workspace_id),
                "-p",
                "1",
                "-s",
                "100",
                "--name",
                name,
                "-o",
                "updated_at_desc",
            ],
            timeout=120,
        )
        projects = payload.get("projectMetaList", [])
        if not isinstance(projects, list):
            return ""
        matches = [
            item
            for item in projects
            if isinstance(item, dict)
            and item.get("name") == name
            and item.get("description") == "清水版批量制作自动创建"
            and str(item.get("projectSpaceId") or item.get("folderId"))
            == str(workspace_id)
        ]
        matches.sort(
            key=lambda item: int(item.get("updatedAtMs") or item.get("createdAtMs") or 0),
            reverse=True,
        )
        for item in matches:
            project_id = cls._project_uuid_from_payload(item)
            if project_id:
                return project_id
        return ""

    def run(
        self,
        manifest: BatchManifest,
        *,
        clean_dir: Path,
        skip_existing: bool = True,
    ) -> None:
        self.last_failed_jobs = []
        try:
            from playwright.sync_api import Page, sync_playwright
        except ImportError as exc:
            raise MediaProcessError("缺少 Playwright，请重新安装桌面版程序。") from exc

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        parents = self._prepare_jobs(manifest, clean_dir, skip_existing)
        if not parents:
            return
        work_root = clean_dir / ".clean-cut-segments"
        work_jobs = self._prepare_segment_jobs(manifest, parents, clean_dir, work_root)
        work_manifest = BatchManifest.load_or_create_jobs(
            work_root / "state.json", work_jobs
        )
        self._parent_manifest = manifest
        self._work_manifest = work_manifest
        self._parents_by_key = {job.key: job for job in parents}
        self._children_by_parent = {}
        for job in work_manifest.jobs.values():
            self._children_by_parent.setdefault(job.parent_key, []).append(job)
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                str(self.profile_dir),
                channel="chrome",
                headless=True,
                viewport={"width": 1920, "height": 1080},
            )
            page: Page = context.pages[0] if context.pages else context.new_page()
            try:
                page.goto(self.project_url, wait_until="domcontentloaded", timeout=120_000)
                self._require_login(page)
                self._ensure_cli_access(page)
                pending = self._prepare_jobs(work_manifest, work_root, True)
                self._upload_all(page, work_manifest, pending)
                upload_failures = [job for job in pending if job.state == JobState.ERROR]
                ready = [job for job in pending if job.state != JobState.ERROR]
                self._submit_all(page, context, work_manifest, ready, work_root)
                self._wait_and_download(page, context, work_manifest, ready, work_root)
                for job in upload_failures:
                    self._sync_parent_status(job)
            finally:
                context.close()
        self._finalize_parent_jobs(manifest, parents, clean_dir)
        self.last_failed_jobs = [
            f"EP{job.episode}" if job.episode is not None else job.source_path.name
            for job in parents
            if job.state == JobState.ERROR
        ]

    def _prepare_segment_jobs(
        self,
        manifest: BatchManifest,
        parents: list[BatchJob],
        clean_dir: Path,
        work_root: Path,
    ) -> list[BatchJob]:
        jobs: list[BatchJob] = []
        for parent in parents:
            self._set(manifest, parent, JobState.PENDING, "正在计算安全分段", 0)
            children = split_for_libtv(parent.source_path, work_root)
            final_output = clean_dir / f"{episode_stem(parent.source_path)}-清水版.mp4"
            parent.clean_output = str(final_output)
            if len(children) == 1:
                children[0].clean_output = str(final_output)
            else:
                self._set(
                    manifest,
                    parent,
                    JobState.PENDING,
                    f"已均分为 {len(children)} 段，每段不超过 58 秒",
                    0,
                )
            jobs.extend(children)
        manifest.save()
        return jobs

    def _finalize_parent_jobs(
        self,
        manifest: BatchManifest,
        parents: list[BatchJob],
        clean_dir: Path,
    ) -> None:
        for parent in parents:
            children = self._children_by_parent.get(parent.key, [])
            failed = [child for child in children if child.state == JobState.ERROR]
            unfinished = [
                child
                for child in children
                if child.state not in {JobState.COMPLETE, JobState.SKIPPED, JobState.ERROR}
            ]
            if failed or unfinished:
                detail = failed[0].message if failed else "仍有分段未完成"
                self._set(
                    manifest,
                    parent,
                    JobState.ERROR,
                    f"分段任务未完成，可继续重试：{detail}",
                    parent.progress,
                )
                continue
            if len(children) == 1:
                child = children[0]
                parent.clean_output = child.clean_output
                state = JobState.SKIPPED if child.state == JobState.SKIPPED else JobState.COMPLETE
                self._set(manifest, parent, state, "处理完成", 100)
                continue
            output = clean_dir / f"{episode_stem(parent.source_path)}-清水版.mp4"
            try:
                self._set(
                    manifest,
                    parent,
                    JobState.RESTORING_COVER,
                    f"{len(children)} 段已完成，正在无缝合并并恢复封面",
                    99,
                )
                merge_libtv_segments(parent.source_path, children, output)
            except Exception as exc:
                self._set(
                    manifest,
                    parent,
                    JobState.ERROR,
                    f"清水分段已保留，合并失败可重试：{exc}",
                    99,
                )
                continue
            parent.clean_output = str(output)
            self._set(manifest, parent, JobState.COMPLETE, "分段合并处理完成", 100)
            discard_segment_sources(children)

    def open_login(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise MediaProcessError("缺少 Playwright，请重新安装桌面版程序。") from exc
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                str(self.profile_dir), channel="chrome", headless=False, viewport=None
            )
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(self.project_url, wait_until="domcontentloaded", timeout=120_000)
            deadline = time.monotonic() + 15 * 60
            while time.monotonic() < deadline:
                if page.is_closed():
                    break
                if page.get_by_role("button", name=re.compile(r"^\d[\d,]*$")).count():
                    break
                page.wait_for_timeout(1_000)
            context.close()

    def ensure_cli_authorized(self) -> None:
        """Refresh CLI authorization from the saved LibTV browser session."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise MediaProcessError("缺少 Playwright，请重新安装桌面版程序。") from exc
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                str(self.profile_dir),
                channel="chrome",
                headless=True,
                viewport={"width": 1920, "height": 1080},
            )
            page = context.pages[0] if context.pages else context.new_page()
            try:
                page.goto(self.project_url, wait_until="domcontentloaded", timeout=120_000)
                self._require_login(page)
                self._ensure_cli_access(page)
            finally:
                context.close()

    def _prepare_jobs(
        self, manifest: BatchManifest, clean_dir: Path, skip_existing: bool
    ) -> list[BatchJob]:
        pending: list[BatchJob] = []
        for job in manifest.jobs.values():
            output = (
                Path(job.clean_output)
                if job.clean_output
                else clean_dir / f"{episode_stem(job.source_path)}-清水版.mp4"
            )
            job.clean_output = str(output)
            if skip_existing and output.exists() and output.stat().st_size > 0:
                self._set(manifest, job, JobState.SKIPPED, "成品已存在，已跳过", 100)
            else:
                if job.state in {JobState.COMPLETE, JobState.SKIPPED}:
                    self._set(manifest, job, JobState.PENDING, "成品缺失，重新处理", 0)
                pending.append(job)
        return pending

    def _require_login(self, page) -> None:
        points = page.get_by_role("button", name=re.compile(r"^\d[\d,]*$"))
        login = page.get_by_role("button", name="注册/登录")
        points.or_(login).first.wait_for(state="visible", timeout=120_000)
        if login.count() and login.first.is_visible():
            raise MediaProcessError("LibTV 尚未登录，请先点击“登录 LibTV”。")

    def _ensure_cli_access(self, page) -> None:
        try:
            self._video_node_ids_by_name()
            return
        except MediaProcessError as exc:
            if not isinstance(exc, LibTvAuthenticationError) and not re.search(
                r"用户未授权|未登录|\b10001\b|\b401\b", str(exc)
            ):
                raise

        self._refresh_cli_login_from_browser(page)
        page.goto(self.project_url, wait_until="domcontentloaded", timeout=120_000)
        self._require_login(page)
        try:
            self._video_node_ids_by_name()
        except MediaProcessError as exc:
            raise MediaProcessError(
                "新画布的 CLI 授权刷新后仍不可用。请确认“登录 LibTV”窗口与该画布"
                "使用的是同一个账号，然后重新运行。"
            ) from exc

    def _refresh_cli_login_from_browser(self, page) -> None:
        process = subprocess.Popen(
            [locate_executable("libtv") or "libtv", "login", "web"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        stderr_lines: list[str] = []
        line_queue: queue.Queue[str] = queue.Queue()

        def read_stderr() -> None:
            assert process.stderr is not None
            for line in iter(process.stderr.readline, ""):
                stderr_lines.append(line)
                line_queue.put(line)

        threading.Thread(target=read_stderr, daemon=True).start()
        login_url = ""
        deadline = time.monotonic() + 30
        try:
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    line = line_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                match = re.search(r"https://[^\s'\"]+", line)
                if match:
                    login_url = match.group(0).rstrip(".,;，。")
                    break
            if not login_url:
                raise MediaProcessError("LibTV CLI 未返回网页登录地址，无法自动刷新授权。")
            page.goto(login_url, wait_until="domcontentloaded", timeout=120_000)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired as exc:
                raise MediaProcessError(
                    "LibTV 网页会话无法自动完成 CLI 授权，请点击“登录 LibTV”重新登录。"
                ) from exc
            if process.returncode != 0:
                detail = "".join(stderr_lines).strip() or "未返回错误详情"
                raise MediaProcessError(f"刷新 LibTV CLI 授权失败：{detail[:500]}")
        finally:
            if process.poll() is None:
                process.terminate()

    def _upload_all(self, page, manifest: BatchManifest, jobs: list[BatchJob]) -> None:
        node_ids = self._video_node_ids_by_name()
        upload_jobs = [job for job in jobs if job.source_path.stem not in node_ids]
        for batch_index, batch in enumerate(
            chunked((job.source_path for job in upload_jobs), self.batch_size), 1
        ):
            self._check_stop()
            related = [manifest.jobs[str(path.resolve()).casefold()] for path in batch]
            for job in related:
                self._set(manifest, job, JobState.UPLOADING, f"上传第 {batch_index} 批")
            self._upload_files(page, batch)
            self._wait_for_uploads(page, manifest, related)
            self._reload_canvas(page)

            node_ids = self._video_node_ids_by_name()
            missing = [job for job in related if not self._source_node_exists(node_ids, job)]
            if missing:
                page.wait_for_timeout(15_000)
                self._reload_canvas(page)
                node_ids = self._video_node_ids_by_name()
                missing = [job for job in related if not self._source_node_exists(node_ids, job)]
            for job in missing:
                self._check_stop()
                self._set(
                    manifest,
                    job,
                    JobState.UPLOADING,
                    "批量上传漏传，正在使用官方 CLI 单集补传",
                )
                try:
                    self._upload_file_with_cli(job)
                except MediaProcessError as exc:
                    self._set(
                        manifest,
                        job,
                        JobState.ERROR,
                        f"单集上传失败，已继续下一集：{exc}",
                    )
                    continue
                self._reload_canvas(page)
                node_ids = self._video_node_ids_by_name()
                if not self._source_node_exists(node_ids, job):
                    self._set(
                        manifest,
                        job,
                        JobState.ERROR,
                        "LibTV 未生成视频节点，已停止且未提交付费任务",
                    )
                    continue
            for job in related:
                if job.state != JobState.ERROR:
                    self._set(manifest, job, JobState.UPLOADED, "上传完成")

    def _upload_file_with_cli(self, job: BatchJob) -> None:
        payload = self._run_libtv_json(
            [
                "upload",
                job.source_path.stem,
                "-f",
                str(job.source_path.resolve()),
                "-p",
                self.project_id,
                "-t",
                "video",
            ],
            timeout=900,
        )
        node_id = payload.get("nodeKey") or payload.get("id")
        if not isinstance(node_id, str) or not node_id:
            raise MediaProcessError(
                f"{job.source_path.name} 已上传，但 CLI 未返回视频节点编号。"
            )

    def _upload_files(self, page, paths: list[Path]) -> None:
        page.get_by_role("button", name="添加节点", exact=True).click()
        with page.expect_file_chooser(timeout=30_000) as chooser_info:
            page.get_by_role("button", name="上传", exact=True).click()
        chooser_info.value.set_files([str(path) for path in paths])

    def _wait_for_uploads(
        self, page, manifest: BatchManifest, jobs: list[BatchJob]
    ) -> None:
        deadline = time.monotonic() + 600
        last_error = ""
        missing_since: dict[str, float] = {}
        while time.monotonic() < deadline:
            self._check_stop()
            try:
                node_ids = self._video_node_ids_by_name()
                waiting: list[BatchJob] = []
                missing: list[BatchJob] = []
                for job in jobs:
                    if not node_ids.get(job.source_path.stem):
                        missing.append(job)
                        missing_since.setdefault(job.key, time.monotonic())
                    elif not self._source_upload_ready(node_ids, job):
                        waiting.append(job)
            except MediaProcessError as exc:
                last_error = str(exc)
                waiting = jobs
                missing = []
            if not waiting and not missing:
                return
            now = time.monotonic()
            if missing and not waiting and all(
                now - missing_since[job.key] >= 90 for job in missing
            ):
                # Let the caller retry only the files whose upload never created a node.
                return
            for job in waiting:
                self._set(
                    manifest=manifest,
                    job=job,
                    state=JobState.UPLOADING,
                    message="视频节点已创建，等待上传完成",
                )
            for job in missing:
                self._set(
                    manifest=manifest,
                    job=job,
                    state=JobState.UPLOADING,
                    message="等待 LibTV 创建视频节点",
                )
            page.wait_for_timeout(3_000)
        detail = f"；最后错误：{last_error}" if last_error else ""
        raise MediaProcessError(
            f"等待 LibTV 上传完成超时{detail}；尚未提交付费任务。"
        )

    def _source_upload_ready(
        self, node_ids: dict[str, list[str]], job: BatchJob
    ) -> bool:
        for node_id in node_ids.get(job.source_path.stem, []):
            details = self._canvas_node_details(node_id)
            if self._media_url_from_details(details):
                return True
        return False

    def _reload_canvas(self, page) -> None:
        page.reload(wait_until="domcontentloaded", timeout=120_000)
        self._require_login(page)
        page.wait_for_timeout(3_000)

    @staticmethod
    def _source_node_exists(node_ids: dict[str, list[str]], job: BatchJob) -> bool:
        return bool(node_ids.get(job.source_path.stem))

    def _submit_all(
        self,
        page,
        context,
        manifest: BatchManifest,
        jobs: list[BatchJob],
        clean_dir: Path,
    ) -> None:
        for job in jobs:
            self._check_stop()
            if job.state in {JobState.COMPLETE, JobState.SKIPPED}:
                continue
            node_ids = self._video_node_ids_by_name()
            output_name = f"视频一键去字幕-{job.source_path.stem}"
            output_ids = node_ids.get(output_name, [])
            if output_ids:
                started = any(
                    self._output_has_started(self._canvas_node_details(output_id))
                    for output_id in output_ids
                )
                if started:
                    self._set(manifest, job, JobState.SUBMITTED, "画布任务已存在")
                    continue
                for output_id in output_ids:
                    self._delete_unstarted_output(output_id)
                self._reload_canvas(page)
                node_ids = self._video_node_ids_by_name()
                self._set(manifest, job, JobState.UPLOADED, "已清理未启动节点，正在重新提交")
            self._wait_for_cloud_slot(
                page, context, manifest, job, jobs, clean_dir
            )
            while True:
                self._select_named_canvas_node(
                    page,
                    job.source_path.stem,
                    node_ids,
                    f"{job.source_path.name} 的画布视频节点不存在；"
                    "已在付费提交前安全停止。",
                )
                smart_erase = page.get_by_role("button", name="智能去字幕", exact=True)
                smart_erase.wait_for(state="attached", timeout=30_000)
                smart_erase.dispatch_event("click")
                page.get_by_text("智能擦除", exact=True).wait_for(
                    state="visible", timeout=30_000
                )
                generate = page.get_by_text("智能擦除", exact=True).locator(
                    "xpath=following::button[1]"
                )
                if generate.count() != 1:
                    raise MediaProcessError(
                        f"无法唯一识别 {job.source_path.name} 的生成按钮。"
                    )
                self._set(manifest, job, JobState.SUBMITTING, "正在提交，禁止自动重试")
                generate.dispatch_event("click")
                output_id, concurrency_limited = self._wait_for_output_id(
                    page, output_name
                )
                if output_id is not None:
                    break
                if concurrency_limited:
                    self._set(
                        manifest,
                        job,
                        JobState.UPLOADED,
                        "LibTV 并发已满，已关闭推广弹窗，等待空闲名额",
                    )
                    self._wait_before_submit_retry(page, seconds=20)
                    node_ids = self._video_node_ids_by_name()
                    continue
                raise MediaProcessError(
                    f"{job.source_path.name} 的提交结果无法确认；为避免重复扣费，已停止。"
                )
            details = self._canvas_node_details(output_id)
            if not self._output_has_started(details):
                details = self._wait_for_output_started(output_id)
            if details is None:
                raise MediaProcessError(
                    f"{job.source_path.name} 的生成状态无法确认；"
                    "为避免重复扣费，已停止且不会自动重试。"
                )
            self._set(manifest, job, JobState.SUBMITTED, "已提交云端任务")

    def _wait_and_download(
        self, page, context, manifest: BatchManifest, jobs: list[BatchJob], clean_dir: Path
    ) -> None:
        remaining = [
            job
            for job in jobs
            if job.state not in {JobState.COMPLETE, JobState.SKIPPED, JobState.ERROR}
        ]
        failures: list[tuple[BatchJob, str]] = []
        retry_counts: dict[str, int] = {}
        node_query_failures = 0
        while remaining:
            self._check_stop()
            try:
                node_ids = self._video_node_ids_by_name()
                node_query_failures = 0
            except Exception as exc:
                node_query_failures += 1
                detail = str(exc) or type(exc).__name__
                if node_query_failures < self.max_job_retries:
                    for job in remaining:
                        self._set(
                            manifest,
                            job,
                            JobState.GENERATING,
                            f"查询 LibTV 临时失败，自动重试 "
                            f"{node_query_failures}/{self.max_job_retries}：{detail}",
                        )
                    time.sleep(min(30, 5 * node_query_failures))
                    continue
                for job in list(remaining):
                    self._set(
                        manifest,
                        job,
                        JobState.ERROR,
                        f"连续 {self.max_job_retries} 次查询失败：{detail}",
                    )
                    failures.append((job, detail))
                    remaining.remove(job)
                break
            for job in list(remaining):
                try:
                    completed = self._poll_and_download_one(
                        context, manifest, job, clean_dir, node_ids
                    )
                except Exception as exc:
                    if self.stop_requested():
                        raise
                    detail = str(exc) or type(exc).__name__
                    attempt = retry_counts.get(job.key, 0) + 1
                    retry_counts[job.key] = attempt
                    if attempt < self.max_job_retries:
                        self._set(
                            manifest,
                            job,
                            JobState.GENERATING,
                            f"临时失败，自动重试 {attempt}/{self.max_job_retries}：{detail}",
                        )
                        continue
                    self._set(
                        manifest,
                        job,
                        JobState.ERROR,
                        f"连续 {self.max_job_retries} 次失败，已继续下一集：{detail}",
                    )
                    failures.append((job, detail))
                    remaining.remove(job)
                    continue
                retry_counts.pop(job.key, None)
                if completed:
                    remaining.remove(job)
            if remaining:
                time.sleep(10)
        self.last_failed_jobs = [
            f"EP{job.episode}" if job.episode is not None else job.source_path.name
            for job, _detail in failures
        ]

    def _poll_and_download_one(
        self,
        context,
        manifest: BatchManifest,
        job: BatchJob,
        clean_dir: Path,
        node_ids: dict[str, list[str]],
    ) -> bool:
        output_name = f"视频一键去字幕-{job.source_path.stem}"
        output_ids = node_ids.get(output_name, [])
        if not output_ids:
            self._set(manifest, job, JobState.GENERATING, "等待云端创建任务节点")
            return False
        details = self._canvas_node_details(output_ids[0])
        media_url = self._media_url_from_details(details)
        if media_url:
            self._download_one(context, manifest, job, clean_dir, media_url)
            return True
        task_info = details.get("data", {}).get("taskInfo", {})
        if not task_info.get("taskId"):
            raise MediaProcessError(f"{job.source_path.name} 的去字幕节点尚未开始生成。")
        progress = int(task_info.get("progressPercent") or job.progress)
        if task_info.get("loading") or progress < 100:
            self._set(manifest, job, JobState.GENERATING, f"生成中 {progress}%", progress)
            return False
        raise MediaProcessError(f"{job.source_path.name} 的云端任务已结束，但未返回视频地址。")

    def _download_one(
        self,
        context,
        manifest,
        job,
        clean_dir: Path,
        media_url: str,
    ) -> None:
        self._set(manifest, job, JobState.DOWNLOADING, "正在下载清水版")
        output = (
            Path(job.clean_output)
            if job.clean_output
            else clean_dir / f"{episode_stem(job.source_path)}-清水版.mp4"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        cookies = context.cookies([media_url])
        cookie_header = "; ".join(f"{item['name']}={item['value']}" for item in cookies)
        headers = {"User-Agent": "Mozilla/5.0"}
        if cookie_header:
            headers["Cookie"] = cookie_header
        request = urllib.request.Request(media_url, headers=headers)

        with tempfile.TemporaryDirectory(prefix="clean-cut-libtv-", dir=clean_dir) as temp_dir:
            raw = Path(temp_dir) / "raw.mp4"
            with urllib.request.urlopen(request, timeout=120) as response, raw.open("wb") as target:
                while block := response.read(1024 * 1024):
                    self._check_stop()
                    target.write(block)
            if job.is_segment:
                if output.exists():
                    output.unlink()
                shutil.move(str(raw), output)
                job.clean_output = str(output)
                self._set(manifest, job, JobState.COMPLETE, "清水分段下载完成", 100)
                return
            frame_count = detect_opening_cover_frames(job.source_path)
            self._set(
                manifest,
                job,
                JobState.RESTORING_COVER,
                f"恢复封面前 {frame_count} 帧",
                99,
            )
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    restore_opening_cover_frames(
                        job.source_path, raw, output, frame_count=frame_count
                    )
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 0:
                        self._set(
                            manifest,
                            job,
                            JobState.RESTORING_COVER,
                            "封面恢复首次失败，正在重试",
                            99,
                        )
                        time.sleep(2)
            if last_error is not None:
                fallback = clean_dir / f"{episode_stem(job.source_path)}-清水版-封面待恢复.mp4"
                if not fallback.exists():
                    shutil.move(str(raw), fallback)
                job.clean_output = str(fallback)
                manifest.save()
                raise MediaProcessError(f"{last_error}；已保留清水视频：{fallback.name}")
        job.clean_output = str(output)
        self._set(manifest, job, JobState.COMPLETE, "处理完成", 100)

    @staticmethod
    def _select_canvas_node(label, error_message: str) -> None:
        if label.count() < 1:
            raise MediaProcessError(error_message)
        node = label.first.locator(
            "xpath=ancestor::*[contains(@class,'react-flow__node')][1]"
        )
        if node.count() != 1:
            raise MediaProcessError(error_message)
        node.dispatch_event("click")

    @staticmethod
    def _canvas_node_by_ids(page, node_ids: list[str]):
        for node_id in node_ids:
            node = page.locator(f'[data-id="{node_id}"]')
            if node.count():
                return node.first
        return None

    def _select_named_canvas_node(
        self,
        page,
        name: str,
        node_ids: dict[str, list[str]],
        error_message: str,
    ):
        ids = node_ids.get(name, [])
        node = self._canvas_node_by_ids(page, ids)
        if node is not None:
            node.dispatch_event("click")
            return node

        if ids:
            self._fit_canvas_to_screen(page)
            node = self._canvas_node_by_ids(page, ids)
            if node is not None:
                node.dispatch_event("click")
                return node

        # The CLI is authoritative for existence, while this fallback keeps the app usable
        # if an older LibTV canvas omits data-id from its rendered React Flow node.
        self._select_canvas_node(page.get_by_text(name, exact=True), error_message)
        label = page.get_by_text(name, exact=True).first
        return label.locator("xpath=ancestor::*[contains(@class,'react-flow__node')][1]")

    def _ensure_canvas_node(self, page, node_ids: list[str]):
        node = self._canvas_node_by_ids(page, node_ids)
        if node is not None:
            return node
        self._fit_canvas_to_screen(page)
        node = self._canvas_node_by_ids(page, node_ids)
        if node is not None:
            return node
        self._reload_canvas(page)
        self._fit_canvas_to_screen(page)
        return self._canvas_node_by_ids(page, node_ids)

    def _wait_for_output_id(self, page, output_name: str) -> tuple[str | None, bool]:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            self._check_stop()
            node_ids = self._video_node_ids_by_name().get(output_name, [])
            if node_ids:
                return node_ids[0], False
            if self._dismiss_concurrency_modal(page):
                return None, True
            page.wait_for_timeout(1_000)
        return None, False

    def _wait_before_submit_retry(self, page, *, seconds: int) -> None:
        for _ in range(seconds):
            self._check_stop()
            page.wait_for_timeout(1_000)

    def _wait_for_cloud_slot(
        self,
        page,
        context,
        manifest: BatchManifest,
        job: BatchJob,
        jobs: list[BatchJob],
        clean_dir: Path,
    ) -> None:
        while True:
            self._check_stop()
            self._download_one_ready_output(context, manifest, jobs, clean_dir)
            active = self._active_cloud_task_count(manifest)
            if active < self.max_cloud_concurrency:
                return
            self._set(
                manifest,
                job,
                JobState.UPLOADED,
                f"云端并发 {active}/{self.max_cloud_concurrency}，等待空闲名额",
            )
            self._wait_before_submit_retry(page, seconds=10)

    def _download_one_ready_output(
        self,
        context,
        manifest: BatchManifest,
        jobs: list[BatchJob],
        clean_dir: Path,
    ) -> bool:
        """Download one completed cloud result while later jobs are still submitting."""
        node_ids = self._video_node_ids_by_name()
        for candidate in jobs:
            if candidate.state in {
                JobState.COMPLETE,
                JobState.SKIPPED,
                JobState.ERROR,
                JobState.DOWNLOADING,
                JobState.RESTORING_COVER,
            }:
                continue
            output_name = f"视频一键去字幕-{candidate.source_path.stem}"
            output_ids = node_ids.get(output_name, [])
            if not output_ids:
                continue
            details = self._canvas_node_details(output_ids[0])
            media_url = self._media_url_from_details(details)
            if not media_url:
                continue
            try:
                self._download_one(context, manifest, candidate, clean_dir, media_url)
            except Exception as exc:
                if self.stop_requested():
                    raise
                detail = str(exc) or type(exc).__name__
                self._set(
                    manifest,
                    candidate,
                    JobState.GENERATING,
                    f"即时下载临时失败，稍后重试：{detail}",
                    100,
                )
            return True
        return False

    def _active_cloud_task_count(self, manifest: BatchManifest | None = None) -> int:
        node_ids = self._video_node_ids_by_name()
        active = 0
        jobs_by_stem = (
            {job.source_path.stem: job for job in manifest.jobs.values()}
            if manifest is not None
            else {}
        )
        for name, ids in node_ids.items():
            if not name.startswith("视频一键去字幕-"):
                continue
            job = jobs_by_stem.get(name.removeprefix("视频一键去字幕-"))
            for node_id in ids:
                details = self._canvas_node_details(node_id)
                data = details.get("data", {})
                task_info = data.get("taskInfo", {})
                media_url = self._media_url_from_details(details)
                progress = int(task_info.get("progressPercent") or 0)
                if (
                    task_info.get("taskId")
                    and not media_url
                    and (task_info.get("loading") or task_info.get("status") in {0, 1})
                ):
                    active += 1
                    self._sync_cloud_progress(
                        manifest, job, progress, f"云端生成中 {progress}%"
                    )
                elif media_url:
                    self._sync_cloud_progress(
                        manifest, job, 100, "云端生成完成，等待下载"
                    )
        return active

    def _sync_cloud_progress(
        self,
        manifest: BatchManifest | None,
        job: BatchJob | None,
        progress: int,
        message: str,
    ) -> None:
        if manifest is None or job is None:
            return
        if job.state in {JobState.COMPLETE, JobState.SKIPPED, JobState.ERROR}:
            return
        if (
            job.state == JobState.GENERATING
            and job.progress == progress
            and job.message == message
        ):
            return
        self._set(manifest, job, JobState.GENERATING, message, progress)

    @staticmethod
    def _dismiss_concurrency_modal(page) -> bool:
        message = page.get_by_text(re.compile(r"并发任务数超过限制"))
        visible = [item for item in message.all() if item.is_visible()]
        if not visible:
            return False

        dialog = visible[0].locator("xpath=ancestor::*[@role='dialog'][1]")
        if dialog.count() == 0:
            dialog = visible[0].locator(
                "xpath=ancestor::*[contains(@class,'mantine-Modal-content')][1]"
            )
        if dialog.count() == 0:
            raise MediaProcessError("识别到 LibTV 并发限制弹窗，但无法定位关闭按钮。")

        named_close = dialog.get_by_role("button", name=re.compile(r"关闭|close", re.I))
        if named_close.count():
            named_close.first.click(force=True)
            return True

        dialog_box = dialog.bounding_box()
        candidates = []
        for button in dialog.locator("button").all():
            box = button.bounding_box()
            if box and not button.inner_text().strip():
                candidates.append((button, box))
        if dialog_box and candidates:
            right = dialog_box["x"] + dialog_box["width"]
            top = dialog_box["y"]
            button, _box = min(
                candidates,
                key=lambda item: abs(item[1]["x"] + item[1]["width"] - right)
                + abs(item[1]["y"] - top),
            )
            button.click(force=True)
            return True
        raise MediaProcessError("识别到 LibTV 并发限制弹窗，但无法定位右上角 X。")

    def _wait_for_output_started(self, output_id: str) -> dict | None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            self._check_stop()
            details = self._canvas_node_details(output_id)
            if self._output_has_started(details):
                return details
            time.sleep(1)
        return None

    def _delete_unstarted_output(self, output_id: str) -> None:
        details = self._canvas_node_details(output_id)
        if self._output_has_started(details):
            raise MediaProcessError("拒绝删除已经开始生成的 LibTV 节点。")
        self._run_libtv_json(
            ["node", "delete", output_id, "-p", self.project_id], timeout=60
        )

    @staticmethod
    def _media_url_from_details(details: dict) -> str:
        urls = details.get("data", {}).get("url", [])
        if isinstance(urls, str) and urls:
            return urls
        if isinstance(urls, list):
            for url in urls:
                if isinstance(url, str) and url:
                    return url
        return ""

    @classmethod
    def _output_has_started(cls, details: dict) -> bool:
        task_info = details.get("data", {}).get("taskInfo", {})
        return bool(task_info.get("taskId") or cls._media_url_from_details(details))

    @staticmethod
    def _fit_canvas_to_screen(page) -> None:
        zoom_options = page.get_by_role("button", name="缩放选项")
        zoom_options.wait_for(state="attached", timeout=30_000)
        zoom_options.dispatch_event("click")
        fit_screen = page.get_by_role("menuitem").filter(has_text=re.compile(r"^适合屏幕"))
        fit_screen.wait_for(state="visible", timeout=30_000)
        fit_screen.dispatch_event("click")
        page.wait_for_timeout(2_000)

    def _video_node_ids_by_name(self) -> dict[str, list[str]]:
        payload = self._run_libtv_json(
            ["node", "list", "-p", self.project_id], timeout=60
        )

        result: dict[str, list[str]] = {}
        for node in payload.get("nodes", []):
            if node.get("type") != "video":
                continue
            name = node.get("name")
            node_id = node.get("id")
            if isinstance(name, str) and isinstance(node_id, str):
                result.setdefault(name, []).append(node_id)
        return result

    def _canvas_node_details(self, node_id: str) -> dict:
        return self._run_libtv_json(
            ["node", node_id, "-p", self.project_id], timeout=60
        )

    @staticmethod
    def _run_libtv_json(arguments: list[str], *, timeout: float | None) -> dict:
        try:
            completed = subprocess.run(
                [locate_executable("libtv") or "libtv", *arguments],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
                timeout=timeout,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                payload = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                detail = (completed.stderr or completed.stdout or str(exc)).strip()
                raise MediaProcessError(
                    f"LibTV CLI 操作失败：{detail[:500]}"
                ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            if re.search(r"用户未授权|未登录|\b10001\b|\b401\b", detail):
                raise LibTvAuthenticationError(
                    "LibTV 登录授权已失效，需要重新登录。"
                ) from exc
            raise MediaProcessError(f"LibTV CLI 查询失败：{detail[:500]}") from exc
        except (OSError, subprocess.SubprocessError) as exc:
            detail = str(exc).strip() or type(exc).__name__
            raise MediaProcessError(f"LibTV CLI 查询失败：{detail[:500]}") from exc
        if not isinstance(payload, dict):
            raise MediaProcessError("LibTV 官方 CLI 返回了无法识别的数据。")
        return payload

    def _set(
        self,
        manifest: BatchManifest,
        job: BatchJob,
        state: JobState,
        message: str,
        progress: int | None = None,
    ) -> None:
        manifest.update(job, state, message, progress)
        if manifest is self._work_manifest and job.parent_source:
            self._sync_parent_status(job)
        else:
            self.status_callback(job)

    def _sync_parent_status(self, changed: BatchJob) -> None:
        manifest = self._parent_manifest
        parent = self._parents_by_key.get(changed.parent_key)
        children = self._children_by_parent.get(changed.parent_key, [])
        if manifest is None or parent is None or not children:
            return
        if len(children) == 1:
            parent.clean_output = changed.clean_output
            manifest.update(parent, changed.state, changed.message, changed.progress)
            self.status_callback(parent)
            return
        progress = min(98, round(sum(item.progress for item in children) / len(children)))
        complete = sum(
            item.state in {JobState.COMPLETE, JobState.SKIPPED} for item in children
        )
        index = changed.segment_index or 1
        state = changed.state
        if state in {JobState.COMPLETE, JobState.SKIPPED, JobState.ERROR}:
            state = JobState.GENERATING
        message = (
            f"分段 {index}/{len(children)}：{changed.message}；"
            f"已完成 {complete}/{len(children)}"
        )
        manifest.update(parent, state, message, progress)
        self.status_callback(parent)

    def _check_stop(self) -> None:
        if self.stop_requested():
            raise MediaProcessError("用户已停止任务；云端已提交的任务不会被取消。")
