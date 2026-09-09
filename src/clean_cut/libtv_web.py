from __future__ import annotations

import re
import tempfile
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from clean_cut.batch import BatchJob, BatchManifest, JobState, chunked, episode_stem
from clean_cut.errors import MediaProcessError
from clean_cut.libtv import detect_opening_cover_frames, restore_opening_cover_frames

StatusCallback = Callable[[BatchJob], None]


class LibTvWebBatchRunner:
    """Visible Playwright automation for LibTV's web-only subtitle eraser."""

    def __init__(
        self,
        *,
        project_url: str,
        profile_dir: Path,
        batch_size: int = 15,
        status_callback: StatusCallback | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        if not re.fullmatch(r"https://www\.liblib\.tv/canvas\?.+", project_url):
            raise ValueError("请输入完整的 LibTV 画布网址。")
        self.project_url = project_url
        self.profile_dir = profile_dir.resolve()
        self.batch_size = batch_size
        self.status_callback = status_callback or (lambda _job: None)
        self.stop_requested = stop_requested or (lambda: False)

    def run(
        self,
        manifest: BatchManifest,
        *,
        clean_dir: Path,
        skip_existing: bool = True,
    ) -> None:
        try:
            from playwright.sync_api import Page, sync_playwright
        except ImportError as exc:
            raise MediaProcessError("缺少 Playwright，请重新安装桌面版程序。") from exc

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        with sync_playwright() as playwright:
            context = playwright.chromium.launch_persistent_context(
                str(self.profile_dir), channel="chrome", headless=False, viewport=None
            )
            page: Page = context.pages[0] if context.pages else context.new_page()
            try:
                page.goto(self.project_url, wait_until="domcontentloaded", timeout=120_000)
                self._require_login(page)
                pending = self._prepare_jobs(manifest, clean_dir, skip_existing)
                self._upload_all(page, manifest, pending)
                self._submit_all(page, manifest, pending)
                self._wait_and_download(page, context, manifest, pending, clean_dir)
            finally:
                context.close()

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

    def _prepare_jobs(
        self, manifest: BatchManifest, clean_dir: Path, skip_existing: bool
    ) -> list[BatchJob]:
        pending: list[BatchJob] = []
        for job in manifest.jobs.values():
            output = clean_dir / f"{episode_stem(job.source_path)}-清水版.mp4"
            job.clean_output = str(output)
            if skip_existing and output.exists() and output.stat().st_size > 0:
                self._set(manifest, job, JobState.SKIPPED, "成品已存在，已跳过", 100)
            elif job.state == JobState.SUBMITTING:
                self._set(
                    manifest,
                    job,
                    JobState.NEEDS_REVIEW,
                    "上次在提交阶段中断，为避免重复扣积分，请检查画布",
                )
            elif job.state not in {JobState.COMPLETE, JobState.SKIPPED, JobState.NEEDS_REVIEW}:
                pending.append(job)
        return pending

    def _require_login(self, page) -> None:
        points = page.get_by_role("button", name=re.compile(r"^\d[\d,]*$"))
        login = page.get_by_role("button", name="注册/登录")
        points.or_(login).first.wait_for(state="visible", timeout=120_000)
        if login.count() and login.first.is_visible():
            raise MediaProcessError("LibTV 尚未登录，请先点击“登录 LibTV”。")

    def _upload_all(self, page, manifest: BatchManifest, jobs: list[BatchJob]) -> None:
        upload_jobs = [
            job for job in jobs if page.get_by_text(job.source_path.stem, exact=True).count() == 0
        ]
        for batch_index, batch in enumerate(
            chunked((job.source_path for job in upload_jobs), self.batch_size), 1
        ):
            self._check_stop()
            related = [manifest.jobs[str(path.resolve()).casefold()] for path in batch]
            for job in related:
                self._set(manifest, job, JobState.UPLOADING, f"上传第 {batch_index} 批")
            self._upload_files(page, batch)
            self._wait_for_uploads(page, related)
            self._reload_canvas(page)

            missing = [job for job in related if not self._source_node_exists(page, job)]
            for job in missing:
                self._check_stop()
                self._set(
                    manifest,
                    job,
                    JobState.UPLOADING,
                    "批量上传未生成节点，正在单集补传",
                )
                self._upload_files(page, [job.source_path])
                self._wait_for_uploads(page, [job])
                self._reload_canvas(page)
                if not self._source_node_exists(page, job):
                    self._set(
                        manifest,
                        job,
                        JobState.ERROR,
                        "LibTV 未生成视频节点，已停止且未提交付费任务",
                    )
                    raise MediaProcessError(
                        f"{job.source_path.name} 上传后仍未生成画布节点；"
                        "程序已在付费提交前安全停止。"
                    )
            for job in related:
                self._set(manifest, job, JobState.UPLOADED, "上传完成")

    def _upload_files(self, page, paths: list[Path]) -> None:
        page.get_by_role("button", name="添加节点", exact=True).click()
        with page.expect_file_chooser(timeout=30_000) as chooser_info:
            page.get_by_role("button", name="上传", exact=True).click()
        chooser_info.value.set_files([str(path) for path in paths])

    def _wait_for_uploads(self, page, jobs: list[BatchJob]) -> None:
        for job in jobs:
            page.get_by_text(job.source_path.stem, exact=True).wait_for(
                state="visible", timeout=600_000
            )
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            uploading = page.get_by_text(re.compile(r"上传中"))
            if not any(item.is_visible() for item in uploading.all()):
                page.wait_for_timeout(3_000)
                return
            self._check_stop()
            page.wait_for_timeout(1_000)
        raise MediaProcessError("等待 LibTV 上传完成超时；尚未提交付费任务。")

    def _reload_canvas(self, page) -> None:
        page.reload(wait_until="domcontentloaded", timeout=120_000)
        self._require_login(page)
        page.wait_for_timeout(3_000)

    @staticmethod
    def _source_node_exists(page, job: BatchJob) -> bool:
        return page.get_by_text(job.source_path.stem, exact=True).count() > 0

    def _submit_all(self, page, manifest: BatchManifest, jobs: list[BatchJob]) -> None:
        for job in jobs:
            self._check_stop()
            output_name = f"视频一键去字幕-{job.source_path.stem}"
            if page.get_by_text(output_name, exact=True).count():
                self._set(manifest, job, JobState.SUBMITTED, "画布任务已存在")
                continue
            source_node = page.get_by_text(job.source_path.stem, exact=True)
            if source_node.count() != 1:
                raise MediaProcessError(
                    f"{job.source_path.name} 的画布视频节点不存在或不唯一；"
                    "已在付费提交前安全停止。"
                )
            source_node.click(timeout=120_000)
            page.get_by_role("button", name="智能去字幕", exact=True).click()
            page.get_by_text("智能擦除", exact=True).wait_for(state="visible", timeout=30_000)
            generate = page.get_by_text("智能擦除", exact=True).locator(
                "xpath=following::button[1]"
            )
            if generate.count() != 1:
                raise MediaProcessError(f"无法唯一识别 {job.source_path.name} 的生成按钮。")
            self._set(manifest, job, JobState.SUBMITTING, "正在提交，禁止自动重试")
            generate.click()
            self._set(manifest, job, JobState.SUBMITTED, "已提交云端任务")

    def _wait_and_download(
        self, page, context, manifest: BatchManifest, jobs: list[BatchJob], clean_dir: Path
    ) -> None:
        remaining = list(jobs)
        while remaining:
            self._check_stop()
            for job in list(remaining):
                output_name = f"视频一键去字幕-{job.source_path.stem}"
                output_label = page.get_by_text(output_name, exact=True)
                if output_label.count() == 0:
                    self._set(manifest, job, JobState.GENERATING, "等待云端创建任务节点")
                    continue
                card_text = output_label.locator("xpath=ancestor::*[contains(., '生成中')][1]")
                if card_text.count():
                    text = card_text.first.inner_text()
                    match = re.search(r"生成中\s*(\d+)%", text)
                    progress = int(match.group(1)) if match else job.progress
                    self._set(manifest, job, JobState.GENERATING, f"生成中 {progress}%", progress)
                    continue
                self._download_one(page, context, manifest, job, clean_dir, output_name)
                remaining.remove(job)
            if remaining:
                time.sleep(10)

    def _download_one(
        self, page, context, manifest, job, clean_dir: Path, output_name: str
    ) -> None:
        self._set(manifest, job, JobState.DOWNLOADING, "正在下载清水版")
        page.get_by_text(output_name, exact=True).click()
        video = page.locator("video:visible").first
        video.wait_for(state="visible", timeout=60_000)
        media_url = video.evaluate("element => element.currentSrc || element.src")
        if not media_url:
            raise MediaProcessError(f"{job.source_path.name} 未找到下载地址。")

        output = clean_dir / f"{episode_stem(job.source_path)}-清水版.mp4"
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
            frame_count = detect_opening_cover_frames(job.source_path)
            self._set(
                manifest,
                job,
                JobState.RESTORING_COVER,
                f"恢复封面前 {frame_count} 帧",
                99,
            )
            restore_opening_cover_frames(job.source_path, raw, output, frame_count=frame_count)
        job.clean_output = str(output)
        self._set(manifest, job, JobState.COMPLETE, "处理完成", 100)

    def _set(
        self,
        manifest: BatchManifest,
        job: BatchJob,
        state: JobState,
        message: str,
        progress: int | None = None,
    ) -> None:
        manifest.update(job, state, message, progress)
        self.status_callback(job)

    def _check_stop(self) -> None:
        if self.stop_requested():
            raise MediaProcessError("用户已停止任务；云端已提交的任务不会被取消。")
