from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from clean_cut.asr import FasterWhisperTranscriber
from clean_cut.batch import (
    BatchJob,
    BatchManifest,
    JobState,
    discover_videos,
    episode_number,
    episode_stem,
)
from clean_cut.dependencies import detect_dependencies, install_dependency
from clean_cut.errors import CleanCutError
from clean_cut.libtv_web import LibTvWebBatchRunner
from clean_cut.series_queue import SeriesQueueStore, SeriesTask, task_from_source
from clean_cut.tools import write_text_atomically

PROJECT_URL_FILE = ".clean-cut-project-url.txt"
DEFAULT_WORKSPACE_ID = 7887875
LOGIN_FALLBACK_URL = (
    "https://www.liblib.tv/canvas?"
    "spaceId=7887875&projectId=a282f30b20a04d8aac4e32d20f901f73"
)


class CleanCutApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("清水版批量制作")
        self.geometry("1120x720")
        self.minsize(920, 700)
        self.configure(bg="#F0FDFA")
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._rows: dict[str, str] = {}
        self._queue_rows: dict[str, str] = {}
        self._active_series_id = ""
        self._queue_store = SeriesQueueStore(self._app_data_dir() / "series-queue.json")
        self._series_tasks = self._queue_store.load()

        self.source_var = tk.StringVar()
        self.clean_var = tk.StringVar()
        self.srt_var = tk.StringVar()
        self.project_var = tk.StringVar()
        self.batch_var = tk.IntVar(value=15)
        self.skip_var = tk.BooleanVar(value=True)
        self.srt_enabled_var = tk.BooleanVar(value=True)
        self.asr_device_var = tk.StringVar(value="auto")
        self.summary_var = tk.StringVar(value="请选择包含剧集视频的文件夹")

        self._configure_style()
        self._build_ui()
        self._render_queue()
        self.after(150, self._drain_events)
        self.after(700, self._check_dependencies_on_startup)

    @staticmethod
    def _app_data_dir() -> Path:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "ProduceCleanCut"

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#F0FDFA")
        style.configure("Card.TFrame", background="#FFFFFF")
        style.configure("TLabel", background="#F0FDFA", foreground="#134E4A", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 22), foreground="#134E4A")
        style.configure("Muted.TLabel", foreground="#475569")
        style.configure(
            "Primary.TButton",
            background="#0D9488",
            foreground="#000000",
            padding=(18, 11),
            font=("Segoe UI Semibold", 10),
        )
        style.map("Primary.TButton", background=[("active", "#14B8A6"), ("disabled", "#99F6E4")])
        style.configure(
            "Accent.TButton", background="#EA580C", foreground="#000000", padding=(18, 11)
        )
        style.map("Accent.TButton", background=[("active", "#F97316")])
        style.configure("Treeview", rowheight=30, font=("Segoe UI", 9), fieldbackground="#FFFFFF")
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9), background="#E8F1F4")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=24)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="清水版批量制作", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            root,
            text="支持多部剧排队无人值守；逐集生成、立即下载并恢复封面。",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 10))

        queue_card = ttk.Frame(root, style="Card.TFrame", padding=10)
        queue_card.pack(fill="x", pady=(0, 10))
        queue_actions = ttk.Frame(queue_card, style="Card.TFrame")
        queue_actions.pack(fill="x", pady=(0, 6))
        ttk.Label(
            queue_actions, text="多剧任务队列", background="#FFFFFF", font=("Segoe UI Semibold", 11)
        ).pack(side="left")
        ttk.Button(
            queue_actions, text="添加剧集文件夹", command=self._add_series
        ).pack(side="left", padx=(14, 4))
        ttk.Button(queue_actions, text="移除选中", command=self._remove_series).pack(
            side="left", padx=4
        )
        ttk.Button(
            queue_actions, text="上移", command=lambda: self._move_series(-1)
        ).pack(side="left", padx=4)
        ttk.Button(
            queue_actions, text="下移", command=lambda: self._move_series(1)
        ).pack(side="left", padx=4)
        ttk.Button(
            queue_actions, text="环境检测 / 一键安装", command=self._open_dependency_center
        ).pack(side="right")
        self.queue_tree = ttk.Treeview(
            queue_card,
            columns=("series", "source", "state", "message"),
            show="headings",
            height=4,
        )
        for column, title, width in (
            ("series", "剧名", 180),
            ("source", "成片文件夹", 430),
            ("state", "状态", 100),
            ("message", "说明", 300),
        ):
            self.queue_tree.heading(column, text=title)
            self.queue_tree.column(column, width=width, anchor="w")
        self.queue_tree.pack(fill="x")
        self.queue_tree.bind("<<TreeviewSelect>>", self._select_series)

        form = ttk.Frame(root, style="Card.TFrame", padding=18)
        form.pack(fill="x")
        self._path_row(form, 0, "成片文件夹", self.source_var, self._choose_source)
        self._path_row(
            form, 1, "清水版文件夹", self.clean_var, lambda: self._choose_dir(self.clean_var)
        )
        self._path_row(form, 2, "SRT 文件夹", self.srt_var, lambda: self._choose_dir(self.srt_var))
        ttk.Label(form, text="LibTV 画布网址", background="#FFFFFF").grid(
            row=3, column=0, sticky="w", pady=7
        )
        ttk.Entry(form, textvariable=self.project_var).grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=(12, 0), pady=7
        )
        form.columnconfigure(1, weight=1)

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=16)
        ttk.Checkbutton(actions, text="跳过已有清水版", variable=self.skip_var).pack(side="left")
        ttk.Checkbutton(actions, text="生成 SRT", variable=self.srt_enabled_var).pack(
            side="left", padx=(20, 0)
        )
        ttk.Label(actions, text="识别设备").pack(side="left", padx=(20, 6))
        ttk.Combobox(
            actions,
            textvariable=self.asr_device_var,
            values=("auto", "cuda", "cpu"),
            width=7,
            state="readonly",
        ).pack(side="left")
        ttk.Label(actions, text="每批").pack(side="left", padx=(20, 6))
        ttk.Spinbox(actions, from_=1, to=15, textvariable=self.batch_var, width=5).pack(side="left")
        ttk.Label(actions, text="集").pack(side="left", padx=(6, 0))
        self.login_button = ttk.Button(actions, text="登录 LibTV", command=self._open_login)
        self.login_button.pack(side="right")
        self.stop_button = ttk.Button(
            actions, text="停止", command=self._request_stop, state="disabled"
        )
        self.stop_button.pack(side="right", padx=8)
        self.start_button = ttk.Button(
            actions, text="开始批量处理", style="Accent.TButton", command=self._start
        )
        self.start_button.pack(side="right")

        ttk.Label(root, textvariable=self.summary_var, style="Muted.TLabel").pack(
            anchor="w", pady=(0, 8)
        )
        columns = ("episode", "file", "state", "progress", "message")
        self.tree = ttk.Treeview(root, columns=columns, show="headings")
        headings = {
            "episode": "集数",
            "file": "文件",
            "state": "状态",
            "progress": "进度",
            "message": "说明",
        }
        widths = {"episode": 70, "file": 330, "state": 110, "progress": 80, "message": 360}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="w")
        self.tree.pack(fill="both", expand=True)

    def _render_queue(self) -> None:
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)
        self._queue_rows.clear()
        for task in self._series_tasks:
            row = self.queue_tree.insert(
                "",
                "end",
                values=(task.name, task.source, task.state, task.message),
            )
            self._queue_rows[task.task_id] = row

    def _save_queue(self) -> None:
        self._queue_store.save(self._series_tasks)

    def _add_series(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("任务运行中", "请先停止当前队列，再添加新剧。")
            return
        folder = filedialog.askdirectory(title="选择一部剧的成片文件夹")
        if not folder:
            return
        source = Path(folder)
        videos = discover_videos(source) if source.is_dir() else []
        if not videos:
            messagebox.showerror("无法添加", "该文件夹中没有支持的视频文件。")
            return
        key = str(source.resolve()).casefold()
        existing = next(
            (
                task
                for task in self._series_tasks
                if str(Path(task.source).resolve()).casefold() == key
            ),
            None,
        )
        if existing:
            existing.state = "pending"
            existing.message = "已重新加入队列"
            task = existing
        else:
            task = task_from_source(source)
            self._series_tasks.append(task)
        self._save_queue()
        self._render_queue()
        row = self._queue_rows[task.task_id]
        self.queue_tree.selection_set(row)
        self.queue_tree.see(row)
        self._load_series_into_form(task)

    def _selected_task(self) -> SeriesTask | None:
        selected = self.queue_tree.selection()
        if len(selected) != 1:
            return None
        row = selected[0]
        return next(
            (task for task in self._series_tasks if self._queue_rows.get(task.task_id) == row),
            None,
        )

    def _select_series(self, _event=None) -> None:
        task = self._selected_task()
        if task and not (self._worker and self._worker.is_alive()):
            self._load_series_into_form(task)

    def _load_series_into_form(self, task: SeriesTask) -> None:
        self.source_var.set(task.source)
        self.clean_var.set(task.clean)
        self.srt_var.set(task.srt)
        self.project_var.set(task.project_url)
        videos = discover_videos(Path(task.source)) if Path(task.source).is_dir() else []
        self._active_series_id = task.task_id
        self._load_rows(videos)

    def _remove_series(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("任务运行中", "请先停止当前队列。")
            return
        task = self._selected_task()
        if not task:
            return
        self._series_tasks = [item for item in self._series_tasks if item.task_id != task.task_id]
        self._save_queue()
        self._render_queue()

    def _move_series(self, offset: int) -> None:
        if self._worker and self._worker.is_alive():
            return
        task = self._selected_task()
        if not task:
            return
        index = self._series_tasks.index(task)
        target = max(0, min(len(self._series_tasks) - 1, index + offset))
        if target == index:
            return
        self._series_tasks.insert(target, self._series_tasks.pop(index))
        self._save_queue()
        self._render_queue()
        self.queue_tree.selection_set(self._queue_rows[task.task_id])

    def _path_row(self, parent, row: int, label: str, variable: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label, background="#FFFFFF").grid(
            row=row, column=0, sticky="w", pady=7
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=12, pady=7
        )
        ttk.Button(parent, text="选择文件夹", command=command).grid(row=row, column=2, pady=7)

    def _choose_source(self) -> None:
        folder = filedialog.askdirectory(title="选择成片文件夹")
        if not folder:
            return
        source = Path(folder)
        self.source_var.set(str(source))
        base = source.parent
        self.clean_var.set(str(base / "清水版"))
        self.srt_var.set(str(base / "srt"))
        project_file = base / "清水版" / PROJECT_URL_FILE
        if project_file.is_file():
            self.project_var.set(project_file.read_text(encoding="utf-8").strip())
        else:
            self.project_var.set("")
        videos = discover_videos(source)
        self._load_rows(videos)
        if not self.project_var.get():
            self.summary_var.set(
                f"已找到 {len(videos)} 集；开始时会为这部剧自动创建 LibTV 画布"
            )

    def _choose_dir(self, variable: tk.StringVar) -> None:
        folder = filedialog.askdirectory(title="选择输出文件夹")
        if folder:
            variable.set(folder)

    def _load_rows(self, videos: list[Path]) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._rows.clear()
        for video in videos:
            number = episode_number(video)
            episode = f"EP{number}" if number is not None else "—"
            row = self.tree.insert(
                "", "end", values=(episode, video.name, "等待", "0%", "等待处理")
            )
            self._rows[str(video.resolve()).casefold()] = row
        batch_count = (len(videos) + self.batch_var.get() - 1) // self.batch_var.get()
        self.summary_var.set(f"已找到 {len(videos)} 集；将分 {batch_count} 批上传")

    def _profile_dir(self) -> Path:
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "ProduceCleanCut" / "libtv-profile"

    def _make_runner(
        self, callback=None, *, project_url: str | None = None
    ) -> LibTvWebBatchRunner:
        return LibTvWebBatchRunner(
            project_url=project_url or self.project_var.get().strip(),
            profile_dir=self._profile_dir(),
            batch_size=self.batch_var.get(),
            status_callback=callback,
            stop_requested=self._stop_event.is_set,
        )

    def _open_login(self) -> None:
        def work() -> None:
            try:
                self._make_runner(
                    project_url=self.project_var.get().strip() or LOGIN_FALLBACK_URL
                ).open_login()
                self._events.put(("info", "LibTV 登录窗口已关闭，登录状态会保留。"))
            except Exception as exc:
                self._events.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        required_missing = [
            item for item in detect_dependencies() if item.required and not item.installed
        ]
        if required_missing:
            messagebox.showerror(
                "运行环境不完整",
                "缺少：" + "、".join(item.name for item in required_missing)
                + "\n\n请点击“环境检测 / 一键安装”完成安装。",
            )
            self._open_dependency_center()
            return

        tasks = [task for task in self._series_tasks if task.state != "completed"]
        ephemeral = False
        if not tasks:
            source = Path(self.source_var.get())
            if not source.is_dir() or not discover_videos(source):
                messagebox.showerror("无法开始", "请添加剧集到队列，或选择有效的成片文件夹。")
                return
            task = task_from_source(source)
            task.clean = self.clean_var.get().strip() or task.clean
            task.srt = self.srt_var.get().strip() or task.srt
            task.project_url = self.project_var.get().strip()
            tasks = [task]
            ephemeral = True

        batch_size = self.batch_var.get()
        skip_existing = self.skip_var.get()
        srt_enabled = self.srt_enabled_var.get()
        asr_device = self.asr_device_var.get()
        self._stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        def work() -> None:
            completed_count = 0
            failed_count = 0
            partial_count = 0
            for task in tasks:
                if self._stop_event.is_set():
                    break
                try:
                    task.state = "running"
                    task.message = "正在准备"
                    self._queue_changed(task, ephemeral)
                    source = Path(task.source)
                    videos = discover_videos(source)
                    if not videos:
                        raise CleanCutError("成片文件夹不存在或没有支持的视频。")
                    clean = Path(task.clean)
                    clean.mkdir(parents=True, exist_ok=True)
                    manifest = BatchManifest.load_or_create(
                        clean / ".clean-cut-state.json", videos
                    )
                    self._activate_series(task, videos)
                    if not task.project_url:
                        task.message = "正在创建或恢复专用 LibTV 画布"
                        self._queue_changed(task, ephemeral)
                        task.project_url = LibTvWebBatchRunner.create_project_url(
                            task.name, workspace_id=DEFAULT_WORKSPACE_ID
                        )
                    write_text_atomically(
                        clean / PROJECT_URL_FILE, task.project_url.strip() + "\n"
                    )
                    self._queue_changed(task, ephemeral)

                    def callback(job: BatchJob, task_id: str = task.task_id) -> None:
                        self._events.put(("job", (task_id, job)))

                    runner = LibTvWebBatchRunner(
                        project_url=task.project_url,
                        profile_dir=self._profile_dir(),
                        batch_size=batch_size,
                        status_callback=callback,
                        stop_requested=self._stop_event.is_set,
                    )
                    srt_errors: list[Exception] = []

                    def generate_srt(
                        active_manifest=manifest,
                        active_task=task,
                        active_callback=callback,
                        errors=srt_errors,
                    ) -> None:
                        try:
                            self._generate_srt(
                                active_manifest,
                                Path(active_task.srt),
                                active_callback,
                                device=asr_device,
                            )
                        except Exception as exc:
                            errors.append(exc)

                    srt_worker = None
                    if srt_enabled:
                        srt_worker = threading.Thread(target=generate_srt, daemon=True)
                        srt_worker.start()
                    runner.run(
                        manifest, clean_dir=clean, skip_existing=skip_existing
                    )
                    if srt_worker:
                        srt_worker.join()
                    if srt_errors:
                        raise srt_errors[0]
                    if runner.last_failed_jobs:
                        task.state = "partial"
                        task.message = "未完成：" + "、".join(runner.last_failed_jobs)
                        partial_count += 1
                    else:
                        task.state = "completed"
                        task.message = "全部处理完成"
                        completed_count += 1
                except Exception as exc:
                    if self._stop_event.is_set():
                        task.state = "pending"
                        task.message = "已安全停止，可继续"
                    else:
                        task.state = "failed"
                        task.message = str(exc)
                        failed_count += 1
                finally:
                    self._queue_changed(task, ephemeral)

            if self._stop_event.is_set():
                summary = "队列已安全停止；再次开始会从记录继续。"
            else:
                summary = (
                    f"多剧队列处理结束：完成 {completed_count} 部，"
                    f"部分完成 {partial_count} 部，失败 {failed_count} 部。"
                )
            self._events.put(("done", summary))
            self._events.put(("idle", None))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    def _activate_series(self, task: SeriesTask, videos: list[Path]) -> None:
        ready: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._events.put(("activate_series", (task, videos, ready)))
        while not self._stop_event.is_set():
            try:
                ready.get(timeout=0.2)
                return
            except queue.Empty:
                continue

    def _queue_changed(self, task: SeriesTask, ephemeral: bool) -> None:
        if not ephemeral:
            self._save_queue()
        self._events.put(("series", task))

    def _generate_srt(
        self,
        manifest: BatchManifest,
        output_dir: Path,
        callback,
        *,
        device: str,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        active_device = "cuda" if device == "auto" else device
        compute_type = "float16" if active_device == "cuda" else "int8"
        pending = []
        for job in manifest.jobs.values():
            destination = output_dir / f"{episode_stem(job.source_path)}.srt"
            job.srt_output = str(destination)
            if not destination.exists():
                pending.append((job, destination))
        if not pending:
            return
        try:
            transcriber = FasterWhisperTranscriber(
                model="turbo", device=active_device, compute_type=compute_type
            )
        except CleanCutError:
            if device != "auto" or active_device != "cuda":
                raise
            active_device = "cpu"
            transcriber = FasterWhisperTranscriber(
                model="turbo", device="cpu", compute_type="int8"
            )
        for index, (job, destination) in enumerate(pending, 1):
            if self._stop_event.is_set():
                raise RuntimeError("用户已停止任务。")
            job.message = f"生成 SRT（{index}/{len(pending)}）"
            callback(job)
            try:
                transcriber.write_srt(job.source_path, destination, language="en")
            except CleanCutError as first_error:
                if device != "auto" or active_device != "cuda":
                    raise
                job.message = "GPU 首次转写失败，正在重新初始化并重试"
                callback(job)
                try:
                    transcriber = FasterWhisperTranscriber(
                        model="turbo", device="cuda", compute_type="float16"
                    )
                    transcriber.write_srt(job.source_path, destination, language="en")
                except CleanCutError as retry_error:
                    active_device = "cpu"
                    self._write_gpu_error(first_error, retry_error)
                    job.message = "GPU 连续失败，已切换 CPU；详细原因已写入日志"
                    callback(job)
                    transcriber = FasterWhisperTranscriber(
                        model="turbo", device="cpu", compute_type="int8"
                    )
                    transcriber.write_srt(job.source_path, destination, language="en")
            job.message = "SRT 完成，等待清水版"
            if job.state == JobState.ERROR:
                job.state = JobState.PENDING
            manifest.save()
            callback(job)

    @staticmethod
    def _write_gpu_error(first_error: Exception, retry_error: Exception) -> None:
        base = Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
        log_path = base / "ProduceCleanCut" / "gpu-error.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"首次错误：{first_error}\n重试错误：{retry_error}\n",
            encoding="utf-8",
        )

    def _check_dependencies_on_startup(self) -> None:
        missing = [
            item for item in detect_dependencies() if item.required and not item.installed
        ]
        if not missing:
            return
        if messagebox.askyesno(
            "需要安装运行组件",
            "检测到缺少："
            + "、".join(item.name for item in missing)
            + "\n\n是否打开一键安装中心？",
        ):
            self._open_dependency_center()

    def _open_dependency_center(self) -> None:
        window = tk.Toplevel(self)
        window.title("运行环境检测与一键安装")
        window.geometry("820x430")
        window.transient(self)
        frame = ttk.Frame(window, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="必需组件缺失时可直接安装；CUDA 是可选加速项，CPU 模式不受影响。",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 10))
        tree = ttk.Treeview(
            frame,
            columns=("name", "required", "state", "detail"),
            show="headings",
            height=10,
        )
        for column, title, width in (
            ("name", "组件", 190),
            ("required", "类型", 80),
            ("state", "状态", 90),
            ("detail", "说明", 410),
        ):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w")
        tree.pack(fill="both", expand=True)
        status_var = tk.StringVar(value="检测完成")
        ttk.Label(frame, textvariable=status_var, style="Muted.TLabel").pack(
            anchor="w", pady=(8, 4)
        )
        actions = ttk.Frame(frame)
        actions.pack(fill="x")

        def refresh() -> list:
            for item in tree.get_children():
                tree.delete(item)
            items = detect_dependencies()
            for item in items:
                tree.insert(
                    "",
                    "end",
                    iid=item.key,
                    values=(
                        item.name,
                        "必需" if item.required else "可选",
                        "已安装" if item.installed else "未安装",
                        item.detail,
                    ),
                )
            return items

        def install(keys: list[str]) -> None:
            if not keys:
                messagebox.showinfo("无需安装", "所选组件均已安装。", parent=window)
                return
            install_selected.configure(state="disabled")
            install_required.configure(state="disabled")

            def progress(text: str) -> None:
                self.after(0, status_var.set, text)

            def worker() -> None:
                try:
                    for key in keys:
                        install_dependency(key, progress)
                except Exception as exc:
                    error = str(exc)
                    self.after(
                        0,
                        lambda error=error: messagebox.showerror(
                            "安装失败", error, parent=window
                        ),
                    )
                finally:
                    self.after(0, refresh)
                    self.after(0, install_selected.configure, {"state": "normal"})
                    self.after(0, install_required.configure, {"state": "normal"})
                    self.after(0, status_var.set, "安装流程结束，已重新检测")

            threading.Thread(target=worker, daemon=True).start()

        def install_selection() -> None:
            items = {item.key: item for item in detect_dependencies()}
            keys = [
                key
                for key in tree.selection()
                if key in items and not items[key].installed and items[key].installable
            ]
            install(keys)

        def install_all_required() -> None:
            keys = [
                item.key
                for item in detect_dependencies()
                if item.required and not item.installed and item.installable
            ]
            install(keys)

        install_required = ttk.Button(
            actions,
            text="一键安装全部必需项",
            style="Primary.TButton",
            command=install_all_required,
        )
        install_required.pack(side="left")
        install_selected = ttk.Button(
            actions, text="安装选中项", command=install_selection
        )
        install_selected.pack(side="left", padx=8)
        ttk.Button(actions, text="重新检测", command=refresh).pack(side="left")
        refresh()

    def _request_stop(self) -> None:
        self._stop_event.set()
        self.summary_var.set("正在安全停止；已提交的 LibTV 云端任务不会取消")
        self.stop_button.configure(state="disabled")

    def _drain_events(self) -> None:
        while True:
            try:
                kind, value = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "job":
                task_id, job = value
                assert isinstance(job, BatchJob)
                row = self._rows.get(job.key) if task_id == self._active_series_id else None
                if row:
                    values = self.tree.item(row, "values")
                    self.tree.item(
                        row,
                        values=(
                            values[0],
                            values[1],
                            job.state.value,
                            f"{job.progress}%",
                            job.message,
                        ),
                    )
                self.summary_var.set(job.message)
            elif kind == "activate_series":
                task, videos, ready = value
                self._active_series_id = task.task_id
                self.source_var.set(task.source)
                self.clean_var.set(task.clean)
                self.srt_var.set(task.srt)
                self.project_var.set(task.project_url)
                self._load_rows(videos)
                self.summary_var.set(f"正在处理《{task.name}》")
                ready.put(True)
            elif kind == "series":
                task = value
                row = self._queue_rows.get(task.task_id)
                if row:
                    self.queue_tree.item(
                        row,
                        values=(task.name, task.source, task.state, task.message),
                    )
                if task.task_id == self._active_series_id:
                    self.project_var.set(task.project_url)
            elif kind == "error":
                self.summary_var.set(str(value))
                messagebox.showerror("任务停止", str(value))
            elif kind == "info":
                messagebox.showinfo("提示", str(value))
            elif kind == "done":
                self.summary_var.set(str(value))
                messagebox.showinfo("处理完成", str(value))
            elif kind == "retry_prompt":
                failed, retry_answer = value
                retry = messagebox.askyesno(
                    "仍有未完成任务",
                    f"以下项目仍未完成：{failed}\n\n"
                    "是否继续重试未完成任务？\n"
                    "继续会复用画布中已有的付费任务，不重复提交已生成项目。\n"
                    "如果 LibTV 已明确提示视频审核未通过，建议选择“否”。",
                )
                self.summary_var.set(
                    "正在继续重试未完成任务" if retry else "已停止重试未完成任务"
                )
                retry_answer.put(retry)
            elif kind == "idle":
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
        self.after(150, self._drain_events)


def main() -> None:
    if "--gpu-smoke-test" in sys.argv:
        source_index = sys.argv.index("--gpu-smoke-test") + 1
        log_path = Path(os.environ["LOCALAPPDATA"]) / "ProduceCleanCut" / "gpu-test.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            source = Path(sys.argv[source_index])
            with tempfile.TemporaryDirectory(prefix="clean-cut-gpu-") as temp_dir:
                transcriber = FasterWhisperTranscriber(
                    model="turbo", device="cuda", compute_type="float16"
                )
                output = Path(temp_dir) / "gpu-test.srt"
                transcriber.write_srt(source, output, language="en")
                log_path.write_text(
                    f"GPU_OK\nsource={source}\nbytes={output.stat().st_size}\n",
                    encoding="utf-8",
                )
        except Exception:
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            raise
        return
    app = CleanCutApp()
    if "--smoke-test" in sys.argv:
        app.update_idletasks()
        app.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()
