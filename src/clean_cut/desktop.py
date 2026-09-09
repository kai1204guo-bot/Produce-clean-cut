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
from clean_cut.errors import CleanCutError
from clean_cut.libtv_web import LibTvWebBatchRunner
from clean_cut.tools import write_text_atomically

PROJECT_URL_FILE = ".clean-cut-project-url.txt"


class CleanCutApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("清水版批量制作")
        self.geometry("1120x720")
        self.minsize(920, 620)
        self.configure(bg="#F0FDFA")
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._rows: dict[str, str] = {}

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
        self.after(150, self._drain_events)

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
            text="每批上传 15 集，全部上传后逐集提交；支持断点记录并防止重复扣积分。",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 18))

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
                f"已找到 {len(videos)} 集；请粘贴这部剧自己的 LibTV 画布网址"
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

    def _make_runner(self, callback=None) -> LibTvWebBatchRunner:
        return LibTvWebBatchRunner(
            project_url=self.project_var.get().strip(),
            profile_dir=self._profile_dir(),
            batch_size=self.batch_var.get(),
            status_callback=callback,
            stop_requested=self._stop_event.is_set,
        )

    def _open_login(self) -> None:
        def work() -> None:
            try:
                self._make_runner().open_login()
                self._events.put(("info", "LibTV 登录窗口已关闭，登录状态会保留。"))
            except Exception as exc:
                self._events.put(("error", str(exc)))

        threading.Thread(target=work, daemon=True).start()

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        source = Path(self.source_var.get())
        clean = Path(self.clean_var.get())
        if not source.is_dir():
            messagebox.showerror("无法开始", "请选择有效的成片文件夹。")
            return
        videos = discover_videos(source)
        if not videos:
            messagebox.showerror("无法开始", "成片文件夹中没有支持的视频。")
            return
        clean.mkdir(parents=True, exist_ok=True)
        self._load_rows(videos)
        manifest = BatchManifest.load_or_create(clean / ".clean-cut-state.json", videos)
        self._stop_event.clear()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        def callback(job: BatchJob) -> None:
            self._events.put(("job", job))

        try:
            runner = self._make_runner(callback)
        except ValueError as exc:
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            messagebox.showerror("无法开始", str(exc))
            return
        write_text_atomically(
            clean / PROJECT_URL_FILE, self.project_var.get().strip() + "\n"
        )
        srt_enabled = self.srt_enabled_var.get()
        srt_output = Path(self.srt_var.get())
        asr_device = self.asr_device_var.get()

        def work() -> None:
            srt_errors: list[Exception] = []

            def generate_srt() -> None:
                try:
                    self._generate_srt(
                        manifest, srt_output, callback, device=asr_device
                    )
                except Exception as exc:
                    srt_errors.append(exc)

            srt_worker = None
            try:
                if srt_enabled:
                    srt_worker = threading.Thread(target=generate_srt, daemon=True)
                    srt_worker.start()
                while True:
                    runner.run(
                        manifest, clean_dir=clean, skip_existing=self.skip_var.get()
                    )
                    if not runner.last_failed_jobs or self._stop_event.is_set():
                        break
                    failed = "、".join(runner.last_failed_jobs)
                    retry_answer: queue.Queue[bool] = queue.Queue(maxsize=1)
                    self._events.put(("retry_prompt", (failed, retry_answer)))
                    while not self._stop_event.is_set():
                        try:
                            retry = retry_answer.get(timeout=0.2)
                            break
                        except queue.Empty:
                            continue
                    else:
                        retry = False
                    if not retry:
                        break
                if srt_worker:
                    srt_worker.join()
                if srt_errors:
                    raise srt_errors[0]
                if runner.last_failed_jobs:
                    failed = "、".join(runner.last_failed_jobs)
                    self._events.put(
                        (
                            "done",
                            f"批次其余任务已处理完成；以下项目自动重试 5 次后仍失败，"
                            f"可重新运行继续：{failed}",
                        )
                    )
                else:
                    self._events.put(("done", "全部任务处理完成。"))
            except Exception as exc:
                self._stop_event.set()
                self._events.put(("error", str(exc)))
            finally:
                self._events.put(("idle", None))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

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
                job = value
                assert isinstance(job, BatchJob)
                row = self._rows.get(job.key)
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
                    f"以下项目连续失败 5 次：{failed}\n\n"
                    "是否继续重试未完成任务？\n"
                    "继续会复用画布中已有的付费任务，不重复提交已生成项目。",
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
