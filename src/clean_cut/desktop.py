from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
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
from clean_cut.libtv_web import LibTvWebBatchRunner

DEFAULT_PROJECT_URL = (
    "https://www.liblib.tv/canvas?spaceId=7887875&projectId=a282f30b20a04d8aac4e32d20f901f73"
)


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
        self.project_var = tk.StringVar(value=DEFAULT_PROJECT_URL)
        self.batch_var = tk.IntVar(value=15)
        self.skip_var = tk.BooleanVar(value=True)
        self.srt_enabled_var = tk.BooleanVar(value=True)
        self.asr_device_var = tk.StringVar(value="cuda")
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
            values=("cuda", "cpu"),
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
        self._load_rows(discover_videos(source))

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

        runner = self._make_runner(callback)
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
                runner.run(
                    manifest, clean_dir=clean, skip_existing=self.skip_var.get()
                )
                if srt_worker:
                    srt_worker.join()
                if srt_errors:
                    raise srt_errors[0]
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
        compute_type = "float16" if device == "cuda" else "int8"
        pending = []
        for job in manifest.jobs.values():
            destination = output_dir / f"{episode_stem(job.source_path)}.srt"
            job.srt_output = str(destination)
            if not destination.exists():
                pending.append((job, destination))
        if not pending:
            return
        transcriber = FasterWhisperTranscriber(
            model="turbo", device=device, compute_type=compute_type
        )
        for index, (job, destination) in enumerate(pending, 1):
            if self._stop_event.is_set():
                raise RuntimeError("用户已停止任务。")
            job.message = f"生成 SRT（{index}/{len(pending)}）"
            callback(job)
            transcriber.write_srt(job.source_path, destination, language="en")
            job.message = "SRT 完成，等待清水版"
            if job.state == JobState.ERROR:
                job.state = JobState.PENDING
            manifest.save()
            callback(job)

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
            elif kind == "idle":
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
        self.after(150, self._drain_events)


def main() -> None:
    app = CleanCutApp()
    if "--smoke-test" in sys.argv:
        app.update_idletasks()
        app.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()
