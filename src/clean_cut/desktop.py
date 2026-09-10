from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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
from clean_cut.errors import CleanCutError, LibTvAuthenticationError
from clean_cut.libtv_web import LibTvWebBatchRunner
from clean_cut.paths import app_data_dir
from clean_cut.series_queue import (
    SeriesQueueStore,
    SeriesTask,
    discover_series_sources,
    task_from_source,
)
from clean_cut.tools import write_text_atomically

PROJECT_URL_FILE = ".clean-cut-project-url.txt"
DEFAULT_WORKSPACE_ID = 7887875
LOGIN_FALLBACK_URL = (
    "https://www.liblib.tv/canvas?"
    "spaceId=7887875&projectId=a282f30b20a04d8aac4e32d20f901f73"
)
UI_FONT = "Microsoft YaHei UI"


class CleanCutApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("清水版批量制作")
        self.geometry("1180x800")
        self.minsize(980, 720)
        self.configure(bg="#F5F5F7")
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._queue_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._rows: dict[str, str] = {}
        self._queue_rows: dict[str, str] = {}
        self._active_series_id = ""
        self._queue_store = SeriesQueueStore(app_data_dir() / "series-queue.json")
        self._settings_path = app_data_dir() / "desktop-settings.json"
        self._series_tasks = self._queue_store.load()
        self._last_scan_root = self._load_last_scan_root()

        self.source_var = tk.StringVar()
        self.clean_var = tk.StringVar()
        self.srt_var = tk.StringVar()
        self.project_var = tk.StringVar()
        self.batch_var = tk.IntVar(value=15)
        self.skip_var = tk.BooleanVar(value=True)
        self.srt_enabled_var = tk.BooleanVar(value=True)
        self.asr_device_var = tk.StringVar(value="auto")
        self.summary_var = tk.StringVar(value="请选择包含剧集视频的文件夹")
        self.queue_summary_var = tk.StringVar(value="队列为空")
        self._details_visible = False

        self._configure_style()
        self._build_ui()
        self._render_queue()
        self.after(150, self._drain_events)
        self.after(700, self._check_dependencies_on_startup)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#F5F5F7")
        style.configure("Card.TFrame", background="#FFFFFF")
        style.configure(
            "TLabel",
            background="#F5F5F7",
            foreground="#1D1D1F",
            font=(UI_FONT, 10),
        )
        style.configure(
            "Surface.TLabel",
            background="#FFFFFF",
            foreground="#1D1D1F",
            font=(UI_FONT, 10),
        )
        style.configure(
            "Title.TLabel",
            font=(UI_FONT, 22, "bold"),
            foreground="#1D1D1F",
        )
        style.configure(
            "Section.TLabel",
            background="#FFFFFF",
            foreground="#1D1D1F",
            font=(UI_FONT, 11, "bold"),
        )
        style.configure("Muted.TLabel", foreground="#6E6E73")
        style.configure(
            "SurfaceMuted.TLabel", background="#FFFFFF", foreground="#6E6E73"
        )
        style.configure(
            "TButton",
            background="#FFFFFF",
            foreground="#1D1D1F",
            bordercolor="#C7C7CC",
            padding=(12, 8),
            font=(UI_FONT, 9),
        )
        style.map(
            "TButton",
            background=[("active", "#E9E9ED"), ("disabled", "#F2F2F7")],
            foreground=[("disabled", "#AEAEB2")],
            bordercolor=[("focus", "#0078D4")],
        )
        style.configure(
            "Primary.TButton",
            background="#0078D4",
            foreground="#FFFFFF",
            bordercolor="#0078D4",
            padding=(18, 10),
            font=(UI_FONT, 10, "bold"),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#106EBE"), ("disabled", "#D2D2D7")],
            foreground=[("disabled", "#8E8E93")],
        )
        style.configure(
            "Accent.TButton",
            background="#0078D4",
            foreground="#FFFFFF",
            bordercolor="#0078D4",
            padding=(18, 10),
            font=(UI_FONT, 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#106EBE"), ("disabled", "#D2D2D7")],
            foreground=[("disabled", "#8E8E93")],
        )
        style.configure(
            "TEntry", fieldbackground="#FFFFFF", bordercolor="#C7C7CC", padding=7
        )
        style.configure(
            "Treeview",
            rowheight=31,
            font=(UI_FONT, 9),
            background="#FFFFFF",
            fieldbackground="#FFFFFF",
            foreground="#1D1D1F",
            bordercolor="#D2D2D7",
        )
        style.map(
            "Treeview",
            background=[("selected", "#DCEEFF")],
            foreground=[("selected", "#1D1D1F")],
        )
        style.configure(
            "Treeview.Heading",
            font=(UI_FONT, 9, "bold"),
            background="#F2F2F7",
            foreground="#3A3A3C",
            padding=(8, 7),
        )

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=(24, 20, 24, 18))
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="清水版批量制作", style="Title.TLabel").pack(
            side="left", anchor="w"
        )
        ttk.Label(header, text="v0.4.0", style="Muted.TLabel").pack(
            side="left", padx=(10, 0), pady=(8, 0)
        )
        ttk.Label(
            root,
            text=(
                "一次扫描全部剧集，按队列无人值守处理；长视频自动分段并无缝合并。"
            ),
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(3, 14))

        queue_card = ttk.Frame(root, style="Card.TFrame", padding=14)
        queue_card.pack(fill="x", pady=(0, 12))
        queue_actions = ttk.Frame(queue_card, style="Card.TFrame")
        queue_actions.pack(fill="x", pady=(0, 10))
        ttk.Label(
            queue_actions, text="任务队列", style="Section.TLabel"
        ).pack(side="left")
        ttk.Button(
            queue_actions,
            text="扫描总文件夹…",
            style="Primary.TButton",
            command=self._scan_series_library,
        ).pack(side="left", padx=(16, 6))
        ttk.Button(
            queue_actions, text="添加单部剧…", command=self._add_series
        ).pack(side="left", padx=4)
        ttk.Button(queue_actions, text="移除选中", command=self._remove_series).pack(
            side="left", padx=4
        )
        ttk.Button(
            queue_actions, text="上移", command=lambda: self._move_series(-1)
        ).pack(side="left", padx=4)
        ttk.Button(
            queue_actions, text="下移", command=lambda: self._move_series(1)
        ).pack(side="left", padx=4)
        self.environment_button = ttk.Button(
            queue_actions, text="运行环境", command=self._open_dependency_center
        )
        self.environment_button.pack(side="right")
        ttk.Label(
            queue_actions,
            textvariable=self.queue_summary_var,
            style="SurfaceMuted.TLabel",
        ).pack(side="right", padx=(0, 14))
        queue_table = ttk.Frame(queue_card, style="Card.TFrame")
        queue_table.pack(fill="x")
        queue_scroll = ttk.Scrollbar(queue_table, orient="vertical")
        self.queue_tree = ttk.Treeview(
            queue_table,
            columns=("series", "source", "state", "message"),
            show="headings",
            height=5,
            yscrollcommand=queue_scroll.set,
        )
        queue_scroll.configure(command=self.queue_tree.yview)
        for column, title, width in (
            ("series", "剧名", 190),
            ("source", "成片文件夹", 440),
            ("state", "状态", 90),
            ("message", "当前进展", 330),
        ):
            self.queue_tree.heading(column, text=title)
            self.queue_tree.column(column, width=width, anchor="w")
        queue_scroll.pack(side="right", fill="y")
        self.queue_tree.pack(side="left", fill="x", expand=True)
        self.queue_tree.bind("<<TreeviewSelect>>", self._select_series)
        self.queue_tree.bind("<Double-1>", lambda _event: self._toggle_details(True))

        details = ttk.Frame(root, style="Card.TFrame", padding=(14, 10))
        details.pack(fill="x")
        details_header = ttk.Frame(details, style="Card.TFrame")
        details_header.pack(fill="x")
        ttk.Label(details_header, text="所选剧集", style="Section.TLabel").pack(
            side="left"
        )
        ttk.Label(
            details_header,
            text="双击队列项目可查看或修改路径",
            style="SurfaceMuted.TLabel",
        ).pack(side="left", padx=(12, 0))
        self.details_button = ttk.Button(
            details_header, text="显示详情", command=self._toggle_details
        )
        self.details_button.pack(side="right")
        self.details_body = ttk.Frame(details, style="Card.TFrame")
        self._path_row(
            self.details_body, 0, "成片文件夹", self.source_var, self._choose_source
        )
        self._path_row(
            self.details_body,
            1,
            "清水版文件夹",
            self.clean_var,
            lambda: self._choose_dir(self.clean_var),
        )
        self._path_row(
            self.details_body,
            2,
            "SRT 文件夹",
            self.srt_var,
            lambda: self._choose_dir(self.srt_var),
        )
        ttk.Label(
            self.details_body, text="LibTV 画布网址", style="Surface.TLabel"
        ).grid(
            row=3, column=0, sticky="w", pady=6
        )
        ttk.Entry(self.details_body, textvariable=self.project_var).grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=(12, 0), pady=6
        )
        self.details_body.columnconfigure(1, weight=1)

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(12, 10))
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
            actions, text="开始全部任务", style="Accent.TButton", command=self._start
        )
        self.start_button.pack(side="right")

        status = ttk.Frame(root, style="Card.TFrame", padding=(12, 8))
        status.pack(fill="x", pady=(0, 8))
        ttk.Label(status, text="当前状态", style="Section.TLabel").pack(side="left")
        ttk.Label(
            status, textvariable=self.summary_var, style="SurfaceMuted.TLabel"
        ).pack(side="left", padx=(12, 0))
        columns = ("episode", "file", "state", "progress", "message")
        episode_table = ttk.Frame(root)
        episode_table.pack(fill="both", expand=True)
        episode_scroll = ttk.Scrollbar(episode_table, orient="vertical")
        self.tree = ttk.Treeview(
            episode_table,
            columns=columns,
            show="headings",
            yscrollcommand=episode_scroll.set,
        )
        episode_scroll.configure(command=self.tree.yview)
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
        episode_scroll.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="both", expand=True)

    def _toggle_details(self, show: bool | None = None) -> None:
        visible = not self._details_visible if show is None else show
        if visible == self._details_visible:
            return
        self._details_visible = visible
        if visible:
            self.details_body.pack(fill="x", pady=(8, 0))
            self.details_button.configure(text="收起详情")
        else:
            self.details_body.pack_forget()
            self.details_button.configure(text="显示详情")

    def _render_queue(self) -> None:
        for item in self.queue_tree.get_children():
            self.queue_tree.delete(item)
        self._queue_rows.clear()
        for task in self._series_tasks:
            row = self.queue_tree.insert(
                "",
                "end",
                values=(
                    task.name,
                    task.source,
                    self._state_label(task.state),
                    task.message,
                ),
                tags=(task.state,),
            )
            self._queue_rows[task.task_id] = row
        self.queue_tree.tag_configure("completed", foreground="#248A3D")
        self.queue_tree.tag_configure("partial", foreground="#B25000")
        self.queue_tree.tag_configure("failed", foreground="#D70015")
        self.queue_tree.tag_configure("running", foreground="#0067B9")
        self._update_queue_summary()

    def _update_queue_summary(self) -> None:
        total = len(self._series_tasks)
        pending = sum(task.state != "completed" for task in self._series_tasks)
        self.queue_summary_var.set(
            "队列为空，请先扫描总文件夹"
            if total == 0
            else f"共 {total} 部 · 待处理 {pending} 部"
        )

    @staticmethod
    def _state_label(state: str) -> str:
        return {
            "pending": "等待",
            "running": "处理中",
            "completed": "已完成",
            "partial": "部分完成",
            "failed": "失败",
        }.get(state, state)

    def _save_queue(self) -> None:
        with self._queue_lock:
            self._queue_store.save(list(self._series_tasks))

    def _load_last_scan_root(self) -> str:
        if not self._settings_path.is_file():
            return ""
        try:
            payload = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        value = payload.get("last_scan_root", "")
        return value if isinstance(value, str) else ""

    def _remember_scan_root(self, root: Path) -> None:
        self._last_scan_root = str(root.resolve())
        write_text_atomically(
            self._settings_path,
            json.dumps({"last_scan_root": self._last_scan_root}, ensure_ascii=False),
        )

    def _scan_series_library(self) -> None:
        initial = self._last_scan_root if Path(self._last_scan_root).is_dir() else ""
        folder = filedialog.askdirectory(
            title="选择包含多部剧的总文件夹", initialdir=initial or None
        )
        if not folder:
            return
        root = Path(folder)
        self._remember_scan_root(root)

        progress = tk.Toplevel(self)
        progress.title("正在扫描")
        progress.geometry("420x150")
        progress.resizable(False, False)
        progress.transient(self)
        progress.grab_set()
        body = ttk.Frame(progress, padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="正在查找剧集文件夹…", font=(UI_FONT, 12, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            body,
            text="会自动排除清水版、SRT 和临时处理目录。",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(6, 14))
        bar = ttk.Progressbar(body, mode="indeterminate")
        bar.pack(fill="x")
        bar.start(12)

        def finish(sources: list[Path] | None, error: str = "") -> None:
            if progress.winfo_exists():
                progress.destroy()
            if error:
                messagebox.showerror("扫描失败", error)
                return
            self._show_scan_preview(root, sources or [])

        def work() -> None:
            try:
                sources = discover_series_sources(root)
            except Exception as exc:
                self.after(0, finish, None, str(exc))
            else:
                self.after(0, finish, sources)

        threading.Thread(target=work, daemon=True).start()

    def _show_scan_preview(self, root: Path, sources: list[Path]) -> None:
        if not sources:
            messagebox.showinfo(
                "未发现剧集",
                "没有找到包含视频的“成片”文件夹。\n\n"
                "请确认结构类似：总文件夹\\剧名\\成片\\EP1.mp4",
            )
            return

        existing = {
            str(Path(task.source).resolve()).casefold() for task in self._series_tasks
        }
        window = tk.Toplevel(self)
        window.title("确认批量加入")
        window.geometry("900x560")
        window.minsize(760, 440)
        window.transient(self)
        window.grab_set()
        frame = ttk.Frame(window, padding=20)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text=f"扫描完成：发现 {len(sources)} 部剧",
            font=(UI_FONT, 16, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text=f"总目录：{root}。请选择要加入任务队列的项目。",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 12))

        table = ttk.Frame(frame)
        table.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(table, orient="vertical")
        tree = ttk.Treeview(
            table,
            columns=("series", "episodes", "state", "source"),
            show="headings",
            selectmode="extended",
            yscrollcommand=scroll.set,
        )
        scroll.configure(command=tree.yview)
        for column, title, width in (
            ("series", "剧名", 180),
            ("episodes", "集数", 70),
            ("state", "队列状态", 100),
            ("source", "成片文件夹", 480),
        ):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w")
        source_by_row: dict[str, Path] = {}
        for source in sources:
            key = str(source.resolve()).casefold()
            row = tree.insert(
                "",
                "end",
                values=(
                    task_from_source(source).name,
                    len(discover_videos(source)),
                    "已在队列" if key in existing else "待加入",
                    str(source),
                ),
                tags=("existing",) if key in existing else (),
            )
            source_by_row[row] = source
        tree.tag_configure("existing", foreground="#6E6E73")
        scroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        tree.selection_set(
            *(
                row
                for row, source in source_by_row.items()
                if str(source).casefold() not in existing
            )
        )

        actions = ttk.Frame(frame)
        actions.pack(fill="x", pady=(12, 0))

        def add_selected() -> None:
            selected = [source_by_row[row] for row in tree.selection()]
            added: list[SeriesTask] = []
            skipped = 0
            current = {
                str(Path(task.source).resolve()).casefold() for task in self._series_tasks
            }
            for source in selected:
                key = str(source.resolve()).casefold()
                if key in current:
                    skipped += 1
                    continue
                task = task_from_source(source)
                self._series_tasks.append(task)
                added.append(task)
                current.add(key)
            self._save_queue()
            self._render_queue()
            window.destroy()
            running = bool(self._worker and self._worker.is_alive())
            if added and not running:
                row = self._queue_rows[added[0].task_id]
                self.queue_tree.selection_set(row)
                self.queue_tree.see(row)
                self._load_series_into_form(added[0])
            running_note = (
                "；当前任务结束前加入的项目会接着处理"
                if running
                else ""
            )
            self.summary_var.set(
                f"已加入 {len(added)} 部，跳过重复 {skipped} 部{running_note}"
            )

        ttk.Button(
            actions,
            text="加入所选项目",
            style="Primary.TButton",
            command=add_selected,
        ).pack(side="right")
        ttk.Button(actions, text="取消", command=window.destroy).pack(
            side="right", padx=(0, 8)
        )
        ttk.Button(
            actions,
            text="全选",
            command=lambda: tree.selection_set(*tree.get_children()),
        ).pack(side="left")
        ttk.Button(
            actions,
            text="清除选择",
            command=lambda: tree.selection_remove(*tree.selection()),
        ).pack(
            side="left", padx=8
        )

    def _add_series(self) -> None:
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
            if existing.state == "completed":
                existing.state = "pending"
                existing.message = "已重新加入队列"
            task = existing
        else:
            task = task_from_source(source)
            self._series_tasks.append(task)
        self._save_queue()
        self._render_queue()
        if self._worker and self._worker.is_alive():
            self.summary_var.set(f"《{task.name}》已加入队列，将接着处理")
        else:
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
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(
            row=row, column=0, sticky="w", pady=6
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=12, pady=6
        )
        ttk.Button(parent, text="选择…", command=command).grid(row=row, column=2, pady=6)

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
        return app_data_dir() / "libtv-profile"

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

        ephemeral = not self._series_tasks
        if self._series_tasks:
            tasks = self._series_tasks
            if not any(task.state != "completed" for task in tasks):
                messagebox.showinfo("没有待处理任务", "队列中的项目均已完成。")
                return
        else:
            source = Path(self.source_var.get())
            if not source.is_dir() or not discover_videos(source):
                messagebox.showerror("无法开始", "请添加剧集到队列，或选择有效的成片文件夹。")
                return
            task = task_from_source(source)
            task.clean = self.clean_var.get().strip() or task.clean
            task.srt = self.srt_var.get().strip() or task.srt
            task.project_url = self.project_var.get().strip()
            tasks = [task]

        disk_issue = self._disk_space_issue(tasks)
        if disk_issue:
            messagebox.showerror("磁盘空间不足", disk_issue)
            return

        batch_size = self.batch_var.get()
        skip_existing = self.skip_var.get()
        srt_enabled = self.srt_enabled_var.get()
        asr_device = self.asr_device_var.get()
        self._stop_event.clear()
        self.start_button.configure(state="disabled", text="正在处理…")
        self.stop_button.configure(state="normal")

        def work() -> None:
            completed_count = 0
            failed_count = 0
            partial_count = 0
            authorization_blocked = False
            for task in tasks:
                if self._stop_event.is_set():
                    break
                if task.state == "completed":
                    continue
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
                        try:
                            task.project_url = LibTvWebBatchRunner.create_project_url(
                                task.name, workspace_id=DEFAULT_WORKSPACE_ID
                            )
                        except LibTvAuthenticationError:
                            task.message = "登录授权失效，正在自动恢复"
                            self._queue_changed(task, ephemeral)
                            try:
                                self._make_runner(
                                    project_url=LOGIN_FALLBACK_URL
                                ).ensure_cli_authorized()
                                task.project_url = (
                                    LibTvWebBatchRunner.create_project_url(
                                        task.name, workspace_id=DEFAULT_WORKSPACE_ID
                                    )
                                )
                            except Exception as auth_exc:
                                raise LibTvAuthenticationError(
                                    "LibTV 登录已失效。请点击“登录 LibTV”完成授权；"
                                    "登录后再次点击开始，任务会从当前剧继续。"
                                ) from auth_exc
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
                                series_callback=lambda text: self._set_series_message(
                                    active_task, text, ephemeral
                                ),
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
                except LibTvAuthenticationError as exc:
                    task.state = "pending"
                    task.message = str(exc)
                    authorization_blocked = True
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

                if authorization_blocked:
                    break

            if self._stop_event.is_set():
                summary = "队列已安全停止；再次开始会从记录继续。"
            elif authorization_blocked:
                summary = (
                    "队列已暂停：LibTV 登录授权失效。请点击“登录 LibTV”完成授权，"
                    "然后再次点击开始；未完成剧集仍保留在队列中。"
                )
            else:
                summary = (
                    f"多剧队列处理结束：完成 {completed_count} 部，"
                    f"部分完成 {partial_count} 部，失败 {failed_count} 部。"
                )
            self._events.put(("done", summary))
            self._events.put(("idle", None))

        self._worker = threading.Thread(target=work, daemon=True)
        self._worker.start()

    @staticmethod
    def _disk_space_issue(tasks: list[SeriesTask]) -> str:
        usage_by_volume: dict[str, tuple[Path, int]] = {}
        for task in tasks:
            if task.state == "completed":
                continue
            source = Path(task.source)
            if not source.is_dir():
                continue
            media_bytes = sum(
                video.stat().st_size
                for video in discover_videos(source)
                if video.is_file()
            )
            destination = Path(task.clean)
            probe = destination
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            volume = destination.anchor.casefold()
            current_probe, current_bytes = usage_by_volume.get(
                volume, (probe, 0)
            )
            usage_by_volume[volume] = (current_probe, current_bytes + media_bytes)

        for probe, media_bytes in usage_by_volume.values():
            required = int(media_bytes * 1.35) + 1024**3
            free = shutil.disk_usage(probe).free
            if free < required:
                return (
                    f"输出盘剩余 {free / 1024**3:.1f} GB，预计至少需要 "
                    f"{required / 1024**3:.1f} GB。\n\n"
                    "临时分段保存在清水版目录，不会占用 C 盘；请清理输出盘"
                    "或减少本次队列后再开始。"
                )
        return ""

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

    def _set_series_message(
        self, task: SeriesTask, message: str, ephemeral: bool
    ) -> None:
        task.message = message
        self._queue_changed(task, ephemeral)

    def _generate_srt(
        self,
        manifest: BatchManifest,
        output_dir: Path,
        callback,
        *,
        device: str,
        series_callback=lambda _text: None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        active_device = "cuda" if device == "auto" else device
        pending = []
        for job in manifest.jobs.values():
            destination = output_dir / f"{episode_stem(job.source_path)}.srt"
            job.srt_output = str(destination)
            if not destination.exists():
                pending.append((job, destination))
        if not pending:
            return
        try:
            self._run_srt_process(
                pending,
                active_device,
                callback,
                series_callback,
                idle_timeout=600 if active_device == "cuda" else 3600,
            )
        except CleanCutError as gpu_error:
            if active_device != "cuda" or self._stop_event.is_set():
                raise
            remaining = [(job, path) for job, path in pending if not path.exists()]
            if remaining:
                series_callback("GPU 转写失败或超时，已自动切换 CPU")
                self._write_gpu_error(gpu_error, gpu_error)
                self._run_srt_process(
                    remaining,
                    "cpu",
                    callback,
                    series_callback,
                    idle_timeout=3600,
                )
        for job, destination in pending:
            if not destination.exists():
                raise CleanCutError(f"SRT 未生成：{destination.name}")
            job.message = "SRT 完成，等待清水版"
            if job.state == JobState.ERROR:
                job.state = JobState.PENDING
            manifest.save()
            callback(job)

    def _run_srt_process(
        self,
        pending,
        device: str,
        callback,
        series_callback,
        *,
        idle_timeout: int,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="clean-cut-asr-") as temporary:
            work_dir = Path(temporary)
            manifest_path = work_dir / "jobs.json"
            status_path = work_dir / "status.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": str(status_path),
                        "jobs": [
                            {"source": str(job.source_path), "destination": str(destination)}
                            for job, destination in pending
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            if getattr(sys, "frozen", False):
                command = [sys.executable, "--asr-worker", str(manifest_path), device]
            else:
                command = [
                    sys.executable,
                    "-m",
                    "clean_cut.desktop",
                    "--asr-worker",
                    str(manifest_path),
                    device,
                ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            last_status = ""
            deadline = time.monotonic() + idle_timeout
            while process.poll() is None:
                if self._stop_event.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise CleanCutError("用户已停止 SRT 任务。")
                if status_path.is_file():
                    try:
                        status_text = status_path.read_text(encoding="utf-8")
                        status = json.loads(status_text)
                    except (OSError, json.JSONDecodeError):
                        status = None
                    if status and status_text != last_status:
                        last_status = status_text
                        deadline = time.monotonic() + idle_timeout
                        index = int(status.get("index", 1))
                        total = int(status.get("total", len(pending)))
                        state = status.get("state")
                        job, _destination = pending[min(max(index - 1, 0), len(pending) - 1)]
                        if state == "running":
                            job.message = f"生成 SRT（{index}/{total}，{device.upper()}）"
                            series_callback(job.message)
                            callback(job)
                        elif state == "complete":
                            job.message = f"SRT 已完成（{index}/{total}）"
                            series_callback(job.message)
                            callback(job)
                if time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise CleanCutError(
                        f"{device.upper()} SRT 单集超过 {idle_timeout // 60} 分钟无进展"
                    )
                time.sleep(1)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                detail = stderr.strip() or stdout.strip()
                if status_path.is_file():
                    try:
                        detail = json.loads(status_path.read_text(encoding="utf-8")).get(
                            "error", detail
                        )
                    except (OSError, json.JSONDecodeError):
                        pass
                raise CleanCutError(f"{device.upper()} SRT 进程失败：{detail[:1000]}")

    @staticmethod
    def _write_gpu_error(first_error: Exception, retry_error: Exception) -> None:
        log_path = app_data_dir() / "gpu-error.log"
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
            self.environment_button.configure(text="运行环境 · 正常")
            return
        self.environment_button.configure(text=f"运行环境 · 缺少 {len(missing)} 项")
        self.summary_var.set(
            "运行环境缺少："
            + "、".join(item.name for item in missing)
            + "；开始前请打开运行环境完成安装"
        )

    def _open_dependency_center(self) -> None:
        window = tk.Toplevel(self)
        window.title("运行环境检测与一键安装")
        window.geometry("900x600")
        window.minsize(760, 520)
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
            height=7,
        )
        for column, title, width in (
            ("name", "组件", 190),
            ("required", "类型", 80),
            ("state", "状态", 90),
            ("detail", "说明", 410),
        ):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="w")
        status_var = tk.StringVar(value="检测完成")
        actions = ttk.Frame(frame)
        actions.pack(side="bottom", fill="x", pady=(8, 0))
        ttk.Label(frame, textvariable=status_var, style="Muted.TLabel").pack(
            side="bottom", anchor="w", pady=(8, 0)
        )
        tree.pack(fill="both", expand=True)

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
            text="一键安装缺失必需项",
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
                        values=(
                            task.name,
                            task.source,
                            self._state_label(task.state),
                            task.message,
                        ),
                        tags=(task.state,),
                    )
                self._update_queue_summary()
                if task.task_id == self._active_series_id:
                    self.project_var.set(task.project_url)
            elif kind == "error":
                self.summary_var.set(str(value))
                messagebox.showerror("任务停止", str(value))
            elif kind == "info":
                messagebox.showinfo("提示", str(value))
            elif kind == "done":
                self.summary_var.set(str(value))
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
                self.start_button.configure(state="normal", text="开始全部任务")
                self.stop_button.configure(state="disabled")
        self.after(150, self._drain_events)


def _asr_worker(manifest_path: Path, device: str) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    status_path = Path(payload["status"])
    jobs = payload["jobs"]
    try:
        transcriber = FasterWhisperTranscriber(
            model="turbo",
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )
        for index, item in enumerate(jobs, 1):
            write_text_atomically(
                status_path,
                json.dumps(
                    {
                        "state": "running",
                        "index": index,
                        "total": len(jobs),
                        "source": item["source"],
                    },
                    ensure_ascii=False,
                ),
            )
            destination = Path(item["destination"])
            if not destination.exists():
                transcriber.write_srt(
                    Path(item["source"]), destination, language="en"
                )
            write_text_atomically(
                status_path,
                json.dumps(
                    {
                        "state": "complete",
                        "index": index,
                        "total": len(jobs),
                        "source": item["source"],
                    },
                    ensure_ascii=False,
                ),
            )
    except Exception:
        write_text_atomically(
            status_path,
            json.dumps(
                {"state": "error", "error": traceback.format_exc()},
                ensure_ascii=False,
            ),
        )
        raise


def main() -> None:
    if "--asr-worker" in sys.argv:
        worker_index = sys.argv.index("--asr-worker")
        _asr_worker(Path(sys.argv[worker_index + 1]), sys.argv[worker_index + 2])
        return
    if "--gpu-smoke-test" in sys.argv:
        source_index = sys.argv.index("--gpu-smoke-test") + 1
        log_path = app_data_dir() / "gpu-test.log"
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
