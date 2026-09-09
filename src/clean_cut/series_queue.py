from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from clean_cut.tools import write_text_atomically


@dataclass(slots=True)
class SeriesTask:
    source: str
    clean: str
    srt: str
    name: str
    project_url: str = ""
    state: str = "pending"
    message: str = "等待处理"
    task_id: str = ""

    def __post_init__(self) -> None:
        if not self.task_id:
            self.task_id = uuid4().hex

    @property
    def source_path(self) -> Path:
        return Path(self.source)


class SeriesQueueStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def load(self) -> list[SeriesTask]:
        if not self.path.is_file():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        tasks: list[SeriesTask] = []
        for item in payload.get("tasks", []):
            if not isinstance(item, dict):
                continue
            try:
                tasks.append(SeriesTask(**item))
            except TypeError:
                continue
        return tasks

    def save(self, tasks: list[SeriesTask]) -> None:
        payload = {"version": 1, "tasks": [asdict(task) for task in tasks]}
        write_text_atomically(
            self.path, json.dumps(payload, ensure_ascii=False, indent=2)
        )


def series_name_from_source(source: Path) -> str:
    import re

    name = re.sub(r"^\s*成片[\s_-]*", "", source.parent.name, flags=re.IGNORECASE).strip()
    if name.casefold() in {"", "成片", "videos", "video"}:
        name = source.name.strip()
    return name


def task_from_source(source: Path) -> SeriesTask:
    source = source.resolve()
    base = source.parent
    clean = base / "清水版"
    project_file = clean / ".clean-cut-project-url.txt"
    project_url = (
        project_file.read_text(encoding="utf-8").strip()
        if project_file.is_file()
        else ""
    )
    return SeriesTask(
        source=str(source),
        clean=str(clean),
        srt=str(base / "srt"),
        name=series_name_from_source(source),
        project_url=project_url,
    )
