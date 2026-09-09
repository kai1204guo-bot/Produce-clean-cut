from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from clean_cut.tools import write_text_atomically

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".ts", ".wmv"}


def episode_number(path: Path) -> int | None:
    match = re.search(r"(?i)(?:^|[^a-z])EP\s*0*(\d+)", path.stem)
    if match:
        return int(match.group(1))
    match = re.search(r"(?<!\d)(\d+)(?!\d)", path.stem)
    return int(match.group(1)) if match else None


def discover_videos(folder: Path) -> list[Path]:
    videos = [
        path.resolve()
        for path in folder.resolve().iterdir()
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    ]
    return sorted(
        videos,
        key=lambda path: (
            episode_number(path) is None,
            episode_number(path) or 0,
            path.name.casefold(),
        ),
    )


def chunked(items: Iterable[Path], size: int) -> Iterator[list[Path]]:
    if size < 1:
        raise ValueError("批次大小必须大于 0。")
    batch: list[Path] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def episode_stem(path: Path) -> str:
    if path.stem.strip().isdigit():
        return path.stem.strip()
    number = episode_number(path)
    return f"EP{number}" if number is not None else path.stem


class JobState(StrEnum):
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    GENERATING = "generating"
    DOWNLOADING = "downloading"
    RESTORING_COVER = "restoring_cover"
    COMPLETE = "complete"
    SKIPPED = "skipped"
    ERROR = "error"
    NEEDS_REVIEW = "needs_review"


@dataclass(slots=True)
class BatchJob:
    source: str
    episode: int | None
    state: JobState = JobState.PENDING
    progress: int = 0
    message: str = "等待处理"
    clean_output: str | None = None
    srt_output: str | None = None

    @property
    def source_path(self) -> Path:
        return Path(self.source)

    @property
    def key(self) -> str:
        return str(self.source_path.resolve()).casefold()


class BatchManifest:
    def __init__(self, path: Path, jobs: Iterable[BatchJob]) -> None:
        self.path = path.resolve()
        self.jobs = {job.key: job for job in jobs}
        self._lock = threading.RLock()

    @classmethod
    def load_or_create(cls, path: Path, videos: Iterable[Path]) -> BatchManifest:
        existing: dict[str, dict[str, object]] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            existing = {
                str(item["source"]).casefold(): item
                for item in payload.get("jobs", [])
                if isinstance(item, dict) and "source" in item
            }

        jobs: list[BatchJob] = []
        for video in videos:
            key = str(video.resolve()).casefold()
            item = existing.get(key)
            if item:
                try:
                    item["state"] = JobState(str(item.get("state", JobState.PENDING)))
                    jobs.append(BatchJob(**item))
                    continue
                except (TypeError, ValueError):
                    pass
            jobs.append(BatchJob(source=str(video.resolve()), episode=episode_number(video)))
        manifest = cls(path, jobs)
        manifest.save()
        return manifest

    def update(
        self,
        job: BatchJob,
        state: JobState,
        message: str,
        progress: int | None = None,
    ) -> None:
        with self._lock:
            job.state = state
            job.message = message
            if progress is not None:
                job.progress = max(0, min(100, progress))
            self.save()

    def save(self) -> None:
        with self._lock:
            payload = {
                "version": 1,
                "jobs": [
                    {**asdict(job), "state": job.state.value}
                    for job in sorted(
                        self.jobs.values(),
                        key=lambda item: (item.episode is None, item.episode or 0, item.source),
                    )
                ],
            }
            write_text_atomically(self.path, json.dumps(payload, ensure_ascii=False, indent=2))
