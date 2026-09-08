from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class SubtitleKind(StrEnum):
    TEXT = "text"
    BITMAP = "bitmap"
    UNKNOWN = "unknown"


class ProcessingRoute(StrEnum):
    TEXT_SOFT_SUBTITLE = "text_soft_subtitle"
    BITMAP_SOFT_SUBTITLE = "bitmap_soft_subtitle"
    POSSIBLE_HARD_SUBTITLE = "possible_hard_subtitle"


@dataclass(frozen=True, slots=True)
class MediaStream:
    index: int
    codec_type: str
    codec_name: str
    language: str | None = None
    title: str | None = None
    is_default: bool = False
    subtitle_kind: SubtitleKind | None = None
    width: int | None = None
    height: int | None = None
    avg_frame_rate: str | None = None
    time_base: str | None = None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    source: Path
    format_name: str
    duration_seconds: float | None
    size_bytes: int | None
    streams: tuple[MediaStream, ...]
    route: ProcessingRoute

    @property
    def subtitle_streams(self) -> tuple[MediaStream, ...]:
        return tuple(stream for stream in self.streams if stream.codec_type == "subtitle")

    @property
    def text_subtitle_streams(self) -> tuple[MediaStream, ...]:
        return tuple(
            stream
            for stream in self.subtitle_streams
            if stream.subtitle_kind == SubtitleKind.TEXT
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = str(self.source)
        return data


@dataclass(slots=True)
class ProcessingReport:
    source: Path
    route: ProcessingRoute
    status: str
    clean_video: Path | None = None
    subtitle_files: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "route": self.route.value,
            "status": self.status,
            "clean_video": str(self.clean_video) if self.clean_video else None,
            "subtitle_files": [str(path) for path in self.subtitle_files],
            "warnings": self.warnings,
        }

