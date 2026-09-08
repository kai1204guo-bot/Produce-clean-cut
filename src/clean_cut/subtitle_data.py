from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Region:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError("区域坐标不能为负数。")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("区域宽度和高度必须大于零。")


@dataclass(frozen=True, slots=True)
class Polygon:
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("多边形至少需要三个点。")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [point.x for point in self.points]
        ys = [point.y for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass(frozen=True, slots=True)
class OcrObservation:
    frame_index: int
    timestamp_ms: int
    duration_ms: int
    text: str
    confidence: float
    polygon: Polygon

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.timestamp_ms < 0:
            raise ValueError("帧序号和时间戳不能为负数。")
        if self.duration_ms <= 0:
            raise ValueError("帧持续时间必须大于零。")
        if not self.text.strip():
            raise ValueError("OCR文字不能为空。")
        if not 0 <= self.confidence <= 1:
            raise ValueError("OCR置信度必须在0到1之间。")


@dataclass(frozen=True, slots=True)
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    observation_count: int


@dataclass(frozen=True, slots=True)
class MaskKeyframe:
    frame_index: int
    timestamp_ms: int
    polygons: tuple[Polygon, ...]


@dataclass(slots=True)
class HardSubtitlePlan:
    schema_version: int
    source: str
    video_width: int
    video_height: int
    sampling_interval_ms: int
    ocr_backend: str
    region: Region
    cues: list[SubtitleCue] = field(default_factory=list)
    mask_keyframes: list[MaskKeyframe] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
