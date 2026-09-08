from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clean_cut.errors import CleanCutError, MediaProcessError


@dataclass(frozen=True, slots=True)
class SceneCut:
    frame_index: int
    timestamp_ms: int
    score: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class SceneDetector:
    def __init__(self, *, threshold: float = 0.6, min_interval_frames: int = 3) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("场景切换阈值必须在0到1之间。")
        if min_interval_frames < 1:
            raise ValueError("场景切换最小间隔至少为1帧。")
        self.threshold = threshold
        self.min_interval_frames = min_interval_frames
        self._previous_gray: Any | None = None
        self._previous_histogram: Any | None = None
        self._last_cut_frame = -min_interval_frames

    def update(self, frame: Any, frame_index: int, fps: float) -> SceneCut | None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise CleanCutError("场景切换检测需要OpenCV与NumPy。") from exc
        resized = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [64], [0, 256])
        cv2.normalize(histogram, histogram)
        if self._previous_gray is None:
            self._previous_gray = gray
            self._previous_histogram = histogram
            return None
        histogram_distance = float(
            cv2.compareHist(
                self._previous_histogram,
                histogram,
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        pixel_delta = float(
            np.mean(cv2.absdiff(self._previous_gray, gray), dtype=np.float64) / 255.0
        )
        score = min(1.0, max(histogram_distance, pixel_delta * 2.0))
        self._previous_gray = gray
        self._previous_histogram = histogram
        if (
            score >= self.threshold
            and frame_index - self._last_cut_frame >= self.min_interval_frames
        ):
            self._last_cut_frame = frame_index
            return SceneCut(frame_index, round(frame_index * 1000 / fps), score)
        return None


def detect_scene_cuts(
    source: Path,
    *,
    threshold: float = 0.6,
    min_interval_frames: int = 3,
) -> list[SceneCut]:
    try:
        import cv2
    except ImportError as exc:
        raise CleanCutError("场景切换检测需要OpenCV。") from exc
    source = source.resolve()
    if not source.is_file():
        raise MediaProcessError(f"输入视频不存在：{source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise MediaProcessError(f"无法打开输入视频：{source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise MediaProcessError("无法确定视频帧率。")
    detector = SceneDetector(threshold=threshold, min_interval_frames=min_interval_frames)
    cuts = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            cut = detector.update(frame, frame_index, fps)
            if cut is not None:
                cuts.append(cut)
            frame_index += 1
    finally:
        capture.release()
    return cuts
