from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clean_cut.errors import CleanCutError
from clean_cut.subtitle_data import MaskKeyframe, Polygon


@dataclass(frozen=True, slots=True)
class MaskRenderConfig:
    dilation_px: int = 8
    feather_px: int = 4
    hold_before_ms: int = 120
    hold_after_ms: int = 120

    def __post_init__(self) -> None:
        if self.dilation_px < 0 or self.feather_px < 0:
            raise ValueError("遮罩扩张和羽化像素不能为负数。")
        if self.hold_before_ms < 0 or self.hold_after_ms < 0:
            raise ValueError("遮罩保持时间不能为负数。")


@dataclass(frozen=True, slots=True)
class RenderedMask:
    binary: Any
    alpha: Any


def polygons_for_timestamp(
    keyframes: list[MaskKeyframe],
    timestamp_ms: int,
    *,
    sampling_interval_ms: int,
    config: MaskRenderConfig,
) -> tuple[Polygon, ...]:
    if not keyframes:
        return ()
    before_limit = sampling_interval_ms // 2 + config.hold_before_ms
    after_limit = sampling_interval_ms // 2 + config.hold_after_ms
    candidates: list[tuple[int, MaskKeyframe]] = []
    for keyframe in keyframes:
        delta = timestamp_ms - keyframe.timestamp_ms
        if -before_limit <= delta <= after_limit:
            candidates.append((abs(delta), keyframe))
    if not candidates:
        return ()
    return min(candidates, key=lambda item: item[0])[1].polygons


def render_mask(
    width: int,
    height: int,
    polygons: tuple[Polygon, ...],
    config: MaskRenderConfig,
) -> RenderedMask:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise CleanCutError("遮罩渲染需要OpenCV与NumPy。") from exc

    binary = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.array(
            [[round(point.x), round(point.y)] for point in polygon.points],
            dtype=np.int32,
        )
        cv2.fillPoly(binary, [points], 255)
    if config.dilation_px:
        size = config.dilation_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        binary = cv2.dilate(binary, kernel, iterations=1)

    if config.feather_px:
        kernel_size = config.feather_px * 2 + 1
        alpha = cv2.GaussianBlur(binary, (kernel_size, kernel_size), 0)
    else:
        alpha = binary.copy()
    return RenderedMask(binary=binary, alpha=alpha)

