from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from clean_cut.errors import CleanCutError
from clean_cut.subtitle_data import OcrObservation, Point, Polygon, Region


class OcrBackend(ABC):
    @abstractmethod
    def recognize(
        self,
        image: Path,
        *,
        frame_index: int,
        timestamp_ms: int,
        duration_ms: int,
        region: Region | None = None,
    ) -> list[OcrObservation]:
        """Recognize text and return observations in full-frame coordinates."""


class RapidOcrBackend(OcrBackend):
    def __init__(self, *, text_score: float = 0.5) -> None:
        if not 0 <= text_score <= 1:
            raise ValueError("OCR最低置信度必须在0到1之间。")
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:
            raise CleanCutError(
                "RapidOCR未安装。请安装项目的ocr可选依赖后重试。"
            ) from exc
        self._engine = RapidOCR(params={"Global.text_score": text_score})

    def recognize(
        self,
        image: Path,
        *,
        frame_index: int,
        timestamp_ms: int,
        duration_ms: int,
        region: Region | None = None,
    ) -> list[OcrObservation]:
        try:
            import cv2
        except ImportError as exc:
            raise CleanCutError("RapidOCR图像运行时不完整：缺少OpenCV。") from exc

        frame = cv2.imread(str(image))
        if frame is None:
            raise CleanCutError(f"无法读取OCR图片：{image}")
        offset_x = region.x if region else 0
        offset_y = region.y if region else 0
        if region:
            frame_height, frame_width = frame.shape[:2]
            right = min(region.x + region.width, frame_width)
            bottom = min(region.y + region.height, frame_height)
            if region.x >= right or region.y >= bottom:
                raise CleanCutError("OCR区域位于图片范围之外。")
            frame = frame[region.y:bottom, region.x:right]

        result = self._engine(frame)
        boxes = getattr(result, "boxes", None)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        boxes = [] if boxes is None else boxes
        texts = [] if texts is None else texts
        scores = [] if scores is None else scores
        observations: list[OcrObservation] = []
        for box, text, score in zip(boxes, texts, scores, strict=False):
            points = tuple(
                Point(float(point[0]) + offset_x, float(point[1]) + offset_y)
                for point in _as_list(box)
            )
            observations.append(
                OcrObservation(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    duration_ms=duration_ms,
                    text=str(text),
                    confidence=float(score),
                    polygon=Polygon(points),
                )
            )
        return observations


def _as_list(value: Any) -> list[Any]:
    return value.tolist() if hasattr(value, "tolist") else list(value)
