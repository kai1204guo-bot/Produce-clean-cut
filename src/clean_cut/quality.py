from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from clean_cut.errors import CleanCutError, MediaProcessError
from clean_cut.masks import MaskRenderConfig
from clean_cut.media import probe_media
from clean_cut.subtitle_data import HardSubtitlePlan
from clean_cut.video_repair import _mask_for_frame


@dataclass(slots=True)
class QualityReport:
    schema_version: int
    source: str
    repaired: str
    reference_clean: str | None
    inspected_frames: int
    mask_pixel_count: int
    duration_delta_ms: int | None
    output_has_audio: bool
    masked_frame_delta: float | None
    masked_mae: float | None
    masked_psnr_db: float | None
    temporal_residual_mae: float | None
    residual_check_status: str = "not_run"
    residual_intervals: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_repair(
    source: Path,
    repaired: Path,
    plan: HardSubtitlePlan,
    *,
    reference_clean: Path | None = None,
    mask_config: MaskRenderConfig | None = None,
) -> QualityReport:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise CleanCutError("修复质量评测需要OpenCV与NumPy。") from exc

    source = source.resolve()
    repaired = repaired.resolve()
    reference_clean = reference_clean.resolve() if reference_clean else None
    for label, path in (("输入视频", source), ("修复视频", repaired)):
        if not path.is_file():
            raise MediaProcessError(f"{label}不存在：{path}")
    if reference_clean is not None and not reference_clean.is_file():
        raise MediaProcessError(f"干净参考视频不存在：{reference_clean}")
    if Path(plan.source).resolve() != source:
        raise MediaProcessError("硬字幕计划与质量评测输入视频路径不匹配。")

    source_info = probe_media(source)
    repaired_info = probe_media(repaired)
    duration_delta_ms = None
    if source_info.duration_seconds is not None and repaired_info.duration_seconds is not None:
        duration_delta_ms = round(
            abs(source_info.duration_seconds - repaired_info.duration_seconds) * 1000
        )
    warnings = []
    if duration_delta_ms is not None and duration_delta_ms > 100:
        warnings.append(f"输入与输出时长相差{duration_delta_ms}毫秒。")
    output_has_audio = any(stream.codec_type == "audio" for stream in repaired_info.streams)
    source_has_audio = any(stream.codec_type == "audio" for stream in source_info.streams)
    if source_has_audio and not output_has_audio:
        warnings.append("输入含音频，但输出未检测到音频轨道。")

    captures = [cv2.VideoCapture(str(source)), cv2.VideoCapture(str(repaired))]
    if reference_clean is not None:
        captures.append(cv2.VideoCapture(str(reference_clean)))
    if not all(capture.isOpened() for capture in captures):
        for capture in captures:
            capture.release()
        raise MediaProcessError("无法打开质量评测所需的视频。")
    fps = float(captures[0].get(cv2.CAP_PROP_FPS))
    width = int(captures[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(captures[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    mask_config = mask_config or MaskRenderConfig()

    mask_pixels = 0
    delta_sum = 0.0
    delta_pixels = 0
    mae_sum = 0.0
    squared_error_sum = 0.0
    temporal_sum = 0.0
    temporal_pixels = 0
    frame_index = 0
    previous_repaired = None
    previous_reference = None
    previous_mask = None
    try:
        while True:
            values = [capture.read() for capture in captures]
            if not all(ok for ok, _ in values):
                break
            source_frame, repaired_frame = values[0][1], values[1][1]
            reference_frame = values[2][1] if reference_clean is not None else None
            if source_frame.shape != repaired_frame.shape or (
                reference_frame is not None and source_frame.shape != reference_frame.shape
            ):
                raise MediaProcessError("参与评测的视频分辨率不一致。")
            rendered = _mask_for_frame(
                frame_index,
                fps,
                width,
                height,
                plan,
                mask_config,
            )
            mask = rendered.binary > 0 if rendered is not None else np.zeros((height, width), bool)
            count = int(mask.sum())
            if count:
                repaired_float = repaired_frame.astype(np.float32)
                if previous_repaired is not None:
                    union = mask | previous_mask
                    delta_sum += float(
                        np.abs(repaired_float - previous_repaired)[union].sum()
                    )
                    delta_pixels += int(union.sum()) * 3
                if reference_frame is not None:
                    reference_float = reference_frame.astype(np.float32)
                    error = repaired_float - reference_float
                    mae_sum += float(np.abs(error)[mask].sum())
                    squared_error_sum += float(np.square(error)[mask].sum())
                    if previous_reference is not None:
                        union = mask | previous_mask
                        temporal_error = (repaired_float - previous_repaired) - (
                            reference_float - previous_reference
                        )
                        temporal_sum += float(np.abs(temporal_error)[union].sum())
                        temporal_pixels += int(union.sum()) * 3
                    previous_reference = reference_float
                previous_repaired = repaired_float
                previous_mask = mask
                mask_pixels += count
            else:
                previous_repaired = None
                previous_reference = None
                previous_mask = None
            frame_index += 1
        if any(ok for ok, _ in values) and not all(ok for ok, _ in values):
            warnings.append("参与评测的视频帧数不一致，指标按最短视频计算。")
    finally:
        for capture in captures:
            capture.release()

    channel_samples = mask_pixels * 3
    masked_mae = mae_sum / channel_samples if reference_clean is not None and mask_pixels else None
    mse = (
        squared_error_sum / channel_samples
        if reference_clean is not None and mask_pixels
        else None
    )
    psnr = None
    if mse is not None:
        psnr = 100.0 if mse == 0 else 10 * math.log10((255.0**2) / mse)
    if frame_index == 0:
        warnings.append("没有读取到可评测的视频帧。")
    if mask_pixels == 0:
        warnings.append("计划覆盖范围内没有有效遮罩像素。")
    return QualityReport(
        schema_version=2,
        source=str(source),
        repaired=str(repaired),
        reference_clean=str(reference_clean) if reference_clean else None,
        inspected_frames=frame_index,
        mask_pixel_count=mask_pixels,
        duration_delta_ms=duration_delta_ms,
        output_has_audio=output_has_audio,
        masked_frame_delta=(delta_sum / delta_pixels if delta_pixels else None),
        masked_mae=masked_mae,
        masked_psnr_db=psnr,
        temporal_residual_mae=(
            temporal_sum / temporal_pixels
            if reference_clean is not None and temporal_pixels
            else None
        ),
        warnings=warnings,
    )
