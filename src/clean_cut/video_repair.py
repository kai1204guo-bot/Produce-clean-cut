from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from clean_cut.errors import CleanCutError, MediaProcessError
from clean_cut.inpaint import InpaintBackend, SequenceInpaintBackend, composite_repair
from clean_cut.masks import MaskRenderConfig, RenderedMask, polygons_for_timestamp, render_mask
from clean_cut.media import has_variable_frame_rate, probe_media
from clean_cut.subtitle_data import HardSubtitlePlan
from clean_cut.tools import require_tool, run_command


def load_hard_subtitle_plan(path: Path) -> HardSubtitlePlan:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return HardSubtitlePlan.from_dict(data)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CleanCutError(f"无法读取硬字幕计划：{path}") from exc


def repair_video(
    source: Path,
    destination: Path,
    plan: HardSubtitlePlan,
    backend: InpaintBackend | SequenceInpaintBackend,
    *,
    mask_config: MaskRenderConfig | None = None,
    crf: int = 18,
    temporal_chunk_frames: int = 30,
    temporal_overlap_frames: int = 5,
) -> Path:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise CleanCutError("视频修复需要OpenCV与NumPy。") from exc

    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise MediaProcessError(f"输入视频不存在：{source}")
    if destination.exists():
        raise MediaProcessError(f"输出文件已存在，未执行覆盖：{destination}")
    if Path(plan.source).resolve() != source:
        raise MediaProcessError("硬字幕计划与输入视频路径不匹配。")
    if not 0 <= crf <= 51:
        raise ValueError("CRF必须在0到51之间。")
    if temporal_chunk_frames <= temporal_overlap_frames or temporal_overlap_frames < 0:
        raise ValueError("时序分块帧数必须大于非负的重叠帧数。")

    media_info = probe_media(source)
    video_stream = next(
        (stream for stream in media_info.streams if stream.codec_type == "video"),
        None,
    )
    if video_stream is None:
        raise MediaProcessError("输入文件不包含视频轨道。")
    if has_variable_frame_rate(video_stream):
        raise MediaProcessError("检测到可变帧率视频；阶段0为避免字幕时间漂移，暂不执行修复。")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise MediaProcessError(f"无法打开输入视频：{source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width != plan.video_width or height != plan.video_height:
        capture.release()
        raise MediaProcessError("硬字幕计划的画面尺寸与输入视频不匹配。")
    if fps <= 0:
        capture.release()
        raise MediaProcessError("无法确定视频帧率，阶段0仅支持恒定帧率修复。")

    mask_config = mask_config or MaskRenderConfig()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    ffmpeg = require_tool("ffmpeg")
    try:
        with tempfile.TemporaryDirectory(prefix="clean-cut-repair-") as temp:
            intermediate = Path(temp) / "repaired.mkv"
            writer = cv2.VideoWriter(
                str(intermediate),
                cv2.VideoWriter_fourcc(*"FFV1"),
                fps,
                (width, height),
            )
            if not writer.isOpened():
                raise MediaProcessError("无法创建FFV1无损中间视频。")
            frame_index = 0
            try:
                if isinstance(backend, SequenceInpaintBackend):
                    buffered_frames: list[object] = []
                    buffered_masks: list[RenderedMask | None] = []

                    def flush_temporal(*, final: bool) -> None:
                        if not buffered_frames:
                            return
                        binary_masks = [
                            item.binary if item is not None else np.zeros((height, width), np.uint8)
                            for item in buffered_masks
                        ]
                        if any(np.any(item) for item in binary_masks):
                            repaired_frames = backend.inpaint_sequence(
                                buffered_frames,
                                binary_masks,
                            )
                        else:
                            repaired_frames = buffered_frames
                        emit_count = (
                            len(buffered_frames)
                            if final
                            else len(buffered_frames) - temporal_overlap_frames
                        )
                        for item_index in range(emit_count):
                            frame = buffered_frames[item_index]
                            mask = buffered_masks[item_index]
                            if mask is not None:
                                frame = composite_repair(
                                    frame,
                                    repaired_frames[item_index],
                                    mask,
                                )
                            writer.write(frame)
                        del buffered_frames[:emit_count]
                        del buffered_masks[:emit_count]

                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        buffered_frames.append(frame)
                        buffered_masks.append(
                            _mask_for_frame(
                                frame_index,
                                fps,
                                width,
                                height,
                                plan,
                                mask_config,
                            )
                        )
                        frame_index += 1
                        if len(buffered_frames) >= temporal_chunk_frames:
                            flush_temporal(final=False)
                    flush_temporal(final=True)
                else:
                    while True:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        mask = _mask_for_frame(
                            frame_index,
                            fps,
                            width,
                            height,
                            plan,
                            mask_config,
                        )
                        if mask is not None:
                            repaired = backend.inpaint(frame, mask.binary)
                            frame = composite_repair(frame, repaired, mask)
                        writer.write(frame)
                        frame_index += 1
            finally:
                writer.release()
                capture.release()
            if frame_index == 0:
                raise MediaProcessError("输入视频没有可解码画面。")

            run_command(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(intermediate),
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a?",
                    "-map_metadata",
                    "1",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    str(crf),
                    "-c:a",
                    "copy",
                    "-shortest",
                    str(temporary_output),
                ],
                description="修复视频编码与音频回封",
                expected_output=temporary_output,
            )
            os.replace(temporary_output, destination)
    finally:
        capture.release()
        temporary_output.unlink(missing_ok=True)
    return destination


def _mask_for_frame(
    frame_index: int,
    fps: float,
    width: int,
    height: int,
    plan: HardSubtitlePlan,
    config: MaskRenderConfig,
) -> RenderedMask | None:
    import numpy as np

    timestamp_ms = round(frame_index * 1000 / fps)
    polygons = polygons_for_timestamp(
        plan.mask_keyframes,
        timestamp_ms,
        sampling_interval_ms=plan.sampling_interval_ms,
        config=config,
    )
    if not polygons:
        return None
    mask = render_mask(width, height, polygons, config)
    return mask if np.any(mask.binary) else None
