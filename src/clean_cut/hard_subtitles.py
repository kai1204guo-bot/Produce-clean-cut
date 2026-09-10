from __future__ import annotations

import json
import tempfile
from pathlib import Path

from clean_cut.errors import MediaProcessError
from clean_cut.media import probe_media
from clean_cut.ocr import OcrBackend
from clean_cut.srt import render_srt
from clean_cut.subtitle_data import HardSubtitlePlan, MaskKeyframe, OcrObservation, Region
from clean_cut.subtitle_tracking import (
    TrackingConfig,
    build_subtitle_cues,
    merge_frame_observations,
)
from clean_cut.tools import require_tool, run_command, write_text_atomically


def _video_dimensions(source: Path) -> tuple[int, int]:
    media = probe_media(source)
    video = next((stream for stream in media.streams if stream.codec_type == "video"), None)
    if video is None or video.width is None or video.height is None:
        raise MediaProcessError("无法确定输入视频的画面尺寸。")
    return video.width, video.height


def _validate_region(region: Region, width: int, height: int) -> None:
    if region.x + region.width > width or region.y + region.height > height:
        raise MediaProcessError(
            f"字幕区域超出视频范围：区域={region}，视频={width}x{height}。"
        )


def extract_sample_frames(
    source: Path,
    destination: Path,
    *,
    interval_ms: int,
) -> list[Path]:
    if interval_ms < 40 or interval_ms > 10_000:
        raise ValueError("抽帧间隔必须在40到10000毫秒之间。")
    destination.mkdir(parents=True, exist_ok=True)
    pattern = destination / "frame_%08d.png"
    ffmpeg = require_tool("ffmpeg")
    run_command(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            f"fps=1000/{interval_ms}",
            "-fps_mode",
            "passthrough",
            str(pattern),
        ],
        description="硬字幕抽帧",
    )
    frames = sorted(destination.glob("frame_*.png"))
    if not frames:
        raise MediaProcessError("硬字幕抽帧失败：没有生成任何画面。")
    return frames


def build_mask_keyframes(observations: list[OcrObservation]) -> list[MaskKeyframe]:
    grouped: dict[tuple[int, int], list[OcrObservation]] = {}
    for item in observations:
        grouped.setdefault((item.frame_index, item.timestamp_ms), []).append(item)
    return [
        MaskKeyframe(
            frame_index=key[0],
            timestamp_ms=key[1],
            polygons=tuple(item.polygon for item in items),
        )
        for key, items in sorted(grouped.items(), key=lambda item: item[0][1])
    ]


def analyze_hard_subtitles(
    source: Path,
    output_dir: Path,
    *,
    region: Region,
    backend: OcrBackend,
    interval_ms: int = 250,
    tracking_config: TrackingConfig | None = None,
) -> HardSubtitlePlan:
    source = source.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height = _video_dimensions(source)
    _validate_region(region, width, height)

    stem = source.stem.rstrip(" .") or "video"
    plan_path = output_dir / f"{stem}_hard_subtitle_plan.json"
    srt_path = output_dir / f"{stem}.srt"
    existing = [path for path in (plan_path, srt_path) if path.exists()]
    if existing:
        raise MediaProcessError(
            "输出文件已存在，未执行覆盖：" + "、".join(str(path) for path in existing)
        )

    observations: list[OcrObservation] = []
    with tempfile.TemporaryDirectory(prefix="clean-cut-frames-") as temp:
        frames = extract_sample_frames(source, Path(temp), interval_ms=interval_ms)
        for sample_index, frame in enumerate(frames):
            observations.extend(
                backend.recognize(
                    frame,
                    frame_index=sample_index,
                    timestamp_ms=sample_index * interval_ms,
                    duration_ms=interval_ms,
                    region=region,
                )
            )

    cue_observations = merge_frame_observations(observations)
    cues = build_subtitle_cues(cue_observations, tracking_config)
    plan = HardSubtitlePlan(
        schema_version=1,
        source=str(source),
        video_width=width,
        video_height=height,
        sampling_interval_ms=interval_ms,
        ocr_backend=type(backend).__name__,
        region=region,
        cues=cues,
        mask_keyframes=build_mask_keyframes(observations),
    )
    write_text_atomically(
        plan_path,
        json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
    )
    write_text_atomically(srt_path, render_srt(cues))
    return plan
