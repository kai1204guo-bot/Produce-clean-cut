from __future__ import annotations

import hashlib
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from clean_cut.batch import BatchJob
from clean_cut.errors import MediaProcessError
from clean_cut.libtv import detect_opening_cover_frames
from clean_cut.media import probe_media
from clean_cut.tools import require_tool, run_command


@dataclass(frozen=True, slots=True)
class SegmentWindow:
    index: int
    count: int
    core_start: float
    core_end: float
    input_start: float
    input_end: float

    @property
    def core_duration(self) -> float:
        return self.core_end - self.core_start

    @property
    def trim_start(self) -> float:
        return self.core_start - self.input_start


def plan_equal_segments(
    duration: float,
    *,
    free_limit: float = 60.0,
    safe_limit: float = 58.0,
    overlap: float = 0.5,
) -> tuple[SegmentWindow, ...]:
    """Split long media evenly, keeping every submitted segment below the safe limit."""
    if duration <= 0:
        raise ValueError("视频时长必须大于 0。")
    if not 3 <= safe_limit < free_limit:
        raise ValueError("安全分段时长必须不少于 3 秒且小于免费时长。")
    if overlap < 0 or overlap * 2 >= safe_limit:
        raise ValueError("分段重叠时长无效。")
    if duration <= free_limit:
        return (
            SegmentWindow(
                index=1,
                count=1,
                core_start=0.0,
                core_end=duration,
                input_start=0.0,
                input_end=duration,
            ),
        )

    # The overlap is part of the uploaded media, so reserve it on both sides.
    max_core = safe_limit - overlap * 2
    count = max(2, math.ceil(duration / max_core))
    core_duration = duration / count
    windows: list[SegmentWindow] = []
    for offset in range(count):
        core_start = core_duration * offset
        core_end = duration if offset == count - 1 else core_duration * (offset + 1)
        input_start = max(0.0, core_start - (overlap if offset else 0.0))
        input_end = min(
            duration,
            core_end + (overlap if offset < count - 1 else 0.0),
        )
        windows.append(
            SegmentWindow(
                index=offset + 1,
                count=count,
                core_start=core_start,
                core_end=core_end,
                input_start=input_start,
                input_end=input_end,
            )
        )
    return tuple(windows)


def _letters_only_token(source: Path) -> str:
    stat = source.stat()
    identity = (
        f"{str(source.resolve()).casefold()}|{stat.st_size}|{stat.st_mtime_ns}"
    ).encode()
    digest = hashlib.sha1(identity).hexdigest()[:10]
    return digest.translate(str.maketrans("0123456789", "ghijklmnop"))


def _part_label(index: int) -> str:
    label = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("a") + remainder) + label
    return label


def split_for_libtv(
    source: Path,
    work_root: Path,
    *,
    free_limit: float = 60.0,
    safe_limit: float = 58.0,
    overlap: float = 0.5,
) -> list[BatchJob]:
    source = source.resolve()
    media = probe_media(source)
    if media.duration_seconds is None:
        raise MediaProcessError(f"无法读取视频时长：{source.name}")
    windows = plan_equal_segments(
        media.duration_seconds,
        free_limit=free_limit,
        safe_limit=safe_limit,
        overlap=overlap,
    )
    if len(windows) == 1:
        return [
            BatchJob(
                source=str(source),
                episode=None,
                parent_source=str(source),
                segment_index=1,
                segment_count=1,
            )
        ]

    ffmpeg = require_tool("ffmpeg")
    token = _letters_only_token(source)
    source_dir = work_root / token / "source"
    clean_dir = work_root / token / "clean"
    source_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[BatchJob] = []
    for window in windows:
        stem = f"seg{token}part{_part_label(window.index)}"
        segment = source_dir / f"{stem}.mp4"
        expected_duration = window.input_end - window.input_start
        reusable = False
        if segment.is_file() and segment.stat().st_size > 0:
            existing = probe_media(segment).duration_seconds
            reusable = existing is not None and abs(existing - expected_duration) <= 0.08
        if not reusable:
            temporary = segment.with_name(f".{segment.stem}.{uuid4().hex}.tmp.mp4")
            has_audio = any(item.codec_type == "audio" for item in media.streams)
            filter_graph = (
                f"[0:v:0]trim=start={window.input_start:.6f}:end={window.input_end:.6f},"
                "setpts=PTS-STARTPTS[v]"
            )
            if has_audio:
                filter_graph += (
                    f";[0:a:0]atrim=start={window.input_start:.6f}:"
                    f"end={window.input_end:.6f},asetpts=PTS-STARTPTS[a]"
                )
            arguments = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-filter_complex",
                filter_graph,
                "-map",
                "[v]",
            ]
            if has_audio:
                arguments.extend(["-map", "[a]"])
            arguments.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-fps_mode",
                    "vfr",
                ]
            )
            if has_audio:
                arguments.extend(["-c:a", "aac", "-b:a", "192k"])
            arguments.extend(["-movflags", "+faststart", str(temporary)])
            try:
                run_command(
                    arguments,
                    description=f"切分 {source.name} 第 {window.index}/{window.count} 段",
                    expected_output=temporary,
                )
                os.replace(temporary, segment)
            finally:
                temporary.unlink(missing_ok=True)
        jobs.append(
            BatchJob(
                source=str(segment),
                episode=None,
                clean_output=str(clean_dir / f"{stem}-清水版.mp4"),
                parent_source=str(source),
                segment_index=window.index,
                segment_count=window.count,
                segment_core_start=window.core_start,
                segment_core_end=window.core_end,
                segment_input_start=window.input_start,
                segment_input_end=window.input_end,
            )
        )
    return jobs


def merge_libtv_segments(
    source: Path,
    segments: list[BatchJob],
    destination: Path,
) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    ordered = sorted(segments, key=lambda item: item.segment_index or 0)
    if not ordered or any(not item.clean_output for item in ordered):
        raise MediaProcessError(f"{source.name} 缺少可合并的清水分段。")
    inputs = [Path(item.clean_output).resolve() for item in ordered]
    if any(not item.is_file() or item.stat().st_size == 0 for item in inputs):
        raise MediaProcessError(f"{source.name} 的清水分段尚未全部下载。")

    ffmpeg = require_tool("ffmpeg")
    media = probe_media(source)
    has_audio = any(item.codec_type == "audio" for item in media.streams)
    filters: list[str] = []
    concat_inputs: list[str] = []
    for offset, (job, path) in enumerate(zip(ordered, inputs, strict=True)):
        if None in {
            job.segment_core_start,
            job.segment_core_end,
            job.segment_input_start,
        }:
            raise MediaProcessError(f"{path.name} 缺少分段时间戳。")
        trim_start = float(job.segment_core_start) - float(job.segment_input_start)
        duration = float(job.segment_core_end) - float(job.segment_core_start)
        filters.append(
            f"[{offset}:v:0]trim=start={trim_start:.6f}:duration={duration:.6f},"
            f"setpts=PTS-STARTPTS[v{offset}]"
        )
        concat_inputs.append(f"[v{offset}]")
        if has_audio:
            filters.append(
                f"[{offset}:a:0]atrim=start={trim_start:.6f}:duration={duration:.6f},"
                f"asetpts=PTS-STARTPTS[a{offset}]"
            )
            concat_inputs.append(f"[a{offset}]")
    concat_outputs = "[clean][audio]" if has_audio else "[clean]"
    filters.append(
        "".join(concat_inputs)
        + f"concat=n={len(ordered)}:v=1:a={1 if has_audio else 0}{concat_outputs}"
    )
    cover_frames = detect_opening_cover_frames(source)
    final_video = "[clean]"
    if cover_frames:
        source_index = len(inputs)
        filters.append(
            f"[{source_index}:v:0]trim=end_frame={cover_frames},"
            "setpts=PTS-STARTPTS[cover]"
        )
        filters.append("[clean][cover]overlay=eof_action=pass:repeatlast=0[video]")
        final_video = "[video]"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    arguments = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    for path in inputs:
        arguments.extend(["-i", str(path)])
    arguments.extend(["-i", str(source), "-filter_complex", ";".join(filters)])
    arguments.extend(["-map", final_video])
    if has_audio:
        arguments.extend(["-map", "[audio]"])
    arguments.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "vfr",
        ]
    )
    if has_audio:
        arguments.extend(["-c:a", "aac", "-b:a", "192k"])
    arguments.extend(["-movflags", "+faststart", str(temporary)])
    try:
        run_command(
            arguments,
            description=f"合并 {source.name} 的清水分段",
            expected_output=temporary,
        )
        output_duration = probe_media(temporary).duration_seconds
        if output_duration is None or media.duration_seconds is None:
            raise MediaProcessError(f"无法校验 {source.name} 合并后的时长。")
        if abs(output_duration - media.duration_seconds) > 0.25:
            raise MediaProcessError(
                f"{source.name} 合并后时长异常：原片 {media.duration_seconds:.3f} 秒，"
                f"成片 {output_duration:.3f} 秒。"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def discard_segment_sources(jobs: list[BatchJob]) -> None:
    roots = {job.source_path.parent.parent for job in jobs if job.is_segment}
    for root in roots:
        shutil.rmtree(root, ignore_errors=True)
