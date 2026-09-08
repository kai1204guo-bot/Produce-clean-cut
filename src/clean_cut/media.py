from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from clean_cut.errors import MediaProbeError
from clean_cut.models import (
    MediaInfo,
    MediaStream,
    ProcessingRoute,
    SubtitleKind,
)
from clean_cut.tools import require_tool

TEXT_SUBTITLE_CODECS = frozenset(
    {
        "ass",
        "eia_608",
        "eia_708",
        "jacosub",
        "microdvd",
        "mov_text",
        "mpl2",
        "realtext",
        "sami",
        "srt",
        "ssa",
        "subrip",
        "subviewer",
        "subviewer1",
        "text",
        "ttml",
        "webvtt",
    }
)

BITMAP_SUBTITLE_CODECS = frozenset(
    {
        "dvb_subtitle",
        "dvd_subtitle",
        "hdmv_pgs_subtitle",
        "xsub",
    }
)


def classify_subtitle_codec(codec_name: str) -> SubtitleKind:
    normalized = codec_name.lower()
    if normalized in TEXT_SUBTITLE_CODECS:
        return SubtitleKind.TEXT
    if normalized in BITMAP_SUBTITLE_CODECS:
        return SubtitleKind.BITMAP
    return SubtitleKind.UNKNOWN


def _optional_float(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "N/A", "") else None
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "N/A", "") else None
    except (TypeError, ValueError):
        return None


def _stream_from_probe(data: dict[str, Any]) -> MediaStream:
    tags = data.get("tags") or {}
    disposition = data.get("disposition") or {}
    codec_type = str(data.get("codec_type") or "unknown")
    codec_name = str(data.get("codec_name") or "unknown")
    subtitle_kind = (
        classify_subtitle_codec(codec_name) if codec_type == "subtitle" else None
    )
    return MediaStream(
        index=int(data["index"]),
        codec_type=codec_type,
        codec_name=codec_name,
        language=tags.get("language"),
        title=tags.get("title"),
        is_default=bool(disposition.get("default", 0)),
        subtitle_kind=subtitle_kind,
        width=_optional_int(data.get("width")),
        height=_optional_int(data.get("height")),
        r_frame_rate=data.get("r_frame_rate"),
        avg_frame_rate=data.get("avg_frame_rate"),
        time_base=data.get("time_base"),
    )


def _select_route(streams: tuple[MediaStream, ...]) -> ProcessingRoute:
    subtitles = tuple(stream for stream in streams if stream.codec_type == "subtitle")
    if any(stream.subtitle_kind == SubtitleKind.TEXT for stream in subtitles):
        return ProcessingRoute.TEXT_SOFT_SUBTITLE
    if subtitles:
        return ProcessingRoute.BITMAP_SOFT_SUBTITLE
    return ProcessingRoute.POSSIBLE_HARD_SUBTITLE


def parse_probe(source: Path, payload: dict[str, Any]) -> MediaInfo:
    format_data = payload.get("format") or {}
    streams = tuple(_stream_from_probe(stream) for stream in payload.get("streams", []))
    return MediaInfo(
        source=source,
        format_name=str(format_data.get("format_name") or "unknown"),
        duration_seconds=_optional_float(format_data.get("duration")),
        size_bytes=_optional_int(format_data.get("size")),
        streams=streams,
        route=_select_route(streams),
    )


def probe_media(source: Path) -> MediaInfo:
    source = source.resolve()
    if not source.is_file():
        raise MediaProbeError(f"输入视频不存在：{source}")

    ffprobe = require_tool("ffprobe")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(source),
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    if result.returncode != 0:
        raise MediaProbeError(f"无法读取媒体信息：{result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MediaProbeError("ffprobe 返回了无效的 JSON 数据。") from exc
    return parse_probe(source, payload)


def has_variable_frame_rate(stream: MediaStream, *, tolerance: float = 0.01) -> bool:
    """Use ffprobe's nominal and average rates as a conservative VFR signal."""
    if stream.codec_type != "video":
        return False
    try:
        nominal = float(Fraction(stream.r_frame_rate or "0/1"))
        average = float(Fraction(stream.avg_frame_rate or "0/1"))
    except (ValueError, ZeroDivisionError):
        return True
    if nominal <= 0 or average <= 0:
        return True
    return abs(nominal - average) / nominal > tolerance
