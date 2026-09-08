from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from clean_cut.models import MediaInfo, MediaStream
from clean_cut.tools import require_tool, run_command


def choose_primary_text_subtitle(media: MediaInfo) -> MediaStream | None:
    streams = media.text_subtitle_streams
    if not streams:
        return None
    return next((stream for stream in streams if stream.is_default), streams[0])


def _temporary_output(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}")


def _replace_atomically(temporary: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, destination)


def extract_text_subtitle(source: Path, stream: MediaStream, destination: Path) -> Path:
    ffmpeg = require_tool("ffmpeg")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_output(destination)
    try:
        run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                f"0:{stream.index}",
                "-c:s",
                "srt",
                str(temporary),
            ],
            description="字幕提取",
            expected_output=temporary,
        )
        _replace_atomically(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def remux_without_subtitles(source: Path, destination: Path) -> Path:
    ffmpeg = require_tool("ffmpeg")
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_output(destination)
    try:
        run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v?",
                "-map",
                "0:a?",
                "-map",
                "0:d?",
                "-map_metadata",
                "0",
                "-map_chapters",
                "0",
                "-c",
                "copy",
                str(temporary),
            ],
            description="无字幕视频封装",
            expected_output=temporary,
        )
        _replace_atomically(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination

