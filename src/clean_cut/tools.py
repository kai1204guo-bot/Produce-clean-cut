from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from clean_cut.errors import MediaProcessError, ToolNotFoundError
from clean_cut.paths import app_data_dir


def locate_executable(name: str) -> str | None:
    located = shutil.which(name)
    if located:
        return located
    candidates: list[Path] = []
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    if local:
        candidates.append(local / "Microsoft" / "WinGet" / "Links" / f"{name}.exe")
        if name.casefold() in {"ffmpeg", "ffprobe"}:
            candidates.append(
                app_data_dir()
                / "tools"
                / "ffmpeg"
                / "bin"
                / f"{name}.exe"
            )
    if name.casefold() == "libtv":
        candidates.append(Path.home() / ".libtv" / "libtv.exe")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def require_tool(name: str) -> str:
    executable = locate_executable(name)
    if executable is None:
        raise ToolNotFoundError(
            f"未找到 {name}。请安装 FFmpeg，并确认 {name} 已加入 PATH。"
        )
    return executable


def run_command(
    arguments: Sequence[str],
    *,
    description: str,
    expected_output: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(arguments),
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode != 0:
        detail = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"进程退出码 {result.returncode}（未返回文字错误）"
        )
        raise MediaProcessError(f"{description}失败：{detail}")
    if expected_output is not None and not expected_output.is_file():
        raise MediaProcessError(f"{description}失败：未生成预期文件 {expected_output}")
    return result


def write_text_atomically(destination: Path, content: str) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
