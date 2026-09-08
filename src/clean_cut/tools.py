from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from clean_cut.errors import MediaProcessError, ToolNotFoundError


def require_tool(name: str) -> str:
    executable = shutil.which(name)
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
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "未返回错误详情"
        raise MediaProcessError(f"{description}失败：{detail}")
    if expected_output is not None and not expected_output.is_file():
        raise MediaProcessError(f"{description}失败：未生成预期文件 {expected_output}")
    return result

