from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from clean_cut.errors import MediaProcessError, ToolNotFoundError
from clean_cut.media import probe_media
from clean_cut.tools import require_tool, run_command

SUBTITLE_ERASER_MODEL = "volcano-subtitle-eraser"
SUBTITLE_ERASER_GENERATOR = "subtitle_erase"
SUPPORTED_VIDEO_EXTENSIONS = frozenset({".mp4", ".flv", ".ts", ".avi", ".mov", ".mkv", ".wmv"})


@dataclass(frozen=True, slots=True)
class LibTvRepairResult:
    source: Path
    output: Path
    project_uuid: str
    template_node: str
    uploaded_node: str
    task_id: str | None
    preserved_cover_frames: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source"] = str(self.source)
        data["output"] = str(self.output)
        return data


class LibTvClient:
    def __init__(self, executable: str | Path | None = None) -> None:
        self.executable = self._find_executable(executable)

    @staticmethod
    def _find_executable(executable: str | Path | None) -> str:
        if executable is not None:
            candidate = Path(executable).expanduser()
            if candidate.is_file():
                return str(candidate.resolve())
            located = shutil.which(str(executable))
            if located is not None:
                return located
            raise ToolNotFoundError(f"未找到 LibTV CLI：{executable}")

        located = shutil.which("libtv")
        if located is not None:
            return located
        windows_default = Path.home() / ".libtv" / "libtv.exe"
        if windows_default.is_file():
            return str(windows_default)
        raise ToolNotFoundError(
            "未找到 LibTV CLI。请安装官方 libtv CLI，并先执行 libtv login web。"
        )

    def _run(self, arguments: list[str], *, description: str) -> str:
        result = subprocess.run(
            [self.executable, *arguments],
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "未返回错误详情"
            raise MediaProcessError(f"{description}失败：{detail}")
        return result.stdout.strip()

    def _run_json(self, arguments: list[str], *, description: str) -> dict[str, Any]:
        output = self._run(arguments, description=description)
        try:
            payload = json.loads(output)
        except json.JSONDecodeError as exc:
            raise MediaProcessError(f"{description}失败：LibTV CLI 返回了无效 JSON。") from exc
        if not isinstance(payload, dict):
            raise MediaProcessError(f"{description}失败：LibTV CLI 返回值不是 JSON 对象。")
        return payload

    def get_node(self, project_uuid: str, node: str) -> dict[str, Any]:
        return self._run_json(
            ["node", node, "-p", project_uuid],
            description="读取 LibTV 节点",
        )

    def get_project(self, project_uuid: str) -> dict[str, Any]:
        return self._run_json(
            ["project", project_uuid],
            description="读取 LibTV 画布",
        )

    def validate_subtitle_template(self, project_uuid: str, node: str) -> str:
        payload = self.get_node(project_uuid, node)
        data = payload.get("data") or {}
        params = data.get("params") or {}
        if data.get("generatorType") != SUBTITLE_ERASER_GENERATOR:
            raise MediaProcessError(
                f"节点“{node}”不是智能去字幕模板：generatorType 应为 {SUBTITLE_ERASER_GENERATOR}。"
            )
        if params.get("model") != SUBTITLE_ERASER_MODEL:
            raise MediaProcessError(
                f"节点“{node}”不是智能去字幕模板：model 应为 {SUBTITLE_ERASER_MODEL}。"
            )
        node_key = payload.get("nodeKey")
        if not isinstance(node_key, str) or not node_key:
            raise MediaProcessError("智能去字幕模板缺少 nodeKey。")
        return node_key

    def upload_video(self, source: Path, project_uuid: str, name: str) -> str:
        payload = self._run_json(
            [
                "upload",
                name,
                "-f",
                str(source.resolve()),
                "-p",
                project_uuid,
                "-t",
                "video",
            ],
            description="上传视频到 LibTV",
        )
        node_key = payload.get("nodeKey")
        if not isinstance(node_key, str) or not node_key:
            raise MediaProcessError("上传视频失败：返回结果缺少 nodeKey。")
        return node_key

    def run_subtitle_template(
        self,
        project_uuid: str,
        template_node: str,
        input_node: str,
    ) -> dict[str, Any]:
        template_key = self.validate_subtitle_template(project_uuid, template_node)
        project = self.get_project(project_uuid)
        old_sources = [
            edge.get("source")
            for edge in project.get("edges", [])
            if edge.get("target") == template_key and isinstance(edge.get("source"), str)
        ]

        arguments = ["node", template_key, "-p", project_uuid]
        for source in old_sources:
            if source != input_node:
                arguments.extend(["--left-rm", source])
        if input_node not in old_sources:
            arguments.extend(["--left-add", input_node])
        arguments.extend(["-s", "mode=Subtitle", "--run"])
        return self._run_json(arguments, description="运行 LibTV 智能去字幕")

    def download_node(
        self,
        project_uuid: str,
        node: str,
        destination: Path,
    ) -> Path:
        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise MediaProcessError(f"输出文件已存在，未执行覆盖：{destination}")

        with tempfile.TemporaryDirectory(
            prefix=".libtv-download-", dir=destination.parent
        ) as temporary:
            temporary_path = Path(temporary)
            self._run(
                [
                    "download",
                    "-n",
                    node,
                    "-p",
                    project_uuid,
                    "-o",
                    str(temporary_path),
                    "--without-ai-watermark",
                    "--vip",
                ],
                description="下载 LibTV 清水视频",
            )
            files = [path for path in temporary_path.rglob("*") if path.is_file()]
            if len(files) != 1:
                raise MediaProcessError(
                    f"下载 LibTV 清水视频失败：预期一个文件，实际得到 {len(files)} 个。"
                )
            shutil.move(str(files[0]), destination)
        return destination


def detect_opening_cover_frames(
    source: Path,
    *,
    scan_frames: int = 15,
    scene_threshold: float = 0.3,
    fallback_frames: int = 2,
) -> int:
    """Detect a short opening cover ending at a strong cut in the first frames."""
    ffmpeg = require_tool("ffmpeg")
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source.resolve()),
            "-vf",
            "select='gte(scene,0)',metadata=print:file=-",
            "-frames:v",
            str(scan_frames),
            "-an",
            "-f",
            "null",
            os.devnull,
        ],
        capture_output=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "未返回错误详情"
        raise MediaProcessError(f"检测片头封面失败：{detail}")

    current_frame: int | None = None
    for line in result.stdout.splitlines():
        frame_match = re.search(r"\bframe:(\d+)", line)
        if frame_match:
            current_frame = int(frame_match.group(1))
            continue
        score_match = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
        if score_match and current_frame is not None:
            if current_frame > 0 and float(score_match.group(1)) >= scene_threshold:
                return current_frame
    return fallback_frames


def restore_opening_cover_frames(
    source: Path,
    repaired: Path,
    destination: Path,
    *,
    frame_count: int,
    crf: int = 16,
) -> Path:
    destination = destination.resolve()
    if frame_count < 0:
        raise ValueError("封面保护帧数不能小于 0。")
    if destination.exists():
        raise MediaProcessError(f"输出文件已存在，未执行覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if frame_count == 0:
        shutil.move(str(repaired.resolve()), destination)
        return destination

    ffmpeg = require_tool("ffmpeg")
    temporary = destination.with_name(f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}")
    filter_graph = (
        f"[1:v]trim=end_frame={frame_count},setpts=PTS-STARTPTS[cover];"
        "[0:v]setpts=PTS-STARTPTS[clean];"
        "[clean][cover]overlay=eof_action=pass:repeatlast=0[video]"
    )
    try:
        run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(repaired.resolve()),
                "-i",
                str(source.resolve()),
                "-filter_complex",
                filter_graph,
                "-map",
                "[video]",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            description="恢复片头封面",
            expected_output=temporary,
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def repair_with_libtv(
    source: Path,
    output: Path,
    *,
    project_uuid: str,
    template_node: str,
    executable: str | Path | None = None,
    cover_frames: int | None = None,
) -> LibTvRepairResult:
    source = source.resolve()
    output = output.resolve()
    media = probe_media(source)
    video = next((item for item in media.streams if item.codec_type == "video"), None)
    if video is None or video.width is None or video.height is None:
        raise MediaProcessError("无法确定 LibTV 输入视频的分辨率。")
    if max(video.width, video.height) > 2048:
        raise MediaProcessError("LibTV 智能去字幕暂不支持最长边超过 2K 的视频。")
    if media.duration_seconds is None or media.duration_seconds < 3:
        raise MediaProcessError("LibTV 智能去字幕要求视频时长不少于 3 秒。")
    if source.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        supported = " / ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise MediaProcessError(f"LibTV 智能去字幕不支持该格式；支持：{supported}")
    if output.exists():
        raise MediaProcessError(f"输出文件已存在，未执行覆盖：{output}")
    if cover_frames is not None and cover_frames < 0:
        raise ValueError("封面保护帧数不能小于 0。")
    preserved_cover_frames = (
        detect_opening_cover_frames(source) if cover_frames is None else cover_frames
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    client = LibTvClient(executable)
    template_key = client.validate_subtitle_template(project_uuid, template_node)
    upload_name = f"clean-cut-{source.stem}-{uuid4().hex[:8]}"
    uploaded_node = client.upload_video(source, project_uuid, upload_name)
    generated = client.run_subtitle_template(project_uuid, template_key, uploaded_node)
    data = generated.get("data") or {}
    task_info = data.get("taskInfo") or {}
    if task_info.get("status") != 2 or not data.get("url"):
        raise MediaProcessError("LibTV 智能去字幕未返回成功终态或下载地址。")
    with tempfile.TemporaryDirectory(prefix=".libtv-raw-", dir=output.parent) as temporary:
        raw_output = Path(temporary) / "libtv-clean.mp4"
        client.download_node(project_uuid, template_key, raw_output)
        restore_opening_cover_frames(
            source,
            raw_output,
            output,
            frame_count=preserved_cover_frames,
        )
    task_id = task_info.get("taskId")
    return LibTvRepairResult(
        source=source,
        output=output,
        project_uuid=project_uuid,
        template_node=template_key,
        uploaded_node=uploaded_node,
        task_id=str(task_id) if task_id is not None else None,
        preserved_cover_frames=preserved_cover_frames,
    )
