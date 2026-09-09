from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from clean_cut.asr import _CUDA_DLL_NAMES, default_cuda_dll_dirs
from clean_cut.errors import CleanCutError
from clean_cut.tools import locate_executable

ProgressCallback = Callable[[str], None]
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True, slots=True)
class DependencyInfo:
    key: str
    name: str
    required: bool
    installed: bool
    detail: str
    installable: bool = True


def app_data_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return base / "ProduceCleanCut"


def _chrome_path() -> Path | None:
    roots = [
        Path(os.environ.get("PROGRAMFILES", "")),
        Path(os.environ.get("PROGRAMFILES(X86)", "")),
        Path(os.environ.get("LOCALAPPDATA", "")),
    ]
    for root in roots:
        candidate = root / "Google" / "Chrome" / "Application" / "chrome.exe"
        if candidate.is_file():
            return candidate
    return None


def _cuda_ready() -> bool:
    files = {
        path.name.casefold()
        for directory in default_cuda_dll_dirs()
        if directory.is_dir()
        for path in directory.iterdir()
        if path.is_file()
    }
    return all(name.casefold() in files for name in _CUDA_DLL_NAMES)


def detect_dependencies() -> list[DependencyInfo]:
    ffmpeg = locate_executable("ffmpeg")
    ffprobe = locate_executable("ffprobe")
    libtv = locate_executable("libtv")
    chrome = _chrome_path()
    whisper = importlib.util.find_spec("faster_whisper") is not None
    nvidia = locate_executable("nvidia-smi") is not None
    return [
        DependencyInfo(
            "ffmpeg",
            "FFmpeg / FFprobe",
            True,
            bool(ffmpeg and ffprobe),
            ffmpeg or "缺失；视频分析和封面恢复需要它",
        ),
        DependencyInfo(
            "libtv",
            "LibTV 官方 CLI",
            True,
            bool(libtv),
            libtv or "缺失；上传、查询和下载需要它",
        ),
        DependencyInfo(
            "chrome",
            "Google Chrome",
            True,
            bool(chrome),
            str(chrome) if chrome else "缺失；智能去字幕网页自动化需要它",
        ),
        DependencyInfo(
            "runtime",
            "程序与 SRT 运行库",
            True,
            whisper,
            "已随安装包提供" if whisper else "安装包不完整，请重新安装软件",
            installable=False,
        ),
        DependencyInfo(
            "cuda",
            "NVIDIA CUDA SRT 加速库",
            False,
            _cuda_ready(),
            (
                "已安装，可使用 GPU 生成 SRT"
                if _cuda_ready()
                else "可选；未安装时自动使用 CPU"
                + ("" if nvidia else "（未检测到 NVIDIA 显卡）")
            ),
            installable=nvidia,
        ),
    ]


def _run(arguments: list[str], description: str) -> None:
    result = subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=CREATE_NO_WINDOW,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or str(result.returncode)
        raise CleanCutError(f"{description}失败：{detail[:1000]}")


def _install_winget(package_id: str, name: str, progress: ProgressCallback) -> None:
    winget = locate_executable("winget")
    if not winget:
        raise CleanCutError("系统缺少 Windows 程序包管理器 winget，请先更新 App Installer。")
    progress(f"正在安装 {name}…")
    _run(
        [
            winget,
            "install",
            "--id",
            package_id,
            "--exact",
            "--silent",
            "--accept-package-agreements",
            "--accept-source-agreements",
            "--disable-interactivity",
        ],
        f"安装 {name}",
    )


def _install_libtv(progress: ProgressCallback) -> None:
    progress("正在读取 LibTV 官方最新版信息…")
    with urllib.request.urlopen(
        "https://api2.liblib.art/api/www/landing-activities/getById?id=240",
        timeout=60,
    ) as response:
        activity = json.load(response)
    links = json.loads(activity["data"]["linkUrl"])
    script_url = links["install"]["PowerShell"]
    with tempfile.TemporaryDirectory(prefix="clean-cut-libtv-") as temp:
        script = Path(temp) / "install-libtv-cli.ps1"
        progress("正在下载 LibTV 官方安装脚本…")
        urllib.request.urlretrieve(script_url, script)
        powershell = locate_executable("powershell") or locate_executable("pwsh")
        if not powershell:
            raise CleanCutError("未找到 PowerShell，无法运行 LibTV 官方安装程序。")
        progress("正在安装 LibTV CLI…")
        _run(
            [
                powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
            ],
            "安装 LibTV CLI",
        )


def _download_cuda_wheel(package: str, version: str, target: Path) -> None:
    api = f"https://pypi.org/pypi/{package}/{version}/json"
    with urllib.request.urlopen(api, timeout=60) as response:
        payload = json.load(response)
    wheels = [
        item
        for item in payload.get("urls", [])
        if item.get("packagetype") == "bdist_wheel"
        and str(item.get("filename", "")).endswith("win_amd64.whl")
    ]
    if not wheels:
        raise CleanCutError(f"{package} 没有适用于当前 Windows x64 的运行库。")
    wheel = wheels[0]
    with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as temporary:
        wheel_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(wheel["url"], wheel_path)
        with zipfile.ZipFile(wheel_path) as archive:
            for member in archive.infolist():
                if member.filename.startswith("nvidia/"):
                    archive.extract(member, target)
    finally:
        wheel_path.unlink(missing_ok=True)


def _install_cuda(progress: ProgressCallback) -> None:
    target = app_data_dir() / "cuda-runtime"
    target.mkdir(parents=True, exist_ok=True)
    packages = (
        ("nvidia-cuda-runtime-cu12", "12.8.90"),
        ("nvidia-cublas-cu12", "12.8.4.1"),
        ("nvidia-cudnn-cu12", "9.10.2.21"),
    )
    for index, (package, version) in enumerate(packages, 1):
        progress(f"正在下载 GPU 运行库 {index}/{len(packages)}：{package}…")
        _download_cuda_wheel(package, version, target)
    if not _cuda_ready():
        raise CleanCutError("GPU 运行库下载完成，但 DLL 完整性检查未通过。")


def install_dependency(key: str, progress: ProgressCallback = lambda _text: None) -> None:
    if key == "ffmpeg":
        _install_winget("Gyan.FFmpeg", "FFmpeg", progress)
    elif key == "chrome":
        _install_winget("Google.Chrome", "Google Chrome", progress)
    elif key == "libtv":
        _install_libtv(progress)
    elif key == "cuda":
        _install_cuda(progress)
    else:
        raise CleanCutError(f"不支持自动安装组件：{key}")
    progress("安装完成，正在重新检测…")
