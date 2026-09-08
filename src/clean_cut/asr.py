from __future__ import annotations

import ctypes
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from clean_cut.errors import CleanCutError
from clean_cut.srt import render_srt
from clean_cut.subtitle_data import SubtitleCue
from clean_cut.tools import write_text_atomically

_CUDA_DLL_NAMES = (
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cudnn64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_adv64_9.dll",
)


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model: str = "turbo",
        device: str = "cuda",
        compute_type: str = "float16",
        model_dir: Path | None = None,
        cuda_dll_dir: Path | None = None,
    ) -> None:
        self._dll_handles: list[Any] = []
        if device == "cuda" and os.name == "nt" and cuda_dll_dir is not None:
            self._load_cuda_dlls(cuda_dll_dir)
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise CleanCutError(
                "缺少 faster-whisper。请安装项目的 asr 可选依赖后重试。"
            ) from exc
        try:
            self._model = WhisperModel(
                model,
                device=device,
                compute_type=compute_type,
                download_root=str(model_dir.resolve()) if model_dir else None,
            )
        except RuntimeError as exc:
            raise CleanCutError(f"无法加载 Faster-Whisper 模型：{exc}") from exc

    def _load_cuda_dlls(self, directory: Path) -> None:
        directory = directory.resolve()
        missing = [name for name in _CUDA_DLL_NAMES if not (directory / name).is_file()]
        if missing:
            raise CleanCutError(
                f"CUDA DLL 目录不完整：{directory}；缺少 {', '.join(missing)}"
            )
        try:
            self._dll_handles = [
                ctypes.WinDLL(str(directory / name))  # type: ignore[attr-defined]
                for name in _CUDA_DLL_NAMES
            ]
        except OSError as exc:
            raise CleanCutError(f"无法加载 CUDA DLL：{exc}") from exc

    def transcribe(self, source: Path, *, language: str = "en") -> list[SubtitleCue]:
        try:
            segments, _ = self._model.transcribe(
                str(source.resolve()),
                language=language,
                beam_size=5,
                vad_filter=False,
                word_timestamps=True,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            return segments_to_cues(segments)
        except RuntimeError as exc:
            raise CleanCutError(f"Faster-Whisper 转写失败：{exc}") from exc

    def write_srt(self, source: Path, destination: Path, *, language: str = "en") -> Path:
        destination = destination.resolve()
        if destination.exists():
            raise CleanCutError(f"输出文件已存在，未执行覆盖：{destination}")
        cues = self.transcribe(source, language=language)
        write_text_atomically(destination, render_srt(cues))
        return destination


def segments_to_cues(segments: Iterable[Any]) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for segment in segments:
        text = str(segment.text).strip()
        if not text:
            continue
        words = list(segment.words or [])
        start = (
            float(words[0].start)
            if words and words[0].start is not None
            else float(segment.start)
        )
        end = (
            float(words[-1].end)
            if words and words[-1].end is not None
            else float(segment.end)
        )
        if end <= start:
            end = float(segment.end)
        probabilities = [float(word.probability) for word in words if word.probability is not None]
        confidence = sum(probabilities) / len(probabilities) if probabilities else 0.0
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,
                start_ms=max(0, round(start * 1000)),
                end_ms=max(round(start * 1000) + 1, round(end * 1000)),
                text=text,
                confidence=round(confidence, 4),
                observation_count=max(1, len(words)),
            )
        )
    return cues
