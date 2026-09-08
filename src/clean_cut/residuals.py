from __future__ import annotations

import re
import tempfile
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from clean_cut.errors import MediaProcessError
from clean_cut.hard_subtitles import extract_sample_frames
from clean_cut.ocr import OcrBackend
from clean_cut.subtitle_data import HardSubtitlePlan, OcrObservation
from clean_cut.subtitle_tracking import build_subtitle_cues, merge_frame_observations


@dataclass(frozen=True, slots=True)
class ResidualInterval:
    start_ms: int
    end_ms: int
    detected_text: str
    confidence: float
    source_match: float
    sample_count: int


@dataclass(slots=True)
class ResidualReport:
    schema_version: int
    repaired: str
    status: str
    inspected_samples: int
    intervals: list[ResidualInterval]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", value, flags=re.UNICODE).casefold()


def _similarity(left: str, right: str) -> float:
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def scan_residual_subtitles(
    repaired: Path,
    plan: HardSubtitlePlan,
    backend: OcrBackend,
    *,
    interval_ms: int | None = None,
    min_source_similarity: float = 0.45,
    cue_margin_ms: int = 200,
) -> ResidualReport:
    repaired = repaired.resolve()
    if not repaired.is_file():
        raise MediaProcessError(f"修复视频不存在：{repaired}")
    if not 0 <= min_source_similarity <= 1:
        raise ValueError("字幕残留相似度阈值必须在0到1之间。")
    if cue_margin_ms < 0:
        raise ValueError("字幕时间边界扩展不能为负数。")
    interval_ms = interval_ms or plan.sampling_interval_ms
    observations: list[OcrObservation] = []
    inspected_samples = 0
    with tempfile.TemporaryDirectory(prefix="clean-cut-residual-") as temp:
        frames = extract_sample_frames(repaired, Path(temp), interval_ms=interval_ms)
        for sample_index, frame in enumerate(frames):
            timestamp_ms = sample_index * interval_ms
            active_cues = [
                cue
                for cue in plan.cues
                if cue.start_ms - cue_margin_ms <= timestamp_ms <= cue.end_ms + cue_margin_ms
            ]
            if not active_cues:
                continue
            inspected_samples += 1
            candidates = backend.recognize(
                frame,
                frame_index=sample_index,
                timestamp_ms=timestamp_ms,
                duration_ms=interval_ms,
                region=plan.region,
            )
            for item in candidates:
                source_similarity = max(
                    _similarity(item.text, cue.text) for cue in active_cues
                )
                if source_similarity >= min_source_similarity:
                    observations.append(item)

    merged = merge_frame_observations(observations)
    detected_cues = build_subtitle_cues(merged)
    intervals = []
    for cue in detected_cues:
        source_match = max(
            (_similarity(cue.text, source.text) for source in plan.cues),
            default=0.0,
        )
        intervals.append(
            ResidualInterval(
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                detected_text=cue.text,
                confidence=cue.confidence,
                source_match=source_match,
                sample_count=cue.observation_count,
            )
        )
    if not plan.cues:
        status = "not_applicable"
    elif inspected_samples == 0:
        status = "inconclusive"
    else:
        status = "needs_review" if intervals else "passed"
    return ResidualReport(
        schema_version=1,
        repaired=str(repaired),
        status=status,
        inspected_samples=inspected_samples,
        intervals=intervals,
    )
