from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from clean_cut.subtitle_data import OcrObservation, Point, Polygon, SubtitleCue

_SPACE_PATTERN = re.compile(r"\s+")


def normalize_ocr_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return _SPACE_PATTERN.sub(" ", normalized)


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        normalize_ocr_text(left).casefold(),
        normalize_ocr_text(right).casefold(),
    ).ratio()


def polygon_iou(left: OcrObservation, right: OcrObservation) -> float:
    left_x1, left_y1, left_x2, left_y2 = left.polygon.bounds
    right_x1, right_y1, right_x2, right_y2 = right.polygon.bounds
    intersection_width = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1))
    intersection_height = max(0.0, min(left_y2, right_y2) - max(left_y1, right_y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass(slots=True)
class _Track:
    observations: list[OcrObservation] = field(default_factory=list)

    @property
    def last(self) -> OcrObservation:
        return self.observations[-1]


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    max_gap_ms: int = 750
    minimum_iou: float = 0.35
    minimum_text_similarity: float = 0.55
    minimum_cue_duration_ms: int = 120
    minimum_single_observation_confidence: float = 0.8


def merge_frame_observations(
    observations: list[OcrObservation],
) -> list[OcrObservation]:
    grouped: dict[tuple[int, int], list[OcrObservation]] = {}
    for item in observations:
        grouped.setdefault((item.frame_index, item.timestamp_ms), []).append(item)

    merged: list[OcrObservation] = []
    for items in grouped.values():
        ordered = sorted(items, key=lambda item: (item.polygon.bounds[1], item.polygon.bounds[0]))
        if len(ordered) == 1:
            merged.append(ordered[0])
            continue
        left = min(item.polygon.bounds[0] for item in ordered)
        top = min(item.polygon.bounds[1] for item in ordered)
        right = max(item.polygon.bounds[2] for item in ordered)
        bottom = max(item.polygon.bounds[3] for item in ordered)
        confidence_weight = sum(item.confidence for item in ordered)
        merged.append(
            OcrObservation(
                frame_index=ordered[0].frame_index,
                timestamp_ms=ordered[0].timestamp_ms,
                duration_ms=max(item.duration_ms for item in ordered),
                text="\n".join(item.text.strip() for item in ordered),
                confidence=confidence_weight / len(ordered),
                polygon=Polygon(
                    (
                        Point(left, top),
                        Point(right, top),
                        Point(right, bottom),
                        Point(left, bottom),
                    )
                ),
            )
        )
    return sorted(merged, key=lambda item: (item.timestamp_ms, item.frame_index))


def _match_score(
    track: _Track,
    observation: OcrObservation,
    config: TrackingConfig,
) -> float | None:
    gap = observation.timestamp_ms - track.last.timestamp_ms
    if gap < 0 or gap > config.max_gap_ms:
        return None
    overlap = polygon_iou(track.last, observation)
    similarity = text_similarity(track.last.text, observation.text)
    if overlap < config.minimum_iou or similarity < config.minimum_text_similarity:
        return None
    return overlap * 0.55 + similarity * 0.45


def _consensus(track: _Track) -> tuple[str, float]:
    # OCR can repeat the same low-confidence mistake across adjacent near-identical
    # frames. Prefer the engine's strongest reading instead of letting that repeated
    # mistake outvote a single high-confidence observation.
    representative = max(
        track.observations,
        key=lambda item: (item.confidence, len(normalize_ocr_text(item.text))),
    )
    confidence = sum(item.confidence for item in track.observations) / len(
        track.observations
    )
    return normalize_ocr_text(representative.text), confidence


def build_subtitle_cues(
    observations: list[OcrObservation],
    config: TrackingConfig | None = None,
) -> list[SubtitleCue]:
    config = config or TrackingConfig()
    tracks: list[_Track] = []
    active: list[_Track] = []

    for observation in sorted(
        observations,
        key=lambda item: (item.timestamp_ms, item.frame_index, item.polygon.bounds),
    ):
        active = [
            track
            for track in active
            if observation.timestamp_ms - track.last.timestamp_ms <= config.max_gap_ms
        ]
        scored = [
            (score, track)
            for track in active
            if (score := _match_score(track, observation, config)) is not None
        ]
        if scored:
            _, selected = max(scored, key=lambda item: item[0])
            selected.observations.append(observation)
        else:
            selected = _Track([observation])
            tracks.append(selected)
            active.append(selected)

    ordered_tracks = sorted(tracks, key=lambda track: track.observations[0].timestamp_ms)
    cues: list[SubtitleCue] = []
    for index, track in enumerate(ordered_tracks, start=1):
        text, confidence = _consensus(track)
        if (
            len(track.observations) == 1
            and len(text) <= 1
            and confidence < config.minimum_single_observation_confidence
        ):
            continue
        first = track.observations[0]
        last = track.observations[-1]
        natural_end = last.timestamp_ms + last.duration_ms
        next_start = (
            ordered_tracks[index].observations[0].timestamp_ms
            if index < len(ordered_tracks)
            else None
        )
        end_ms = min(natural_end, next_start) if next_start is not None else natural_end
        end_ms = max(end_ms, first.timestamp_ms + config.minimum_cue_duration_ms)
        cues.append(
            SubtitleCue(
                index=len(cues) + 1,
                start_ms=first.timestamp_ms,
                end_ms=end_ms,
                text=text,
                confidence=round(confidence, 4),
                observation_count=len(track.observations),
            )
        )
    return cues
