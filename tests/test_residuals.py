import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

from clean_cut.ocr import OcrBackend
from clean_cut.residuals import scan_residual_subtitles
from clean_cut.subtitle_data import (
    HardSubtitlePlan,
    OcrObservation,
    Point,
    Polygon,
    Region,
    SubtitleCue,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class _ResidualOcrBackend(OcrBackend):
    def __init__(self, text: str) -> None:
        self.text = text

    def recognize(self, image: Path, **kwargs: object) -> list[OcrObservation]:
        polygon = Polygon((Point(20, 60), Point(140, 60), Point(140, 80), Point(20, 80)))
        return [
            OcrObservation(
                frame_index=int(kwargs["frame_index"]),
                timestamp_ms=int(kwargs["timestamp_ms"]),
                duration_ms=int(kwargs["duration_ms"]),
                text=self.text,
                confidence=0.9,
                polygon=polygon,
            )
        ]


@skipUnless(HAS_FFMPEG, "FFmpeg is required for residual OCR integration tests")
class ResidualScanTests(TestCase):
    def test_flags_text_similar_to_original_subtitle(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "repaired.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=black:s=160x90:d=1:r=10",
                    "-c:v",
                    "libx264",
                    "-y",
                    str(video),
                ],
                check=True,
            )
            plan = HardSubtitlePlan(
                1,
                "source.mp4",
                160,
                90,
                250,
                "test",
                Region(0, 45, 160, 45),
                cues=[SubtitleCue(1, 0, 900, "字幕残留测试", 0.95, 4)],
            )
            report = scan_residual_subtitles(
                video,
                plan,
                _ResidualOcrBackend("字幕残留测式"),
            )
            self.assertEqual(report.status, "needs_review")
            self.assertTrue(report.intervals)

    def test_ignores_unrelated_background_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "repaired.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=black:s=160x90:d=1:r=10",
                    "-c:v",
                    "libx264",
                    "-y",
                    str(video),
                ],
                check=True,
            )
            plan = HardSubtitlePlan(
                1,
                "source.mp4",
                160,
                90,
                250,
                "test",
                Region(0, 45, 160, 45),
                cues=[SubtitleCue(1, 0, 900, "对白字幕", 0.95, 4)],
            )
            report = scan_residual_subtitles(video, plan, _ResidualOcrBackend("商店招牌"))
            self.assertEqual(report.status, "passed")
