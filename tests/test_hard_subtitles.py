from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

from clean_cut.hard_subtitles import analyze_hard_subtitles, build_mask_keyframes
from clean_cut.ocr import OcrBackend
from clean_cut.subtitle_data import (
    HardSubtitlePlan,
    MaskKeyframe,
    OcrObservation,
    Point,
    Polygon,
    Region,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class FixedOcrBackend(OcrBackend):
    def recognize(
        self,
        image: Path,
        *,
        frame_index: int,
        timestamp_ms: int,
        duration_ms: int,
        region: Region | None = None,
    ) -> list[OcrObservation]:
        del image
        selected = region or Region(0, 0, 100, 30)
        polygon = Polygon(
            (
                Point(selected.x, selected.y),
                Point(selected.x + selected.width, selected.y),
                Point(selected.x + selected.width, selected.y + selected.height),
                Point(selected.x, selected.y + selected.height),
            )
        )
        return [
            OcrObservation(
                frame_index,
                timestamp_ms,
                duration_ms,
                "端到端硬字幕",
                0.95,
                polygon,
            )
        ]


class HardSubtitlePlanTests(TestCase):
    def test_plan_json_data_round_trips(self) -> None:
        polygon = Polygon((Point(1, 2), Point(4, 2), Point(4, 6), Point(1, 6)))
        plan = HardSubtitlePlan(
            schema_version=1,
            source="sample.mp4",
            video_width=1920,
            video_height=1080,
            sampling_interval_ms=250,
            ocr_backend="test",
            region=Region(0, 700, 1920, 300),
            mask_keyframes=[MaskKeyframe(2, 500, (polygon,))],
        )
        self.assertEqual(HardSubtitlePlan.from_dict(plan.to_dict()).to_dict(), plan.to_dict())

    def test_groups_polygons_into_mask_keyframes(self) -> None:
        polygon = Polygon(
            (Point(1, 1), Point(10, 1), Point(10, 5), Point(1, 5))
        )
        observations = [
            OcrObservation(0, 0, 40, "A", 0.9, polygon),
            OcrObservation(0, 0, 40, "B", 0.8, polygon),
            OcrObservation(1, 40, 40, "A", 0.9, polygon),
        ]
        keyframes = build_mask_keyframes(observations)
        self.assertEqual(len(keyframes), 2)
        self.assertEqual(len(keyframes[0].polygons), 2)
        self.assertEqual(keyframes[1].timestamp_ms, 40)

    @skipUnless(HAS_FFMPEG, "FFmpeg is required for the integration test")
    def test_video_to_plan_and_srt_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "hard.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=320x180:d=1:r=25",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(source),
                ],
                check=True,
            )
            output = root / "output"
            plan = analyze_hard_subtitles(
                source,
                output,
                region=Region(20, 120, 280, 50),
                backend=FixedOcrBackend(),
                interval_ms=500,
            )
            self.assertEqual(len(plan.cues), 1)
            self.assertEqual(plan.cues[0].text, "端到端硬字幕")
            self.assertEqual(len(plan.mask_keyframes), 2)
            self.assertEqual(plan.sampling_interval_ms, 500)
            self.assertTrue((output / "hard.srt").is_file())
            self.assertTrue((output / "hard_hard_subtitle_plan.json").is_file())
