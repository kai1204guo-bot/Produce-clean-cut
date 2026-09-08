import shutil
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

from clean_cut.quality import evaluate_repair
from clean_cut.subtitle_data import HardSubtitlePlan, MaskKeyframe, Point, Polygon, Region

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@skipUnless(HAS_FFMPEG, "FFmpeg is required for the quality integration test")
class QualityTests(TestCase):
    def test_evaluates_against_clean_reference(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mkv"
            repaired = root / "repaired.mkv"
            reference = root / "reference.mkv"
            writers = [
                cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 10, (160, 90))
                for path in (source, repaired, reference)
            ]
            try:
                for index in range(5):
                    clean = np.full((90, 160, 3), 80 + index, np.uint8)
                    captioned = clean.copy()
                    captioned[55:70, 50:110] = 255
                    writers[0].write(captioned)
                    writers[1].write(clean)
                    writers[2].write(clean)
            finally:
                for writer in writers:
                    writer.release()
            polygon = Polygon((Point(50, 55), Point(110, 55), Point(110, 70), Point(50, 70)))
            plan = HardSubtitlePlan(
                1,
                str(source.resolve()),
                160,
                90,
                100,
                "test",
                Region(40, 45, 80, 35),
                mask_keyframes=[
                    MaskKeyframe(index, index * 100, (polygon,)) for index in range(5)
                ],
            )
            report = evaluate_repair(source, repaired, plan, reference_clean=reference)
            self.assertEqual(report.inspected_frames, 5)
            self.assertEqual(report.masked_mae, 0)
            self.assertEqual(report.masked_psnr_db, 100)
            self.assertEqual(report.temporal_residual_mae, 0)
