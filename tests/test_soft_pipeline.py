from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

from clean_cut.errors import MediaProcessError
from clean_cut.media import probe_media
from clean_cut.models import ProcessingRoute
from clean_cut.pipeline import process_media


HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@skipUnless(HAS_FFMPEG, "FFmpeg is required for the integration test")
class SoftSubtitlePipelineTests(TestCase):
    def test_extracts_srt_and_remuxes_without_subtitles(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            subtitles = root / "captions.srt"
            source = root / "source.mkv"
            output_dir = root / "output"
            subtitles.write_text(
                "1\n00:00:00,200 --> 00:00:01,200\n阶段零字幕测试\n",
                encoding="utf-8",
            )
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
                    "color=c=black:s=320x180:d=2:r=25",
                    "-f",
                    "lavfi",
                    "-i",
                    "anullsrc=r=48000:cl=stereo:d=2",
                    "-i",
                    str(subtitles),
                    "-map",
                    "0:v",
                    "-map",
                    "1:a",
                    "-map",
                    "2:s",
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-c:s",
                    "srt",
                    "-shortest",
                    str(source),
                ],
                check=True,
            )

            before = probe_media(source)
            self.assertEqual(before.route, ProcessingRoute.TEXT_SOFT_SUBTITLE)

            report = process_media(source, output_dir)
            self.assertEqual(report.status, "completed")
            self.assertIsNotNone(report.clean_video)
            self.assertTrue(report.subtitle_files[0].is_file())
            self.assertIn("阶段零字幕测试", report.subtitle_files[0].read_text(encoding="utf-8-sig"))

            after = probe_media(report.clean_video or Path())
            self.assertEqual(len(after.subtitle_streams), 0)
            self.assertTrue(any(stream.codec_type == "audio" for stream in after.streams))

            with self.assertRaises(MediaProcessError):
                process_media(source, output_dir)
