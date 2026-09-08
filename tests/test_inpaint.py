import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest import TestCase, skipUnless

from clean_cut.inpaint import (
    LamaOnnxBackend,
    OpenCvInpaintBackend,
    SequenceInpaintBackend,
    SttnBackend,
    composite_repair,
)
from clean_cut.masks import MaskRenderConfig, render_mask
from clean_cut.media import probe_media
from clean_cut.subtitle_data import HardSubtitlePlan, MaskKeyframe, Point, Polygon, Region
from clean_cut.video_repair import repair_video

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class InpaintTests(TestCase):
    def test_opencv_backend_changes_only_mask_neighborhood(self) -> None:
        import numpy as np

        frame = np.full((80, 120, 3), 100, dtype=np.uint8)
        frame[30:50, 40:80] = 255
        polygon = Polygon((Point(40, 30), Point(80, 30), Point(80, 50), Point(40, 50)))
        mask = render_mask(
            120,
            80,
            (polygon,),
            MaskRenderConfig(dilation_px=1, feather_px=2),
        )
        repaired = OpenCvInpaintBackend().inpaint(frame, mask.binary)
        result = composite_repair(frame, repaired, mask)
        self.assertLess(float(result[40, 60].mean()), 200)
        self.assertTrue((result[0, 0] == frame[0, 0]).all())

    @skipUnless(HAS_FFMPEG, "FFmpeg is required for the integration test")
    def test_repairs_cfr_video_and_preserves_audio(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            destination = root / "clean.mp4"
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
                    "color=c=gray:s=160x90:d=1:r=10,drawbox=x=50:y=55:w=60:h=15:color=white:t=fill",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(source),
                ],
                check=True,
            )
            polygon = Polygon(
                (Point(50, 55), Point(110, 55), Point(110, 70), Point(50, 70))
            )
            plan = HardSubtitlePlan(
                schema_version=1,
                source=str(source.resolve()),
                video_width=160,
                video_height=90,
                sampling_interval_ms=500,
                ocr_backend="test",
                region=Region(40, 45, 80, 35),
                mask_keyframes=[
                    MaskKeyframe(0, 0, (polygon,)),
                    MaskKeyframe(1, 500, (polygon,)),
                ],
            )
            repair_video(
                source,
                destination,
                plan,
                OpenCvInpaintBackend(),
                mask_config=MaskRenderConfig(dilation_px=2, feather_px=2),
            )

            output_info = probe_media(destination)
            self.assertTrue(any(stream.codec_type == "audio" for stream in output_info.streams))
            capture = cv2.VideoCapture(str(destination))
            ok, frame = capture.read()
            capture.release()
            self.assertTrue(ok)
            self.assertLess(float(frame[60, 80].mean()), 200)

    @skipUnless(HAS_FFMPEG, "FFmpeg is required for the integration test")
    def test_repairs_video_in_overlapping_temporal_chunks(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mkv"
            destination = root / "clean.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"FFV1"),
                10,
                (160, 90),
            )
            for _ in range(10):
                frame = np.full((90, 160, 3), 120, np.uint8)
                frame[55:70, 50:110] = 255
                writer.write(frame)
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
                    MaskKeyframe(index, index * 100, (polygon,)) for index in range(10)
                ],
            )
            backend = _FlatSequenceBackend()
            repair_video(
                source,
                destination,
                plan,
                backend,
                temporal_chunk_frames=6,
                temporal_overlap_frames=2,
            )
            self.assertGreaterEqual(backend.calls, 2)
            capture = cv2.VideoCapture(str(destination))
            output_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            self.assertEqual(output_frames, 10)

    @skipUnless(HAS_FFMPEG, "FFmpeg is required for the integration test")
    def test_scene_cut_flushes_temporal_context(self) -> None:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mkv"
            destination = root / "clean.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"FFV1"),
                10,
                (160, 90),
            )
            for index in range(8):
                value = 0 if index < 4 else 255
                writer.write(np.full((90, 160, 3), value, np.uint8))
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
                    MaskKeyframe(index, index * 100, (polygon,)) for index in range(8)
                ],
            )
            backend = _FlatSequenceBackend()
            repair_video(
                source,
                destination,
                plan,
                backend,
                temporal_chunk_frames=20,
                temporal_overlap_frames=2,
                scene_threshold=0.6,
            )
            self.assertEqual(backend.calls, 2)

    def test_lama_model_inference_when_model_is_configured(self) -> None:
        import numpy as np

        model_value = os.environ.get("CLEAN_CUT_LAMA_MODEL")
        if not model_value:
            self.skipTest("CLEAN_CUT_LAMA_MODEL is not configured")
        frame = np.full((96, 160, 3), 120, dtype=np.uint8)
        frame[35:60, 50:110] = 255
        mask = np.zeros((96, 160), dtype=np.uint8)
        mask[32:63, 47:113] = 255
        use_gpu = os.environ.get("CLEAN_CUT_LAMA_DEVICE") == "cuda"
        backend = LamaOnnxBackend(Path(model_value), use_gpu=use_gpu)
        result = backend.inpaint(frame, mask)
        if use_gpu:
            self.assertEqual(backend.provider, "CUDAExecutionProvider")
        self.assertEqual(result.shape, frame.shape)
        self.assertEqual(result.dtype, frame.dtype)
        self.assertFalse(np.array_equal(result, frame))

    def test_sttn_model_inference_when_model_is_configured(self) -> None:
        import numpy as np

        model_value = os.environ.get("CLEAN_CUT_STTN_MODEL")
        if not model_value:
            self.skipTest("CLEAN_CUT_STTN_MODEL is not configured")
        frames = [np.full((90, 160, 3), 100 + index * 2, np.uint8) for index in range(3)]
        masks = [np.zeros((90, 160), np.uint8) for _ in frames]
        for mask in masks:
            mask[50:70, 50:110] = 255
        use_gpu = os.environ.get("CLEAN_CUT_STTN_DEVICE") == "cuda"
        backend = SttnBackend(Path(model_value), use_gpu=use_gpu)
        results = backend.inpaint_sequence(frames, masks)
        self.assertEqual(len(results), len(frames))
        self.assertTrue(all(result.shape == frames[0].shape for result in results))
        self.assertTrue(all(result.dtype == frames[0].dtype for result in results))
        self.assertFalse(np.array_equal(results[0], frames[0]))


class _FlatSequenceBackend(SequenceInpaintBackend):
    def __init__(self) -> None:
        self.calls = 0

    def inpaint_sequence(self, frames: list[object], masks: list[object]) -> list[object]:
        import numpy as np

        self.calls += 1
        return [np.full_like(frame, 80) for frame in frames]
