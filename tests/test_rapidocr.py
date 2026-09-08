from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, skipUnless

HAS_OCR = importlib.util.find_spec("rapidocr") is not None
HAS_PILLOW = importlib.util.find_spec("PIL") is not None


@skipUnless(HAS_OCR and HAS_PILLOW, "RapidOCR and Pillow are required")
class RapidOcrBackendTests(TestCase):
    def test_rejects_detection_spanning_most_of_ocr_region(self) -> None:
        from PIL import Image

        from clean_cut.ocr import RapidOcrBackend
        from clean_cut.subtitle_data import Region

        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "frame.png"
            Image.new("RGB", (320, 200), "black").save(image_path)
            backend = RapidOcrBackend.__new__(RapidOcrBackend)
            backend._max_box_height_ratio = 0.5
            backend._engine = lambda _frame: SimpleNamespace(
                boxes=[[[0, 0], [319, 0], [319, 199], [0, 199]]],
                txts=["M"],
                scores=[0.9],
            )

            observations = backend.recognize(
                image_path,
                frame_index=0,
                timestamp_ms=0,
                duration_ms=250,
                region=Region(x=0, y=0, width=320, height=200),
            )

            self.assertEqual(observations, [])

    def test_recognizes_generated_subtitle_inside_region(self) -> None:
        from PIL import Image, ImageDraw, ImageFont

        from clean_cut.ocr import RapidOcrBackend
        from clean_cut.subtitle_data import Region

        font_candidates = (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        )
        font_path = next((path for path in font_candidates if path.is_file()), None)
        if font_path is None:
            self.skipTest("No suitable test font is installed")

        with tempfile.TemporaryDirectory() as temp:
            unicode_dir = Path(temp) / "中文字幕测试"
            unicode_dir.mkdir()
            image_path = unicode_dir / "subtitle.png"
            image = Image.new("RGB", (960, 240), "black")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(str(font_path), 72)
            draw.text((170, 105), "CLEAN CUT 123", fill="white", font=font, stroke_width=2)
            image.save(image_path)

            backend = RapidOcrBackend(text_score=0.4)
            observations = backend.recognize(
                image_path,
                frame_index=25,
                timestamp_ms=1000,
                duration_ms=40,
                region=Region(x=100, y=80, width=760, height=140),
            )

            self.assertTrue(observations)
            combined = " ".join(item.text.upper() for item in observations)
            self.assertIn("123", combined)
            self.assertTrue(all(item.timestamp_ms == 1000 for item in observations))
            self.assertTrue(all(item.polygon.bounds[0] >= 100 for item in observations))

    def test_recognizes_generated_chinese_subtitle_when_font_is_available(self) -> None:
        from PIL import Image, ImageDraw, ImageFont

        from clean_cut.ocr import RapidOcrBackend
        from clean_cut.subtitle_data import Region

        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        if not font_path.is_file():
            self.skipTest("Microsoft YaHei is not installed")

        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "chinese-subtitle.png"
            image = Image.new("RGB", (960, 240), "white")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(str(font_path), 64)
            draw.text(
                (170, 70),
                "你好世界字幕测试",
                fill="black",
                font=font,
            )
            image.save(image_path)

            observations = RapidOcrBackend(text_score=0.4).recognize(
                image_path,
                frame_index=0,
                timestamp_ms=0,
                duration_ms=40,
                region=Region(x=80, y=30, width=800, height=170),
            )
            combined = "".join(item.text for item in observations)
            self.assertIn("你好世界字幕测试", combined)
