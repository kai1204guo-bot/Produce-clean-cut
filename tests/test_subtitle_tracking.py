from unittest import TestCase

from clean_cut.srt import format_srt_timestamp, render_srt
from clean_cut.subtitle_data import OcrObservation, Point, Polygon
from clean_cut.subtitle_tracking import build_subtitle_cues, merge_frame_observations


def observation(
    frame: int,
    timestamp: int,
    text: str,
    x: float = 100,
    confidence: float = 0.9,
) -> OcrObservation:
    return OcrObservation(
        frame_index=frame,
        timestamp_ms=timestamp,
        duration_ms=500,
        text=text,
        confidence=confidence,
        polygon=Polygon(
            (
                Point(x, 800),
                Point(x + 300, 800),
                Point(x + 300, 860),
                Point(x, 860),
            )
        ),
    )


class SubtitleTrackingTests(TestCase):
    def test_drops_low_confidence_single_character_noise(self) -> None:
        cues = build_subtitle_cues(
            [observation(0, 0, "M", confidence=0.6)]
        )

        self.assertEqual(cues, [])

    def test_prefers_high_confidence_reading_over_repeated_ocr_error(self) -> None:
        cues = build_subtitle_cues(
            [
                observation(0, 0, "you know what I said", confidence=0.97),
                observation(1, 250, "Yot w wo nid", confidence=0.70),
                observation(2, 500, "Yot w wo nid", confidence=0.68),
                observation(3, 750, "Yot w wo nid", confidence=0.68),
            ]
        )

        self.assertEqual(cues[0].text, "you know what I said")

    def test_merges_jittered_text_and_splits_changed_subtitle(self) -> None:
        cues = build_subtitle_cues(
            [
                observation(0, 0, "开始测试"),
                observation(1, 500, "开始测式", x=102),
                observation(2, 1000, "开始测试", x=101),
                observation(3, 1500, "第二句话", x=100),
                observation(4, 2000, "第二句话", x=101),
            ]
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0].text, "开始测试")
        self.assertEqual(cues[0].start_ms, 0)
        self.assertEqual(cues[0].end_ms, 1500)
        self.assertEqual(cues[0].observation_count, 3)
        self.assertEqual(cues[1].text, "第二句话")

    def test_same_text_at_different_locations_creates_separate_tracks(self) -> None:
        cues = build_subtitle_cues(
            [
                observation(0, 0, "位置测试", x=50),
                observation(0, 0, "位置测试", x=800),
            ]
        )
        self.assertEqual(len(cues), 2)

    def test_merges_multiple_lines_from_the_same_frame(self) -> None:
        merged = merge_frame_observations(
            [
                observation(0, 0, "第一行", x=100),
                OcrObservation(
                    frame_index=0,
                    timestamp_ms=0,
                    duration_ms=500,
                    text="第二行",
                    confidence=0.8,
                    polygon=Polygon(
                        (
                            Point(120, 870),
                            Point(380, 870),
                            Point(380, 930),
                            Point(120, 930),
                        )
                    ),
                ),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text, "第一行\n第二行")

    def test_renders_standard_srt(self) -> None:
        cues = build_subtitle_cues([observation(0, 3_723_004, "SRT测试")])
        rendered = render_srt(cues)
        self.assertIn("01:02:03,004 --> 01:02:03,504", rendered)
        self.assertTrue(rendered.endswith("\n"))
        self.assertEqual(format_srt_timestamp(0), "00:00:00,000")
