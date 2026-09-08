from pathlib import Path
from unittest import TestCase

from clean_cut.media import classify_subtitle_codec, has_variable_frame_rate, parse_probe
from clean_cut.models import MediaStream, ProcessingRoute, SubtitleKind


class SubtitleClassificationTests(TestCase):
    def test_classifies_known_text_and_bitmap_codecs(self) -> None:
        self.assertEqual(classify_subtitle_codec("subrip"), SubtitleKind.TEXT)
        self.assertEqual(classify_subtitle_codec("mov_text"), SubtitleKind.TEXT)
        self.assertEqual(classify_subtitle_codec("hdmv_pgs_subtitle"), SubtitleKind.BITMAP)
        self.assertEqual(classify_subtitle_codec("mystery"), SubtitleKind.UNKNOWN)

    def test_text_subtitle_route_has_priority(self) -> None:
        payload = {
            "format": {"format_name": "matroska", "duration": "3.0", "size": "100"},
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264"},
                {
                    "index": 1,
                    "codec_type": "subtitle",
                    "codec_name": "subrip",
                    "tags": {"language": "zho"},
                    "disposition": {"default": 1},
                },
            ],
        }
        info = parse_probe(Path("sample.mkv"), payload)
        self.assertEqual(info.route, ProcessingRoute.TEXT_SOFT_SUBTITLE)
        self.assertEqual(info.text_subtitle_streams[0].language, "zho")
        self.assertTrue(info.text_subtitle_streams[0].is_default)

    def test_missing_subtitle_routes_to_hard_subtitle_analysis(self) -> None:
        payload = {
            "format": {"format_name": "mov,mp4", "duration": "1.0"},
            "streams": [{"index": 0, "codec_type": "video", "codec_name": "h264"}],
        }
        info = parse_probe(Path("sample.mp4"), payload)
        self.assertEqual(info.route, ProcessingRoute.POSSIBLE_HARD_SUBTITLE)

    def test_detects_variable_frame_rate_from_probe_rates(self) -> None:
        cfr = MediaStream(0, "video", "h264", r_frame_rate="25/1", avg_frame_rate="25/1")
        vfr = MediaStream(0, "video", "h264", r_frame_rate="30/1", avg_frame_rate="24/1")
        self.assertFalse(has_variable_frame_rate(cfr))
        self.assertTrue(has_variable_frame_rate(vfr))
