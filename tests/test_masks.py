from unittest import TestCase

from clean_cut.masks import MaskRenderConfig, polygons_for_timestamp, render_mask
from clean_cut.subtitle_data import MaskKeyframe, Point, Polygon


class MaskTests(TestCase):
    def test_selects_nearest_keyframe_with_hold(self) -> None:
        polygon = Polygon((Point(10, 10), Point(20, 10), Point(20, 20), Point(10, 20)))
        keyframes = [MaskKeyframe(0, 1000, (polygon,))]
        config = MaskRenderConfig(hold_before_ms=100, hold_after_ms=100)
        self.assertTrue(
            polygons_for_timestamp(
                keyframes,
                800,
                sampling_interval_ms=250,
                config=config,
            )
        )
        self.assertFalse(
            polygons_for_timestamp(
                keyframes,
                700,
                sampling_interval_ms=250,
                config=config,
            )
        )

    def test_renders_dilated_and_feathered_mask(self) -> None:
        polygon = Polygon((Point(20, 20), Point(30, 20), Point(30, 30), Point(20, 30)))
        mask = render_mask(
            64,
            64,
            (polygon,),
            MaskRenderConfig(dilation_px=3, feather_px=2),
        )
        self.assertEqual(mask.binary.shape, (64, 64))
        self.assertEqual(int(mask.binary[25, 25]), 255)
        self.assertGreater(int(mask.binary.sum()), 10 * 10 * 255)
        self.assertTrue((mask.alpha > 0).any())
