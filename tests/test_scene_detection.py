from unittest import TestCase

from clean_cut.scene_detection import SceneDetector


class SceneDetectorTests(TestCase):
    def test_detects_abrupt_cut_but_not_static_frames(self) -> None:
        import numpy as np

        detector = SceneDetector(threshold=0.6, min_interval_frames=2)
        black = np.zeros((90, 160, 3), np.uint8)
        white = np.full((90, 160, 3), 255, np.uint8)
        self.assertIsNone(detector.update(black, 0, 10))
        self.assertIsNone(detector.update(black, 1, 10))
        cut = detector.update(white, 2, 10)
        self.assertIsNotNone(cut)
        self.assertEqual(cut.frame_index if cut else None, 2)
        self.assertIsNone(detector.update(white, 3, 10))
