from types import SimpleNamespace
from unittest import TestCase

from clean_cut.asr import segments_to_cues


class FasterWhisperCueTests(TestCase):
    def test_uses_word_timestamps_and_confidence(self) -> None:
        segment = SimpleNamespace(
            start=1.0,
            end=9.0,
            text=" Hold him down! ",
            words=[
                SimpleNamespace(start=1.2, end=1.5, probability=0.8),
                SimpleNamespace(start=1.5, end=2.0, probability=1.0),
            ],
        )

        cues = segments_to_cues([segment])

        self.assertEqual(cues[0].start_ms, 1200)
        self.assertEqual(cues[0].end_ms, 2000)
        self.assertEqual(cues[0].text, "Hold him down!")
        self.assertEqual(cues[0].confidence, 0.9)
