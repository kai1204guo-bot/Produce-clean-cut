import json
from pathlib import Path
from unittest.mock import patch

from clean_cut.desktop import _asr_worker


class FakeTranscriber:
    def __init__(self, **_kwargs) -> None:
        pass

    def write_srt(self, _source: Path, destination: Path, **_kwargs) -> Path:
        destination.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        return destination


def test_asr_worker_writes_output_and_completion_status(tmp_path: Path) -> None:
    source = tmp_path / "episode.mp4"
    source.touch()
    destination = tmp_path / "episode.srt"
    status = tmp_path / "status.json"
    jobs = tmp_path / "jobs.json"
    jobs.write_text(
        json.dumps(
            {
                "status": str(status),
                "jobs": [{"source": str(source), "destination": str(destination)}],
            }
        ),
        encoding="utf-8",
    )

    with patch("clean_cut.desktop.FasterWhisperTranscriber", FakeTranscriber):
        _asr_worker(jobs, "cuda")

    assert destination.is_file()
    assert json.loads(status.read_text(encoding="utf-8"))["state"] == "complete"
