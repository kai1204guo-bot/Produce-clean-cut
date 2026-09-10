from pathlib import Path

import pytest

from clean_cut.batch import BatchJob, BatchManifest, JobState
from clean_cut.segmentation import plan_equal_segments


def test_exactly_sixty_seconds_stays_as_one_job() -> None:
    windows = plan_equal_segments(60.0)

    assert len(windows) == 1
    assert windows[0].input_start == 0
    assert windows[0].input_end == 60.0


def test_two_minutes_one_second_is_split_evenly() -> None:
    windows = plan_equal_segments(121.0)

    assert len(windows) == 3
    assert [window.core_duration for window in windows] == pytest.approx(
        [121 / 3, 121 / 3, 121 / 3]
    )
    assert max(window.input_end - window.input_start for window in windows) < 58
    assert windows[0].input_start == 0
    assert windows[-1].input_end == 121
    assert windows[1].trim_start == pytest.approx(0.5)


def test_long_episode_has_no_tiny_tail() -> None:
    windows = plan_equal_segments(181.0)

    durations = [window.core_duration for window in windows]
    assert max(durations) - min(durations) < 0.001
    assert min(durations) >= 3
    assert max(window.input_end - window.input_start for window in windows) < 58


def test_segment_metadata_survives_manifest_resume(tmp_path: Path) -> None:
    source = tmp_path / "segment-a.mp4"
    source.touch()
    output = tmp_path / "clean" / "segment-a-clean.mp4"
    requested = BatchJob(
        source=str(source),
        episode=None,
        clean_output=str(output),
        parent_source=str(tmp_path / "EP1.mp4"),
        segment_index=1,
        segment_count=3,
        segment_core_start=0,
        segment_core_end=40,
        segment_input_start=0,
        segment_input_end=40.5,
    )
    state = tmp_path / "state.json"
    first = BatchManifest.load_or_create_jobs(state, [requested])
    first.update(next(iter(first.jobs.values())), JobState.SUBMITTED, "submitted", 25)

    resumed = BatchManifest.load_or_create_jobs(state, [requested])
    restored = next(iter(resumed.jobs.values()))

    assert restored.state is JobState.SUBMITTED
    assert restored.progress == 25
    assert restored.parent_key == requested.parent_key
    assert restored.segment_core_end == 40
