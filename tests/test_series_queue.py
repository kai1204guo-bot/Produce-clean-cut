import json
from pathlib import Path

from clean_cut.series_queue import (
    SeriesQueueStore,
    discover_series_sources,
    task_from_source,
)


def test_task_from_source_derives_output_folders(tmp_path: Path) -> None:
    source = tmp_path / "成片 MY SHOW" / "成片"
    source.mkdir(parents=True)

    task = task_from_source(source)

    assert task.name == "MY SHOW"
    assert Path(task.clean) == source.parent / "清水版"
    assert Path(task.srt) == source.parent / "srt"


def test_series_queue_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "剧A" / "成片"
    source.mkdir(parents=True)
    task = task_from_source(source)
    store = SeriesQueueStore(tmp_path / "queue.json")

    store.save([task])
    loaded = store.load()

    assert len(loaded) == 1
    assert loaded[0].task_id == task.task_id
    assert loaded[0].source == str(source.resolve())
    assert json.loads(store.path.read_text(encoding="utf-8"))["version"] == 1


def test_discover_series_sources_finds_all_series_and_excludes_outputs(
    tmp_path: Path,
) -> None:
    first = tmp_path / "剧A" / "成片"
    second = tmp_path / "剧B" / "videos"
    clean = tmp_path / "剧C" / "清水版" / "成片"
    work = tmp_path / "剧D" / ".clean-cut-segments" / "成片"
    for folder in (first, second, clean, work):
        folder.mkdir(parents=True)
        (folder / "EP1.mp4").write_bytes(b"video")

    assert discover_series_sources(tmp_path) == [first.resolve(), second.resolve()]


def test_discover_series_sources_ignores_empty_source_folders(tmp_path: Path) -> None:
    (tmp_path / "剧A" / "成片").mkdir(parents=True)

    assert discover_series_sources(tmp_path) == []
