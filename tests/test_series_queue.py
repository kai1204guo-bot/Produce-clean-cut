import json
from pathlib import Path

from clean_cut.series_queue import SeriesQueueStore, task_from_source


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
