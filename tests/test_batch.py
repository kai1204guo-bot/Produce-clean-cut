from pathlib import Path

import pytest

from clean_cut.batch import chunked, discover_videos, episode_number, episode_stem


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Hollow CityEP1(More inDramal).mp4", 1),
        ("EP015.mp4", 15),
        ("第30集.mp4", 30),
        ("trailer.mp4", None),
    ],
)
def test_episode_number(name: str, expected: int | None) -> None:
    assert episode_number(Path(name)) == expected


def test_discovers_videos_in_episode_order(tmp_path: Path) -> None:
    for name in ("show EP10.mp4", "show EP2.mp4", "notes.txt", "show EP1.mov"):
        (tmp_path / name).touch()
    assert [path.name for path in discover_videos(tmp_path)] == [
        "show EP1.mov",
        "show EP2.mp4",
        "show EP10.mp4",
    ]


def test_chunks_thirty_episodes_into_two_batches() -> None:
    videos = [Path(f"EP{index}.mp4") for index in range(1, 31)]
    batches = list(chunked(videos, 15))
    assert [len(batch) for batch in batches] == [15, 15]
    assert episode_stem(batches[1][-1]) == "EP30"


def test_numeric_filename_keeps_numeric_output_name() -> None:
    assert episode_stem(Path("35.mp4")) == "35"
