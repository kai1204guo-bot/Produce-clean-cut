from pathlib import Path
from unittest.mock import patch

from clean_cut.dependencies import detect_dependencies, install_dependency
from clean_cut.tools import locate_executable


def test_locate_libtv_uses_official_user_install_location(tmp_path: Path) -> None:
    executable = tmp_path / ".libtv" / "libtv.exe"
    executable.parent.mkdir()
    executable.touch()

    with (
        patch("clean_cut.tools.shutil.which", return_value=None),
        patch("clean_cut.tools.Path.home", return_value=tmp_path),
    ):
        assert locate_executable("libtv") == str(executable)


def test_dependency_detection_marks_required_tools(monkeypatch) -> None:
    available = {
        "ffmpeg": "ffmpeg.exe",
        "ffprobe": "ffprobe.exe",
        "libtv": "libtv.exe",
        "nvidia-smi": None,
    }
    monkeypatch.setattr(
        "clean_cut.dependencies.locate_executable", lambda name: available.get(name)
    )
    monkeypatch.setattr("clean_cut.dependencies._chrome_path", lambda: Path("chrome.exe"))
    monkeypatch.setattr("clean_cut.dependencies._cuda_ready", lambda: False)
    monkeypatch.setattr("clean_cut.dependencies.importlib.util.find_spec", lambda _name: object())

    statuses = {item.key: item for item in detect_dependencies()}

    assert statuses["ffmpeg"].installed
    assert statuses["libtv"].installed
    assert statuses["chrome"].installed
    assert statuses["runtime"].installed
    assert not statuses["cuda"].required


def test_ffmpeg_install_falls_back_when_winget_is_missing() -> None:
    with (
        patch("clean_cut.dependencies._install_winget", return_value=False),
        patch("clean_cut.dependencies._install_ffmpeg_direct") as direct,
    ):
        install_dependency("ffmpeg")

    direct.assert_called_once()


def test_chrome_install_falls_back_when_winget_is_missing() -> None:
    with (
        patch("clean_cut.dependencies._install_winget", return_value=False),
        patch("clean_cut.dependencies._install_chrome_direct") as direct,
    ):
        install_dependency("chrome")

    direct.assert_called_once()
