class CleanCutError(RuntimeError):
    """Base error for failures that should be shown to the user."""


class ToolNotFoundError(CleanCutError):
    """A required external executable is unavailable."""


class MediaProbeError(CleanCutError):
    """ffprobe could not read the input media."""


class MediaProcessError(CleanCutError):
    """ffmpeg could not produce an expected output."""


class LibTvAuthenticationError(MediaProcessError):
    """LibTV CLI authorization is missing or expired."""
