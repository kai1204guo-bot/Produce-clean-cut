from __future__ import annotations

from clean_cut.subtitle_data import SubtitleCue


def format_srt_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError("SRT时间戳不能为负数。")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def render_srt(cues: list[SubtitleCue]) -> str:
    blocks: list[str] = []
    for cue in cues:
        if cue.end_ms <= cue.start_ms:
            raise ValueError(f"第{cue.index}条字幕的结束时间必须晚于开始时间。")
        text = cue.text.replace("\r\n", "\n").replace("\r", "\n").strip()
        blocks.append(
            f"{cue.index}\n"
            f"{format_srt_timestamp(cue.start_ms)} --> {format_srt_timestamp(cue.end_ms)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")

