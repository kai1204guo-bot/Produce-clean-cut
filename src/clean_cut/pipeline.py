from __future__ import annotations

import json
from pathlib import Path

from clean_cut.errors import MediaProcessError
from clean_cut.media import probe_media
from clean_cut.models import ProcessingReport, ProcessingRoute
from clean_cut.soft_subtitles import (
    choose_primary_text_subtitle,
    extract_text_subtitle,
    remux_without_subtitles,
)
from clean_cut.tools import write_text_atomically


def _safe_stem(path: Path) -> str:
    return path.stem.rstrip(" .") or "video"


def _require_new_outputs(paths: list[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "、".join(str(path) for path in existing)
        raise MediaProcessError(f"输出文件已存在，未执行覆盖：{joined}")


def process_media(source: Path, output_dir: Path) -> ProcessingReport:
    media = probe_media(source)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(media.source)
    report_path = output_dir / f"{stem}_report.json"
    report = ProcessingReport(source=media.source, route=media.route, status="analyzed")

    if media.route == ProcessingRoute.TEXT_SOFT_SUBTITLE:
        subtitle_stream = choose_primary_text_subtitle(media)
        if subtitle_stream is None:
            report.status = "needs_hard_subtitle_pipeline"
            report.warnings.append("未找到可转换的文本字幕轨道。")
        else:
            subtitle_path = output_dir / f"{stem}.srt"
            clean_path = output_dir / f"{stem}_clean{media.source.suffix.lower()}"
            _require_new_outputs([subtitle_path, clean_path, report_path])
            extract_text_subtitle(media.source, subtitle_stream, subtitle_path)
            remux_without_subtitles(media.source, clean_path)
            report.subtitle_files.append(subtitle_path)
            report.clean_video = clean_path
            report.status = "completed"
    elif media.route == ProcessingRoute.BITMAP_SOFT_SUBTITLE:
        report.status = "needs_bitmap_ocr"
        report.warnings.append("检测到图像字幕轨道，阶段0尚未接入图像字幕OCR。")
    else:
        report.status = "needs_hard_subtitle_pipeline"
        report.warnings.append("未检测到软字幕轨道，需要进入硬字幕OCR与画面修复流程。")

    _require_new_outputs([report_path])
    write_text_atomically(
        report_path,
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
    )
    return report
