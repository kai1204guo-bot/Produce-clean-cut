from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clean_cut.errors import CleanCutError
from clean_cut.hard_subtitles import analyze_hard_subtitles
from clean_cut.media import probe_media
from clean_cut.ocr import RapidOcrBackend
from clean_cut.pipeline import process_media
from clean_cut.subtitle_data import Region


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clean-cut",
        description="提取视频字幕并生成无字幕版本。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="分析媒体与字幕类型")
    inspect_parser.add_argument("source", type=Path, help="输入视频路径")

    process_parser = subparsers.add_parser("process", help="执行当前支持的处理流程")
    process_parser.add_argument("source", type=Path, help="输入视频路径")
    process_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="输出目录",
    )

    hard_parser = subparsers.add_parser("analyze-hard", help="分析固定区域硬字幕")
    hard_parser.add_argument("source", type=Path, help="输入视频路径")
    hard_parser.add_argument("--output-dir", type=Path, required=True, help="输出目录")
    hard_parser.add_argument(
        "--region",
        type=_parse_region,
        required=True,
        help="字幕区域，格式为x,y,width,height",
    )
    hard_parser.add_argument(
        "--interval-ms",
        type=int,
        default=250,
        help="OCR抽帧间隔，默认250毫秒",
    )
    return parser


def _parse_region(value: str) -> Region:
    try:
        x, y, width, height = (int(part.strip()) for part in value.split(","))
        return Region(x=x, y=y, width=width, height=height)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("区域格式必须为x,y,width,height") from exc


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = probe_media(args.source).to_dict()
        elif args.command == "process":
            result = process_media(args.source, args.output_dir).to_dict()
        else:
            backend = RapidOcrBackend()
            result = analyze_hard_subtitles(
                args.source,
                args.output_dir,
                region=args.region,
                backend=backend,
                interval_ms=args.interval_ms,
            ).to_dict()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except CleanCutError as exc:
        error = json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
