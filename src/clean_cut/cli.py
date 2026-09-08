from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clean_cut.errors import CleanCutError
from clean_cut.media import probe_media
from clean_cut.pipeline import process_media


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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = probe_media(args.source).to_dict()
        else:
            result = process_media(args.source, args.output_dir).to_dict()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except CleanCutError as exc:
        error = json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
