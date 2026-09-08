from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clean_cut.errors import CleanCutError
from clean_cut.hard_subtitles import analyze_hard_subtitles
from clean_cut.inpaint import LamaOnnxBackend, OpenCvInpaintBackend, SttnBackend
from clean_cut.masks import MaskRenderConfig
from clean_cut.media import probe_media
from clean_cut.model_store import download_lama_model, download_sttn_model
from clean_cut.ocr import RapidOcrBackend
from clean_cut.pipeline import process_media
from clean_cut.quality import evaluate_repair
from clean_cut.residuals import scan_residual_subtitles
from clean_cut.scene_detection import detect_scene_cuts
from clean_cut.subtitle_data import Region
from clean_cut.tools import write_text_atomically
from clean_cut.video_repair import load_hard_subtitle_plan, repair_video


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

    repair_parser = subparsers.add_parser("repair", help="根据硬字幕计划修复视频")
    repair_parser.add_argument("source", type=Path, help="输入视频路径")
    repair_parser.add_argument("--plan", type=Path, required=True, help="硬字幕计划JSON")
    repair_parser.add_argument("--output", type=Path, required=True, help="清水视频输出路径")
    repair_parser.add_argument(
        "--backend",
        choices=("opencv", "lama", "sttn"),
        default="opencv",
        help="修复后端，默认opencv",
    )
    repair_parser.add_argument("--model", type=Path, help="LaMa ONNX模型路径")
    repair_parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="LaMa运行设备，默认cpu",
    )
    repair_parser.add_argument("--dilation", type=int, default=8, help="遮罩扩张像素")
    repair_parser.add_argument("--feather", type=int, default=4, help="遮罩羽化像素")
    repair_parser.add_argument("--hold-before-ms", type=int, default=120)
    repair_parser.add_argument("--hold-after-ms", type=int, default=120)
    repair_parser.add_argument("--crf", type=int, default=18, help="H.264输出CRF")
    repair_parser.add_argument("--temporal-chunk-frames", type=int, default=30)
    repair_parser.add_argument("--temporal-overlap-frames", type=int, default=5)
    repair_parser.add_argument(
        "--scene-threshold",
        type=float,
        default=0.6,
        help="STTN场景切换阈值，默认0.6",
    )

    model_parser = subparsers.add_parser("download-lama", help="下载并校验OpenCV LaMa模型")
    model_parser.add_argument("destination", type=Path, help="模型保存路径")
    model_parser.add_argument(
        "--variant",
        choices=("fp32", "opencv-quantized"),
        default="fp32",
        help="默认下载支持CUDA的FP32版本",
    )

    sttn_model_parser = subparsers.add_parser("download-sttn", help="下载并校验STTN模型")
    sttn_model_parser.add_argument("destination", type=Path, help="模型保存路径")

    quality_parser = subparsers.add_parser("evaluate", help="评测修复视频质量")
    quality_parser.add_argument("source", type=Path, help="带字幕的输入视频")
    quality_parser.add_argument("--repaired", type=Path, required=True, help="修复后视频")
    quality_parser.add_argument("--plan", type=Path, required=True, help="硬字幕计划JSON")
    quality_parser.add_argument("--reference-clean", type=Path, help="可选的无字幕参考视频")
    quality_parser.add_argument("--output", type=Path, help="可选的质量报告JSON输出")
    quality_parser.add_argument(
        "--check-residual",
        action="store_true",
        help="对原字幕时段执行OCR残留复检",
    )
    quality_parser.add_argument("--ocr-score", type=float, default=0.5)
    quality_parser.add_argument("--residual-similarity", type=float, default=0.45)
    quality_parser.add_argument("--residual-interval-ms", type=int)

    scenes_parser = subparsers.add_parser("detect-scenes", help="检测视频场景切换")
    scenes_parser.add_argument("source", type=Path, help="输入视频路径")
    scenes_parser.add_argument("--threshold", type=float, default=0.6)
    scenes_parser.add_argument("--min-interval-frames", type=int, default=3)
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
        elif args.command == "analyze-hard":
            backend = RapidOcrBackend()
            result = analyze_hard_subtitles(
                args.source,
                args.output_dir,
                region=args.region,
                backend=backend,
                interval_ms=args.interval_ms,
            ).to_dict()
        elif args.command == "repair":
            plan = load_hard_subtitle_plan(args.plan)
            if args.backend == "lama":
                if args.model is None:
                    parser.error("使用lama后端时必须提供--model")
                repair_backend = LamaOnnxBackend(
                    args.model,
                    use_gpu=args.device == "cuda",
                )
            elif args.backend == "sttn":
                if args.model is None:
                    parser.error("使用sttn后端时必须提供--model")
                repair_backend = SttnBackend(
                    args.model,
                    use_gpu=args.device == "cuda",
                )
            else:
                repair_backend = OpenCvInpaintBackend()
            result = {
                "output": str(
                    repair_video(
                        args.source,
                        args.output,
                        plan,
                        repair_backend,
                        mask_config=MaskRenderConfig(
                            dilation_px=args.dilation,
                            feather_px=args.feather,
                            hold_before_ms=args.hold_before_ms,
                            hold_after_ms=args.hold_after_ms,
                        ),
                        crf=args.crf,
                        temporal_chunk_frames=args.temporal_chunk_frames,
                        temporal_overlap_frames=args.temporal_overlap_frames,
                        scene_threshold=args.scene_threshold,
                    )
                ),
                "backend": args.backend,
                "status": "completed",
            }
        elif args.command == "download-lama":
            result = {
                "model": str(
                    download_lama_model(args.destination, variant=args.variant)
                ),
                "status": "completed",
            }
        elif args.command == "download-sttn":
            result = {
                "model": str(download_sttn_model(args.destination)),
                "status": "completed",
            }
        elif args.command == "evaluate":
            if args.output is not None and args.output.exists():
                raise CleanCutError(f"质量报告已存在，未执行覆盖：{args.output.resolve()}")
            plan = load_hard_subtitle_plan(args.plan)
            quality_report = evaluate_repair(
                args.source,
                args.repaired,
                plan,
                reference_clean=args.reference_clean,
            )
            if args.check_residual:
                residual_report = scan_residual_subtitles(
                    args.repaired,
                    plan,
                    RapidOcrBackend(text_score=args.ocr_score),
                    interval_ms=args.residual_interval_ms,
                    min_source_similarity=args.residual_similarity,
                )
                quality_report.residual_check_status = residual_report.status
                quality_report.residual_intervals = [
                    {
                        "start_ms": item.start_ms,
                        "end_ms": item.end_ms,
                        "detected_text": item.detected_text,
                        "confidence": item.confidence,
                        "source_match": item.source_match,
                        "sample_count": item.sample_count,
                    }
                    for item in residual_report.intervals
                ]
                if residual_report.status == "needs_review":
                    quality_report.warnings.append(
                        f"检测到{len(residual_report.intervals)}个疑似字幕残留区间。"
                    )
                elif residual_report.status == "not_applicable":
                    quality_report.warnings.append("原字幕计划没有可用于残留比对的字幕条目。")
                elif residual_report.status == "inconclusive":
                    quality_report.warnings.append("没有抽取到处于原字幕时段的复检样本。")
            result = quality_report.to_dict()
            if args.output is not None:
                write_text_atomically(
                    args.output,
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                )
        else:
            cuts = detect_scene_cuts(
                args.source,
                threshold=args.threshold,
                min_interval_frames=args.min_interval_frames,
            )
            result = {
                "source": str(args.source.resolve()),
                "scene_count": len(cuts) + 1,
                "cuts": [cut.to_dict() for cut in cuts],
            }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (CleanCutError, ValueError) as exc:
        error = json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False)
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
