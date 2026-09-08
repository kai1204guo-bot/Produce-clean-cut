from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from uuid import uuid4

from clean_cut.errors import CleanCutError
from clean_cut.inpaint import (
    FP32_LAMA_SHA256,
    FP32_LAMA_URL,
    OPENCV_LAMA_SHA256,
    OPENCV_LAMA_URL,
    sha256_file,
)


def download_lama_model(destination: Path, *, variant: str = "fp32") -> Path:
    variants = {
        "fp32": (FP32_LAMA_URL, FP32_LAMA_SHA256),
        "opencv-quantized": (OPENCV_LAMA_URL, OPENCV_LAMA_SHA256),
    }
    if variant not in variants:
        raise ValueError(f"未知LaMa模型版本：{variant}")
    model_url, expected_hash = variants[variant]
    destination = destination.resolve()
    if destination.is_file():
        if sha256_file(destination) == expected_hash:
            return destination
        raise CleanCutError(f"目标位置已有不同文件，未执行覆盖：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.download")
    try:
        request = urllib.request.Request(
            model_url,
            headers={"User-Agent": "Produce-Clean-Cut/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        actual_hash = sha256_file(temporary)
        if actual_hash != expected_hash:
            raise CleanCutError(
                f"下载的LaMa模型校验失败：期望{expected_hash}，实际{actual_hash}。"
            )
        os.replace(temporary, destination)
        return destination
    except OSError as exc:
        raise CleanCutError(f"LaMa模型下载失败：{exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)
