from __future__ import annotations

import hashlib
import importlib.util
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from clean_cut.errors import CleanCutError
from clean_cut.masks import RenderedMask

OPENCV_LAMA_SHA256 = "7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2"
OPENCV_LAMA_URL = (
    "https://huggingface.co/opencv/inpainting_lama/resolve/main/"
    "inpainting_lama_2025jan.onnx"
)
FP32_LAMA_SHA256 = "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6"
FP32_LAMA_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
APPROVED_LAMA_HASHES = frozenset({OPENCV_LAMA_SHA256, FP32_LAMA_SHA256})


class InpaintBackend(ABC):
    @abstractmethod
    def inpaint(self, frame: Any, mask: Any) -> Any:
        """Return a full-frame BGR reconstruction for a binary uint8 mask."""


class OpenCvInpaintBackend(InpaintBackend):
    def __init__(self, radius: float = 3.0) -> None:
        if radius <= 0:
            raise ValueError("OpenCV修复半径必须大于零。")
        self.radius = radius

    def inpaint(self, frame: Any, mask: Any) -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise CleanCutError("OpenCV修复后端不可用。") from exc
        return cv2.inpaint(frame, mask, self.radius, cv2.INPAINT_TELEA)


class LamaDnnBackend(InpaintBackend):
    def __init__(self, model_path: Path, *, verify_hash: bool = True) -> None:
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise CleanCutError(f"LaMa模型不存在：{model_path}")
        model_hash = sha256_file(model_path)
        if verify_hash and model_hash not in APPROVED_LAMA_HASHES:
            raise CleanCutError("LaMa模型SHA-256不匹配，拒绝加载。")
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise CleanCutError("LaMa DNN后端需要OpenCV 5。") from exc
        self._cv2 = cv2
        try:
            self._network = cv2.dnn.readNetFromONNX(str(model_path))
        except cv2.error:
            model_buffer = np.frombuffer(model_path.read_bytes(), dtype=np.uint8)
            try:
                self._network = cv2.dnn.readNetFromONNX(model_buffer)
            except cv2.error as exc:
                raise CleanCutError(f"OpenCV无法加载LaMa模型：{model_path}") from exc

    def inpaint(self, frame: Any, mask: Any) -> Any:
        import numpy as np

        height, width = frame.shape[:2]
        image_blob = self._cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 255.0,
            size=(512, 512),
            mean=(0, 0, 0),
            swapRB=False,
            crop=False,
        )
        mask_blob = self._cv2.dnn.blobFromImage(
            mask,
            scalefactor=1.0,
            size=(512, 512),
            mean=(0,),
            swapRB=False,
            crop=False,
        )
        mask_blob = (mask_blob > 0).astype(np.float32)
        self._network.setInput(image_blob, "image")
        self._network.setInput(mask_blob, "mask")
        output = self._network.forward()[0]
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output, 0, 255).astype(np.uint8)
        return self._cv2.resize(output, (width, height), interpolation=self._cv2.INTER_CUBIC)


class LamaOnnxBackend(InpaintBackend):
    def __init__(
        self,
        model_path: Path,
        *,
        use_gpu: bool = False,
        verify_hash: bool = True,
    ) -> None:
        model_path = model_path.resolve()
        if not model_path.is_file():
            raise CleanCutError(f"LaMa模型不存在：{model_path}")
        model_hash = sha256_file(model_path)
        if verify_hash and model_hash not in APPROVED_LAMA_HASHES:
            raise CleanCutError("LaMa模型SHA-256不匹配，拒绝加载。")
        if use_gpu and model_hash == OPENCV_LAMA_SHA256:
            raise CleanCutError("OpenCV量化LaMa模型不支持当前CUDA后端，请使用FP32模型。")
        try:
            import cv2
            import onnxruntime as ort
        except ImportError as exc:
            raise CleanCutError("LaMa ONNX后端需要OpenCV与ONNX Runtime。") from exc

        self._dll_directory_handles = []
        nvidia_binary_dirs: list[str] = []
        if use_gpu and os.name == "nt":
            for package in (
                "nvidia.cublas",
                "nvidia.cuda_runtime",
                "nvidia.cudnn",
                "nvidia.cufft",
                "nvidia.curand",
                "nvidia.nvjitlink",
            ):
                spec = importlib.util.find_spec(package)
                locations = spec.submodule_search_locations if spec else None
                if not locations:
                    continue
                binary_dir = Path(next(iter(locations))) / "bin"
                if binary_dir.is_dir():
                    self._dll_directory_handles.append(os.add_dll_directory(binary_dir))
                    nvidia_binary_dirs.append(str(binary_dir))
            if nvidia_binary_dirs:
                os.environ["PATH"] = os.pathsep.join(
                    [*nvidia_binary_dirs, os.environ.get("PATH", "")]
                )
        if use_gpu and hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu
            else ["CPUExecutionProvider"]
        )
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=providers,
        )
        active_providers = self._session.get_providers()
        if use_gpu and "CUDAExecutionProvider" not in active_providers:
            raise CleanCutError("请求使用CUDA，但ONNX Runtime未能启用CUDA执行提供程序。")
        self._cv2 = cv2
        self.provider = active_providers[0]

    def inpaint(self, frame: Any, mask: Any) -> Any:
        import numpy as np

        height, width = frame.shape[:2]
        image = self._cv2.resize(frame, (512, 512), interpolation=self._cv2.INTER_LINEAR)
        image = image.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
        mask_input = self._cv2.resize(
            mask,
            (512, 512),
            interpolation=self._cv2.INTER_NEAREST,
        )
        mask_input = (mask_input[None, None, ...] > 0).astype(np.float32)
        output = self._session.run(None, {"image": image, "mask": mask_input})[0][0]
        output = np.transpose(output, (1, 2, 0))
        output = np.clip(output, 0, 255).astype(np.uint8)
        return self._cv2.resize(output, (width, height), interpolation=self._cv2.INTER_CUBIC)


def composite_repair(original: Any, repaired: Any, mask: RenderedMask) -> Any:
    import numpy as np

    alpha = mask.alpha.astype(np.float32)[:, :, None] / 255.0
    composed = original.astype(np.float32) * (1.0 - alpha) + repaired.astype(np.float32) * alpha
    return np.clip(composed, 0, 255).astype(np.uint8)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
