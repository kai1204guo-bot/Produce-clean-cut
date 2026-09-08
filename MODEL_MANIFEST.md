# 模型清单

模型文件不进入 Git，由 `clean-cut download-lama` 下载并在加载前校验 SHA-256。

| 变体 | 来源 | 许可证 | SHA-256 | 用途 |
| --- | --- | --- | --- | --- |
| `fp32` | [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX) | Apache-2.0 | `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6` | 默认；支持 ONNX Runtime CPU/CUDA |
| `opencv-quantized` | [OpenCV Zoo inpainting_lama](https://github.com/opencv/opencv_zoo/tree/main/models/inpainting_lama) | Apache-2.0 | `7df918ac3921d3daf0aae1d219776cf0dc4e4935f035af81841b40adcf74fdf2` | 较小的量化模型；当前仅用于 CPU |
| `sttn` | [researchmm/STTN](https://github.com/researchmm/STTN) | MIT | `25b0c2c30042d82efd1893bd42ec726764262d94115393a1718f8d65d2a7817b` | STTN 时序修复；下载镜像与官方 Google Drive 权重哈希一致 |

模型输入固定为 512×512。程序把原始画面与二值遮罩缩放后推理，再将结果缩回原分辨率，并只在羽化后的遮罩范围内合成。固定尺寸会影响细小纹理和超宽画面质量，因此当前实现定位为可运行基线，不代表最终长视频生产质量。

STTN 模型输入固定为 432×240。下载器使用 Hugging Face 镜像以支持普通 HTTPS 下载；开发验证时另从官方 README 指向的 Google Drive 下载权重，两份文件的 SHA-256 完全一致。
