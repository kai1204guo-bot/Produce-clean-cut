# Produce Clean Cut

本地视频清水版处理工具。目标是从输入视频中提取字幕，并输出不带字幕的视频与对应的 SRT 文件。

## Windows 桌面版

桌面版按剧集批量处理 LibTV 清水视频：默认每批上传 15 集，两批即可上传完
30 集；全部上传后逐集提交智能去字幕，随后并行等待、完成即下载，并自动恢复片头
封面。程序用独立的 Chrome 登录目录保存 LibTV 会话，不保存账号或密码。

开发环境启动：

```powershell
pip install -e ".[desktop]"
clean-cut-gui
```

首次使用先点击“登录 LibTV”，在打开的 Chrome 窗口完成登录。选择“成片”文件夹后，
程序默认把成品写入同级“清水版”文件夹，并通过 `.clean-cut-state.json` 记录每集状态。
处于 `submitting` 状态时不会自动重试，避免异常中断后重复扣积分。
桌面版还会通过已登录的官方 LibTV CLI 读取画布节点 ID，用于可靠地判断上传结果、
避免重复付费提交，并在画布缩放或移动后继续定位节点。

构建可双击运行的 Windows 程序：

```powershell
.\build_windows.ps1
```

生成目录为 `dist\清水版批量制作\`，主程序是 `清水版批量制作.exe`。

项目目前处于阶段 0 技术验证。现已覆盖：

- 使用 ffprobe 分析视频、音频和字幕轨道；
- 区分文本软字幕、图像软字幕和可能的硬字幕；
- 将文本软字幕转换为 SRT；
- 无重编码移除所有软字幕轨道；
- 使用 RapidOCR 分析用户指定区域内的硬字幕；
- 使用 Faster-Whisper 从视频音频批量生成 SRT，避免把画面标题和警戒带等文字误识别成对白；
- 将跨帧OCR结果合并为字幕轨迹，过滤异常大框和低证据单字符噪声；
- 从同一轮OCR生成SRT和逐帧多边形遮罩计划；
- 将遮罩扩张、羽化，并为字幕淡入淡出增加前后保持；
- 使用 OpenCV Telea、LaMa ONNX 或 STTN 修复硬字幕画面；
- STTN 使用带重叠上下文的短片分块和时序联合遮罩，降低长视频内存占用并避免从参考帧复制字幕残影；
- 检测硬切镜头并在切镜处重置 STTN 上下文，避免跨镜头污染；
- 通过 FFmpeg 输出 H.264 视频并复制原始音频；
- 输出时长、音频、遮罩误差、PSNR、时序残差和字幕残留质量报告；
- 输出机器可读的任务报告；
- 使用合成测试视频验证软字幕、硬字幕分析和视频修复链路。

当前修复实现是阶段 0 基线：仅接受恒定帧率（CFR）视频。OpenCV 后端无需模型、速度快但复杂背景质量有限；LaMa 适合单帧纹理修复；STTN 同时利用邻帧，作为运动镜头和时序一致性优先的后端。

## 环境要求

- Windows 10/11；
- Python 3.11；
- `ffmpeg` 与 `ffprobe` 可从 `PATH` 访问。

## 开发运行

基础媒体分析无需第三方 Python 依赖。硬字幕分析需要安装 OCR 可选依赖；只生成对白 SRT 时优先使用 ASR 可选依赖。
项目位于中文路径时，开发环境使用 `--no-install-project`，避免 Windows 可编辑安装路径的编码问题：

```powershell
uv sync --python 3.11 --extra ocr --extra repair --extra dev --no-install-project
$env:PYTHONPATH = "src"
python -m clean_cut inspect "input.mp4"
python -m clean_cut process "input.mkv" --output-dir "output"
python -m clean_cut asr-batch "input-directory" --output-dir "srt" --model turbo --device cuda --cuda-dll-dir ".venv/Lib/site-packages/torch/lib"
python -m clean_cut analyze-hard "input.mp4" --region "0,700,1920,300" --output-dir "output"
python -m clean_cut repair "input.mp4" --plan "output/input_hard_subtitle_plan.json" --output "output/input.clean.mp4"
python -m clean_cut detect-scenes "input.mp4"
python -m clean_cut evaluate "input.mp4" --repaired "output/input.clean.mp4" --plan "output/input_hard_subtitle_plan.json" --check-residual --output "output/input.quality.json"
```

`inspect` 只分析媒体；`process` 会根据字幕类型执行当前阶段支持的安全处理。文本软字幕将被提取为 SRT，同时生成移除字幕轨道后的清水视频。

`analyze-hard` 按指定字幕区域抽帧并运行 RapidOCR，输出 SRT 和硬字幕计划 JSON。计划中的OCR文字轨迹与逐帧多边形遮罩来自同一轮识别，可供后续视频修复直接使用。

## 批量语音字幕

视频对白字幕应优先从音频识别。`asr-batch` 会按剧集编号顺序处理目录内所有 MP4，并输出同名 SRT。NVIDIA 显卡建议使用 `turbo`、CUDA 和 `float16`；Windows 上若 CTranslate2 找不到 CUDA 12/cuDNN 9 DLL，可通过 `--cuda-dll-dir` 指向 PyTorch 的 `torch/lib` 目录。输出文件已存在时不会覆盖，避免误删人工校对结果。

```powershell
uv sync --python 3.11 --extra asr --extra dev --no-install-project
$env:PYTHONPATH = "src"
python -m clean_cut asr-batch "F:/videos" `
  --output-dir "F:/videos/srt" `
  --model turbo `
  --device cuda `
  --cuda-dll-dir ".venv/Lib/site-packages/torch/lib"
```

OCR 仍用于定位画面中的硬字幕遮罩。画面内存在招牌、封面标题或警戒带时，缩小 `--region` 并使用 `--min-observations 2` 可过滤只出现一帧的干扰文字。

## LaMa GPU 修复

Windows + NVIDIA 显卡可安装 CUDA 版 ONNX Runtime。依赖固定在 `<1.27`，对应 CUDA 12 系列，避免新版 wheel 切换到 CUDA 13 后造成环境不兼容：

```powershell
uv sync --python 3.11 --extra ocr-gpu --extra dev --no-install-project
$env:PYTHONPATH = "src"
python -m clean_cut download-lama ".clean-cut-models/lama_fp32.onnx"
python -m clean_cut repair "input.mp4" --plan "output/input_hard_subtitle_plan.json" --output "output/input.clean.mp4" --backend lama --model ".clean-cut-models/lama_fp32.onnx" --device cuda
```

下载器会校验 SHA-256；模型不会写入 Git。CPU 环境也可将 `--device` 改为 `cpu`，但逐帧 LaMa 会明显较慢。模型来源、许可证与固定哈希见 [MODEL_MANIFEST.md](MODEL_MANIFEST.md)。

输出文件已存在时程序会拒绝覆盖。修复前也会检查计划路径、画面尺寸和帧率类型，避免用错计划或让 VFR 视频产生字幕时间漂移。

## STTN 时序修复

STTN 使用现代 PyTorch 兼容实现，固定输入为 432×240，默认每 30 帧一块、保留 5 帧上下文重叠。CPU 可以运行，但实际视频建议使用 NVIDIA CUDA：

```powershell
uv sync --python 3.11 --extra ocr-gpu --extra repair --extra sttn --extra dev --no-install-project
uv pip install --python ".venv/Scripts/python.exe" --index-url "https://download.pytorch.org/whl/cu128" "torch==2.11.0+cu128" --reinstall
$env:PYTHONPATH = "src"
python -m clean_cut download-sttn ".clean-cut-models/sttn.pth"
python -m clean_cut repair "input.mp4" --plan "output/input_hard_subtitle_plan.json" --output "output/input.sttn.clean.mp4" --backend sttn --model ".clean-cut-models/sttn.pth" --device cuda
```

STTN 修复默认使用 `--scene-threshold 0.6` 检测硬切，并在切镜前刷新当前时序块。`detect-scenes` 可以在正式修复前单独查看切镜帧、时间和分数；素材误切较多时可适当调高阈值。

`evaluate` 不提供干净参考视频时检查结构完整性并报告遮罩区域帧间变化；技术基准可增加 `--reference-clean clean.mp4`，此时额外计算遮罩 MAE、PSNR 与时序残差。增加 `--check-residual` 后，只在原字幕活跃时间附近重新执行 OCR，并将与原字幕文字相似的检测结果标记为 `needs_review`。复检状态还包括 `passed`、`not_applicable` 和 `inconclusive`，避免在没有字幕条目或没有有效样本时误报“通过”。这能减少背景招牌误报，但仍需要人工复核。

## LibTV 智能去字幕

安装并登录官方 `libtv` CLI 后，可把 LibTV 的“智能去字幕”作为云端修复后端。由于 CLI 1.1.3 不能直接新建隐藏的擦除模型节点，需要先在目标画布中对任意视频手动执行一次“智能去字幕”，并把生成节点作为模板。后续视频会由程序自动上传、替换模板输入、同步等待处理完成并下载；模板画布不适合并发运行多个任务。

> 当前官方最新版 CLI 1.1.3 还存在隐藏模型执行缺陷：它能读取 `volcano-subtitle-eraser`，但 `--run` 会被 `supportModels.video` 白名单错误拒绝。程序会在上传前明确停止，避免产生无结果的批量上传；待官方修复版发布后，现有模板复用流程即可继续验证。

以下命令一次生成 SRT、硬字幕计划和 LibTV 清水视频。输出视频默认命名为 `原文件名-清水版.mp4`：

```powershell
python -m clean_cut libtv-process "input.mp4" `
  --output-dir "output" `
  --region "0,1150,1080,600" `
  --project "画布UUID" `
  --template-node "视频一键去字幕-35"
```

如果已经有 SRT，只运行云端修复：

```powershell
python -m clean_cut libtv-repair "input.mp4" `
  --output "output/input-清水版.mp4" `
  --project "画布UUID" `
  --template-node "视频一键去字幕-35"
```

LibTV 当前要求视频不少于 3 秒、最长边不超过 2K，支持 MP4、FLV、TS、AVI、MOV、MKV 和 WMV。生成服务可能产生积分或会员额度消耗；程序不会自动重复失败的任务。

程序会检查开头 15 帧中的首个明显切镜，自动把切镜前的原始封面帧恢复到清水视频，防止封面标题被误当字幕删除；没有检测到切镜时默认保护前 2 帧。可用 `--cover-frames 3` 明确指定帧数，或用 `--cover-frames 0` 关闭保护。

STTN 推理结构改编自 MIT 许可的官方实现，署名和许可全文见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

完整技术规划见 [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)。
