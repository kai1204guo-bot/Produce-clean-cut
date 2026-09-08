# Produce Clean Cut

本地视频清水版处理工具。目标是从输入视频中提取字幕，并输出不带字幕的视频与对应的 SRT 文件。

项目目前处于阶段 0 技术验证。现已覆盖：

- 使用 ffprobe 分析视频、音频和字幕轨道；
- 区分文本软字幕、图像软字幕和可能的硬字幕；
- 将文本软字幕转换为 SRT；
- 无重编码移除所有软字幕轨道；
- 使用 RapidOCR 分析用户指定区域内的硬字幕；
- 将跨帧OCR结果合并为字幕轨迹；
- 从同一轮OCR生成SRT和逐帧多边形遮罩计划；
- 输出机器可读的任务报告；
- 使用合成测试视频验证软字幕和硬字幕分析链路。

STTN/LaMa画面修复将在后续阶段接入。

## 环境要求

- Windows 10/11；
- Python 3.11；
- `ffmpeg` 与 `ffprobe` 可从 `PATH` 访问。

## 开发运行

基础媒体分析无需第三方 Python 依赖。硬字幕分析需要安装 OCR 可选依赖。
项目位于中文路径时，开发环境使用 `--no-install-project`，避免 Windows 可编辑安装路径的编码问题：

```powershell
uv sync --python 3.11 --extra ocr --extra dev --no-install-project
$env:PYTHONPATH = "src"
python -m clean_cut inspect "input.mp4"
python -m clean_cut process "input.mkv" --output-dir "output"
python -m clean_cut analyze-hard "input.mp4" --region "0,700,1920,300" --output-dir "output"
```

`inspect` 只分析媒体；`process` 会根据字幕类型执行当前阶段支持的安全处理。文本软字幕将被提取为 SRT，同时生成移除字幕轨道后的清水视频。

`analyze-hard` 按指定字幕区域抽帧并运行 RapidOCR，输出 SRT 和硬字幕计划 JSON。计划中的OCR文字轨迹与逐帧多边形遮罩来自同一轮识别，可供后续视频修复直接使用。

完整技术规划见 [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)。
