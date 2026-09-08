# Produce Clean Cut

本地视频清水版处理工具。目标是从输入视频中提取字幕，并输出不带字幕的视频与对应的 SRT 文件。

项目目前处于阶段 0 技术验证。现已覆盖：

- 使用 ffprobe 分析视频、音频和字幕轨道；
- 区分文本软字幕、图像软字幕和可能的硬字幕；
- 将文本软字幕转换为 SRT；
- 无重编码移除所有软字幕轨道；
- 输出机器可读的任务报告；
- 使用合成测试视频验证完整软字幕链路。

硬字幕 OCR、字幕轨迹合并和画面修复将在后续阶段接入。

## 环境要求

- Windows 10/11；
- Python 3.11；
- `ffmpeg` 与 `ffprobe` 可从 `PATH` 访问。

## 开发运行

无需安装第三方 Python 依赖，可以直接从源码运行：

```powershell
$env:PYTHONPATH = "src"
python -m clean_cut inspect "input.mp4"
python -m clean_cut process "input.mkv" --output-dir "output"
```

`inspect` 只分析媒体；`process` 会根据字幕类型执行当前阶段支持的安全处理。文本软字幕将被提取为 SRT，同时生成移除字幕轨道后的清水视频。

完整技术规划见 [TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)。

