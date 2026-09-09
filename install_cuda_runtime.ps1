param(
    [string]$TargetDirectory = "$env:LOCALAPPDATA\ProduceCleanCut\cuda-runtime"
)

$ErrorActionPreference = "Stop"
& py -3.13 -m pip install --upgrade --target $TargetDirectory `
    "nvidia-cuda-runtime-cu12==12.8.90" `
    "nvidia-cublas-cu12==12.8.4.1" `
    "nvidia-cudnn-cu12==9.10.2.21"
if ($LASTEXITCODE -ne 0) {
    throw "NVIDIA CUDA 运行库安装失败。"
}

Write-Host "CUDA 运行库安装完成：$TargetDirectory"
