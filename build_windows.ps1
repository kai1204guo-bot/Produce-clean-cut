param(
    [string]$OutputDirectory = "dist"
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$buildRoot = Join-Path ([IO.Path]::GetPathRoot($projectRoot)) "ProduceCleanCutBuild"
$temporaryRoot = Join-Path $buildRoot "tmp"
$venv = Join-Path $buildRoot "venv"
$python = Join-Path $venv "Scripts\python.exe"
$pythonBase = py -3.13 -c "import sys; print(sys.base_prefix)"
if (-not (Test-Path -LiteralPath (Join-Path $pythonBase "DLLs\_tkinter.pyd"))) {
    throw "Python 3.13 缺少 Tkinter，无法构建桌面界面。"
}

New-Item -ItemType Directory -Force -Path $buildRoot, $temporaryRoot | Out-Null
$env:TEMP = $temporaryRoot
$env:TMP = $temporaryRoot
if (-not (Test-Path -LiteralPath $python)) {
    py -3.13 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "创建构建环境失败。" }
}

& $python -m pip install "$projectRoot[desktop]" "pyinstaller>=6.15,<7"
if ($LASTEXITCODE -ne 0) { throw "安装桌面版构建依赖失败。" }
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --icon (Join-Path $projectRoot "assets\app-icon.ico") `
    --name "清水版批量制作" `
    --distpath (Join-Path $projectRoot $OutputDirectory) `
    --workpath (Join-Path $buildRoot "work") `
    --specpath $buildRoot `
    --hidden-import tkinter `
    --add-data "$(Join-Path $pythonBase 'Lib\tkinter');tkinter" `
    --add-binary "$(Join-Path $pythonBase 'DLLs\_tkinter.pyd');." `
    --add-binary "$(Join-Path $pythonBase 'DLLs\tcl86t.dll');." `
    --add-binary "$(Join-Path $pythonBase 'DLLs\tk86t.dll');." `
    --add-data "$(Join-Path $pythonBase 'tcl\tcl8.6');_tcl_data" `
    --add-data "$(Join-Path $pythonBase 'tcl\tk8.6');_tk_data" `
    --runtime-hook (Join-Path $projectRoot "src\clean_cut\pyi_rth_tk.py") `
    --add-data "$(Join-Path $projectRoot 'assets');assets" `
    --collect-all playwright `
    --collect-all faster_whisper `
    --collect-all ctranslate2 `
    --collect-all tokenizers `
    (Join-Path $projectRoot "src\clean_cut\desktop.py")
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 构建失败。" }

Write-Host "构建完成：$(Join-Path $projectRoot $OutputDirectory)"
