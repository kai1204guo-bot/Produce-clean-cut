param(
    [switch]$SkipApplicationBuild
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
if (-not $SkipApplicationBuild) {
    & (Join-Path $projectRoot "build_windows.ps1")
    if ($LASTEXITCODE -ne 0) { throw "应用程序构建失败。" }
}

$compilerCandidates = @(
    (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6\ISCC.exe"),
    (Join-Path $env:LOCALAPPDATA "Programs\Inno Setup 6\ISCC.exe"),
    (Join-Path $env:ProgramFiles "Inno Setup 6\ISCC.exe")
)
$compiler = $compilerCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $compiler) {
    throw "未找到 Inno Setup 6。开发机可执行：winget install --id JRSoftware.InnoSetup --exact"
}

& $compiler (Join-Path $projectRoot "installer.iss")
if ($LASTEXITCODE -ne 0) { throw "安装包构建失败。" }
$result = Get-ChildItem -LiteralPath (Join-Path $projectRoot "installer-dist") `
    -Filter "清水版批量制作-安装程序-v*.exe" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $result) { throw "安装包编译结束，但没有找到版本化安装文件。" }
Write-Host "安装包已生成：$($result.FullName)"
