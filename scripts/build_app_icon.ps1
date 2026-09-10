param(
    [string]$Source = (Join-Path $PSScriptRoot "..\assets\logo-source.png"),
    [string]$Icon = (Join-Path $PSScriptRoot "..\assets\app-icon.ico"),
    [string]$Header = (Join-Path $PSScriptRoot "..\assets\app-logo.png")
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

function New-SquareLogo([System.Drawing.Image]$image, [int]$size) {
    $bitmap = [System.Drawing.Bitmap]::new($size, $size)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.Clear([System.Drawing.Color]::White)
        $graphics.CompositingQuality = "HighQuality"
        $graphics.InterpolationMode = "HighQualityBicubic"
        $graphics.PixelOffsetMode = "HighQuality"
        $margin = [Math]::Max(1, [Math]::Round($size * 0.035))
        $available = $size - 2 * $margin
        $scale = [Math]::Min($available / $image.Width, $available / $image.Height)
        $width = [Math]::Max(1, [Math]::Round($image.Width * $scale))
        $height = [Math]::Max(1, [Math]::Round($image.Height * $scale))
        $x = [Math]::Floor(($size - $width) / 2)
        $y = [Math]::Floor(($size - $height) / 2)
        $graphics.DrawImage($image, $x, $y, $width, $height)
    }
    finally {
        $graphics.Dispose()
    }
    return $bitmap
}

$sourceImage = [System.Drawing.Image]::FromFile((Resolve-Path -LiteralPath $Source))
try {
    $headerBitmap = New-SquareLogo $sourceImage 64
    try {
        $headerBitmap.Save(
            [IO.Path]::GetFullPath($Header),
            [System.Drawing.Imaging.ImageFormat]::Png
        )
    }
    finally {
        $headerBitmap.Dispose()
    }

    $frames = @()
    foreach ($size in 16, 24, 32, 48, 64, 128, 256) {
        $bitmap = New-SquareLogo $sourceImage $size
        $stream = [IO.MemoryStream]::new()
        try {
            $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
            $frames += ,@($size, $stream.ToArray())
        }
        finally {
            $stream.Dispose()
            $bitmap.Dispose()
        }
    }
}
finally {
    $sourceImage.Dispose()
}

$iconPath = [IO.Path]::GetFullPath($Icon)
$file = [IO.File]::Open($iconPath, [IO.FileMode]::Create)
$writer = [IO.BinaryWriter]::new($file)
try {
    $writer.Write([UInt16]0)
    $writer.Write([UInt16]1)
    $writer.Write([UInt16]$frames.Count)
    $offset = 6 + 16 * $frames.Count
    foreach ($frame in $frames) {
        $size = [int]$frame[0]
        $bytes = [byte[]]$frame[1]
        $writer.Write([byte]($(if ($size -eq 256) { 0 } else { $size })))
        $writer.Write([byte]($(if ($size -eq 256) { 0 } else { $size })))
        $writer.Write([byte]0)
        $writer.Write([byte]0)
        $writer.Write([UInt16]1)
        $writer.Write([UInt16]32)
        $writer.Write([UInt32]$bytes.Length)
        $writer.Write([UInt32]$offset)
        $offset += $bytes.Length
    }
    foreach ($frame in $frames) {
        $writer.Write([byte[]]$frame[1])
    }
}
finally {
    $writer.Dispose()
    $file.Dispose()
}

Write-Host "应用图标已生成：$iconPath"
