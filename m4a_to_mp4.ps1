# m4a_to_mp4.ps1
# powershell -ExecutionPolicy Bypass -File .\m4a_to_mp4.ps1 -Image "cover.jpg" -InputDir ".\music" -OutputDir ".\video_out"
param(
    [Parameter(Mandatory=$true)][string]$Image,
    [Parameter(Mandatory=$true)][string]$InputDir,
    [string]$OutputDir = ""
)
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Error "ffmpeg not found. Please install ffmpeg first."
    exit 1
}
if (-not (Test-Path $Image)) { Write-Error "Image not found: $Image"; exit 1 }
if (-not (Test-Path $InputDir)) { Write-Error "Directory not found: $InputDir"; exit 1 }
if ($OutputDir -eq "") { $OutputDir = Join-Path $InputDir "output" }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$count = 0
Get-ChildItem -Path $InputDir -Filter *.m4a | ForEach-Object {
    $out = Join-Path $OutputDir ($_.BaseName + ".mp4")
    Write-Host "Converting: $($_.Name) -> $out"
    ffmpeg -y -loop 1 -i $Image -i $_.FullName -shortest `
        -c:v libx264 -preset medium -pix_fmt yuv420p `
        -c:a aac -b:a 192k $out
    $count++
}
Write-Host "Done. $count file(s) converted. Output: $OutputDir"