param(
    [string]$DataPath = (Join-Path $PSScriptRoot "patchcore-inspection\data\mvtec"),
    [string[]]$Category = @("bottle", "pill", "transistor")
)

$ErrorActionPreference = "Stop"
$implementation = Join-Path $PSScriptRoot "patchcore-inspection"
$python = Join-Path $implementation ".venv\Scripts\python.exe"
$resultRoot = Join-Path $implementation "results"
$outputRoot = Join-Path $PSScriptRoot "results\patchcore_visualizations"

if (-not (Test-Path -LiteralPath $python)) { throw "가상환경이 없음: $python" }
if (-not (Test-Path -LiteralPath $DataPath)) { throw "MVTec AD 경로가 없음: $DataPath" }

Set-Location $implementation
$env:PYTHONPATH = Join-Path $implementation "src"

foreach ($item in $Category) {
    foreach ($className in ($item -split ',')) {
    $className = $className.Trim()
    if ([string]::IsNullOrWhiteSpace($className)) { continue }
    $modelDirectory = Get-ChildItem -LiteralPath $resultRoot -Recurse -Directory -Filter "mvtec_$className" |
        Where-Object { $_.FullName -match "IM224_WR50_L2-3_P01" } |
        Select-Object -First 1
    if (-not $modelDirectory) { throw "저장 모델을 찾을 수 없음: $className" }

    $savePath = Join-Path $outputRoot $className
    & $python bin\load_and_evaluate_patchcore.py --gpu 0 --seed 0 --save_segmentation_images $savePath `
        patch_core_loader -p $modelDirectory.FullName `
        dataset --resize 256 --imagesize 224 -d $className mvtec $DataPath
    if ($LASTEXITCODE -ne 0) { throw "시각화 생성 실패: $className" }
    }
}
