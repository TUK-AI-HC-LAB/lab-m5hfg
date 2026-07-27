param(
    [string]$DataPath = (Join-Path $PSScriptRoot "patchcore-inspection\data\mvtec"),
    [string[]]$Category = @("bottle"),
    [ValidateSet("224", "320")]
    [string]$ImageSize = "224",
    [ValidateSet("0.01", "0.1")]
    [string]$CoresetRatio = "0.1",
    [string]$RunName = ""
)

$ErrorActionPreference = "Stop"
$implementation = Join-Path $PSScriptRoot "patchcore-inspection"
$python = Join-Path $implementation ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "가상환경이 없음: $python"
}
if (-not (Test-Path -LiteralPath $DataPath)) {
    throw "MVTec AD 경로가 없음: $DataPath"
}

Set-Location $implementation
$env:PYTHONPATH = (Join-Path $implementation "src")
$categories = $Category |
    ForEach-Object { $_ -split "," } |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ }
$ratioLabel = if ($CoresetRatio -eq "0.1") { "P01" } else { "P001" }
$defaultRunName = "IM{0}_WR50_L2-3_{1}_D1024-1024_PS-3_AN-1_S0_{2}" -f $ImageSize, $ratioLabel, ($categories -join "_")
if ([string]::IsNullOrWhiteSpace($RunName)) {
    $RunName = $defaultRunName
}
$resultPath = Join-Path $implementation ("results\{0}" -f $RunName)
$datasetFlags = foreach ($item in $categories) { @("-d", $item) }

# 논문/공식 예제의 image-level detection 설정.
# Windows에서는 faiss-gpu 패키지를 제공하지 않아 FAISS 검색만 CPU에서 수행함.
& $python bin\run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model $resultPath `
    patch_core -b wideresnet50 -le layer2 -le layer3 `
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 `
    --anomaly_scorer_num_nn 1 --patchsize 3 `
    sampler -p $CoresetRatio approx_greedy_coreset `
    dataset --resize $(if ($ImageSize -eq "224") { 256 } else { 366 }) --imagesize $ImageSize `
    $datasetFlags mvtec $DataPath
