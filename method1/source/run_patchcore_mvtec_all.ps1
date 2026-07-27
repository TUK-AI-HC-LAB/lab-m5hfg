param(
    [string]$DataPath = (Join-Path $PSScriptRoot "patchcore-inspection\data\mvtec"),
    [ValidateSet("224", "320")]
    [string]$ImageSize = "224",
    [ValidateSet("0.01", "0.1")]
    [string]$CoresetRatio = "0.1",
    [string[]]$Category = @(
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
        "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
    )
)

$ErrorActionPreference = "Stop"
$runner = Join-Path $PSScriptRoot "run_patchcore_mvtec.ps1"
foreach ($item in $Category) {
    foreach ($category in ($item -split ',')) {
        $category = $category.Trim()
        if ([string]::IsNullOrWhiteSpace($category)) { continue }
    Write-Host "`n===== Running PatchCore: $category =====`n"
    & $runner -DataPath $DataPath -Category $category -ImageSize $ImageSize -CoresetRatio $CoresetRatio
    if ($LASTEXITCODE -ne 0) {
        throw "PatchCore failed for category: $category"
    }
    }
}

Write-Host "`nAll 15 MVTec AD categories completed."
