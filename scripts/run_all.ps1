# 完整实验矩阵：4 个漂移协议 x 全部模型 x 3 个种子。
# 前置条件：已经按 data/README.md 下载好 TSB-AD 数据。
# 用法：pwsh scripts/run_all.ps1 [-Subsets "SMAP,MSL,SWaT,SMD"] [-Seeds 42,43,44]

param(
    [string]$Subsets = "SMAP,MSL,SWaT,SMD",
    [string]$Seeds = "42,43,44",
    [string]$Models = "zscore,adaptive_zscore,iforest,convae,ours,ours_tta"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $drifts = @("none", "temporal", "channel", "channel_mask", "amplitude")
    $seedList = $Seeds.Split(",")
    $modelList = $Models.Split(",")
    $subsetList = "[" + $Subsets + "]"

    foreach ($seed in $seedList) {
        foreach ($drift in $drifts) {
            foreach ($model in $modelList) {
                Write-Host "==> seed=$seed model=$model drift=$drift" -ForegroundColor Cyan
                python -m src.train `
                    -c configs/base.yaml `
                    -c "configs/model/$model.yaml" `
                    --drift $drift `
                    --seed $seed `
                    -o "data.subsets=$subsetList" `
                    --tag "s${seed}_${model}_${drift}"
                if ($LASTEXITCODE -ne 0) { throw "训练失败: seed=$seed model=$model drift=$drift" }
            }
        }
    }

    Write-Host "==> 汇总" -ForegroundColor Cyan
    python -m src.evaluate `
        --results results `
        --out results/summary.md `
        --metric vus_pr `
        --ours ours `
        --baseline zscore `
        --baseline adaptive_zscore `
        --baseline iforest `
        --baseline convae `
        --per-seed
}
finally {
    Pop-Location
}