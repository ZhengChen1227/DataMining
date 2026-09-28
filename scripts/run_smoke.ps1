# 冒烟测试：合成数据 + 全部漂移协议 + 4 个方法。
# 不需要下载真实数据，CPU 也能在几分钟内跑完。
# 用法：pwsh scripts/run_smoke.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $models = @("zscore", "adaptive_zscore", "convae", "ours")
    $drifts = @("none", "temporal", "channel", "amplitude")

    foreach ($drift in $drifts) {
        foreach ($model in $models) {
            Write-Host "==> model=$model drift=$drift" -ForegroundColor Cyan
            python -m src.train `
                -c configs/base.yaml `
                -c "configs/model/$model.yaml" `
                --drift $drift `
                -o data.name=synthetic `
                --tag "smoke_${model}_${drift}"
            if ($LASTEXITCODE -ne 0) { throw "训练失败: model=$model drift=$drift" }
        }
    }

    Write-Host "==> 汇总" -ForegroundColor Cyan
    python -m src.evaluate `
        --results results `
        --out results/summary_smoke.md `
        --metric vus_pr `
        --ours ours `
        --baseline zscore `
        --baseline adaptive_zscore `
        --baseline convae
}
finally {
    Pop-Location
}