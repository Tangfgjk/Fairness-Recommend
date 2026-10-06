param(
    [string]$RunId = (Get-Date -Format 'yyyyMMdd-HHmmss'),
    [switch]$Resume,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Py = (Get-Command python -ErrorAction Stop).Source
$DataRoot = Join-Path $Root 'Data_Fin_New'
$MainRunsRoot = Join-Path $Root 'runs_Data_Fin_New_main'
$ComparisonRunsRoot = Join-Path $Root 'runs_Data_Fin_New_comparison'
$AblationRoot = Join-Path $Root 'ablation_data_Data_Fin_New'
$TopKs = '5,10,15,20,25,30,35,40,45,50'
$EpThresholds = '0.4,0.5,0.6,0.7'
$Datasets = @('Eedi', 'algebra2005', 'XES3G5M-sub-small')
$Seeds = @('2024', '2025', '2026')

if (-not (Test-Path -LiteralPath $Py)) { throw "Python environment missing: $Py" }
& $Py (Join-Path $PSScriptRoot 'check_data_fin_new.py')
if ($LASTEXITCODE -ne 0) { throw 'Data_Fin_New preflight failed' }
if ($PreflightOnly) { return }

foreach ($Dataset in $Datasets) {
    $MainRun = Join-Path (Join-Path $MainRunsRoot $Dataset) $RunId
    $ComparisonRun = Join-Path (Join-Path $ComparisonRunsRoot $Dataset) $RunId
    for ($Index = 0; $Index -lt $Seeds.Count; $Index++) {
        $Seed = $Seeds[$Index]
        $ResumeSeed = $Resume -or ($Index -gt 0)
        Write-Host "[$Dataset] seed ${Seed}: training main and ablation models"
        $Main = @(
            (Join-Path $Root 'codes-New-ConvE\run_semantic_experiments.py'),
            '--dataset', $Dataset, '--data-root', $DataRoot, '--graph-subdir', 'er_graph',
            '--runs-root', $MainRunsRoot, '--ablation-data-root', $AblationRoot,
            '--run-id', $RunId, '--seeds', $Seed, '--ablations', 'all',
            '--top-ks', $TopKs, '--ep-thresholds', $EpThresholds, '--cuda', 'true'
        )
        if ($ResumeSeed) { $Main += '--resume' }
        & $Py @Main
        if ($LASTEXITCODE -ne 0) { throw "Main/ablation training failed: $Dataset seed $Seed" }
        & $Py (Join-Path $Root 'scripts\summarize_paper_tables.py') `
            --dataset $Dataset --main-run-dir $MainRun --comparison-run-dir $ComparisonRun `
            --output-dir (Join-Path $MainRun 'summaries\paper_tables')
        if ($LASTEXITCODE -ne 0) { throw "Partial paper table update failed: $Dataset seed $Seed" }

        Write-Host "[$Dataset] seed ${Seed}: training comparison models"
        $Comparison = @(
            (Join-Path $Root 'comparison_models\run_v9_comparison_experiments.py'),
            '--dataset', $Dataset, '--data-root', $DataRoot, '--graph-subdir', 'er_graph',
            '--run-root', $ComparisonRunsRoot, '--run-id', $RunId, '--seeds', $Seed,
            '--top-ks', $TopKs, '--ep-thresholds', $EpThresholds, '--cuda', 'true'
        )
        if ($ResumeSeed) { $Comparison += '--resume' }
        & $Py @Comparison
        if ($LASTEXITCODE -ne 0) { throw "Comparison training failed: $Dataset seed $Seed" }

        & $Py (Join-Path $Root 'codes-New-ConvE\summarize_semantic_results.py') `
            --dataset $Dataset --run-id $RunId --runs-root $MainRunsRoot `
            --seeds '2024,2025,2026' --ablations all --top-ks $TopKs
        if ($LASTEXITCODE -ne 0) { throw "Ablation summary failed: $Dataset seed $Seed" }
        & $Py (Join-Path $Root 'scripts\report_topk_comparison.py') `
            --run-dir "ablation=$MainRun" --run-dir "comparison=$ComparisonRun" `
            --top-ks $TopKs --output-dir (Join-Path $MainRun 'summaries\combined')
        if ($LASTEXITCODE -ne 0) { throw "Combined summary failed: $Dataset seed $Seed" }
        & $Py (Join-Path $Root 'scripts\summarize_paper_tables.py') `
            --dataset $Dataset --main-run-dir $MainRun --comparison-run-dir $ComparisonRun `
            --output-dir (Join-Path $MainRun 'summaries\paper_tables')
        if ($LASTEXITCODE -ne 0) { throw "Paper table generation failed: $Dataset seed $Seed" }
        Write-Host "[$Dataset] seed ${Seed}: paper tables updated"
    }
}
