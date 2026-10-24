param(
    [string]$Python = 'C:\Users\21709\.conda\envs\rtx5060\python.exe',
    [string]$RunId = 'maed_v2_semantic_compgcn_seed2024_20261024_release',
    [ValidateSet('cuda', 'cpu')][string]$Device = 'cuda',
    [switch]$Plan
)

$ErrorActionPreference = 'Stop'
$repoPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoPath

if ($Plan) {
    & $Python -u .\scripts\run_maed_v2_training.py --stage plan --run-id $RunId --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'MAED-v2 plan failed' }
    return
}

& $Python -m unittest discover -s tests -p 'test_maed_v2*.py' -v
if ($LASTEXITCODE -ne 0) { throw 'MAED-v2 tests failed' }

$gatePath = Join-Path $repoPath "runs\$RunId\engineering_gate.json"
if (-not (Test-Path -LiteralPath $gatePath)) {
    & $Python -u .\scripts\run_maed_v2_training.py --stage preflight --run-id $RunId --device $Device
    if ($LASTEXITCODE -ne 0) { throw 'MAED-v2 engineering gate failed' }
}

# Python rechecks the gate/source/config identity before any training starts.
& $Python -u .\scripts\run_maed_v2_training.py --stage run --run-id $RunId --device $Device
if ($LASTEXITCODE -ne 0) { throw 'MAED-v2 interrupted; inspect stage.log, then rerun the same command' }
