$ErrorActionPreference = "Stop"

Set-Location "E:\Fairness-aware\2CKG4ER\Code\codes-New-ConvE"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python .\run_fair_loss_sensitivity_experiments.py `
  --dataset Eedi `
  --run-prefix stage4_loss `
  --methods fairreg_item,fairreg_kc,fairreg_item_kc,fairreg_item_kc_hybrid_rerank `
  --seeds 2024 `
  --epochs 25 `
  --bs 1024 `
  --learning-rate 0.001 `
  --negative-ratio 5 `
  --cuda auto `
  --data-root "E:\Fairness-aware\2CKG4ER\Code\Data_Fin" `
  --runs-root "E:\Fairness-aware\2CKG4ER\Code\runs" `
  --source-graph-subdir er_graph `
  --fair-graph-subdir fair_kg_graph `
  --top-ks "5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100" `
  --alpha 1.0 `
  --gamma 0.25 `
  --experiments "random:1,random:10,random:50,random:100,mixed:50,mixed:100" `
  --fairness-candidate-size 150 `
  --fairness-temperature 1.0

if ($LASTEXITCODE -ne 0) {
  Write-Host "Stage-4 fair loss sensitivity experiments failed." -ForegroundColor Red
  exit $LASTEXITCODE
}

Write-Host "Stage-4 fair loss sensitivity experiments completed." -ForegroundColor Green
