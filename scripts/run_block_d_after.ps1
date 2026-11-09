# Wait until the given UTC time, then run block D (Sonnet 5.5) on the big-product tier.
# Used because claude-sonnet-5-5 access was suspended for the org until 2026-11-09T05:07:05Z.
param([string]$NotBeforeUtc = "2026-11-09T05:08:00Z")
$target = [DateTime]::Parse($NotBeforeUtc).ToUniversalTime()
while ((Get-Date).ToUniversalTime() -lt $target) { Start-Sleep -Seconds 30 }
uv run python -m badapple.run_grid --only D --n-steps 6 --tier bigmul --max-cost 35 --concurrency 16
