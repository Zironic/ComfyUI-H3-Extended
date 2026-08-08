$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$h3Root = (Resolve-Path (Join-Path $scriptRoot ".." )).Path
$comfyRoot = (Resolve-Path (Join-Path $h3Root "..\.." )).Path
$preflight = Join-Path $comfyRoot ".agents\skills\operate-comfy2-install\scripts\comfy_gpu_preflight.ps1"
& $preflight
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$python = Join-Path $comfyRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Comfy virtual environment Python was not found: $python"
}

$benchmark = Join-Path $scriptRoot "bench_convrot_mlp_c.py"
& $python $benchmark --i-understand-this-uses-gpu @args
exit $LASTEXITCODE
