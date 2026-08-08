$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$h3Root = (Resolve-Path (Join-Path $scriptRoot ".." )).Path
$comfyRoot = (Resolve-Path (Join-Path $h3Root "..\.." )).Path
$preflight = Join-Path $comfyRoot ".agents\skills\operate-comfy2-install\scripts\comfy_gpu_preflight.ps1"
& $preflight
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$image = "nvcr.io/nvidia/pytorch:26.07-py3"
& docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Required local image is missing: $image (pull it explicitly before running this script)"
}

$mount = "{0}:/workspace/h3" -f $h3Root
& docker run --gpus all --ipc=host --rm -v $mount $image `
    python /workspace/h3/benchmarks/bench_te_fp8_mlp.py `
    --i-understand-this-uses-gpu @args
exit $LASTEXITCODE
