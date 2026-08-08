$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$h3Root = (Resolve-Path (Join-Path $scriptRoot ".." )).Path
$comfyRoot = (Resolve-Path (Join-Path $h3Root "..\.." )).Path
$preflight = Join-Path $comfyRoot ".agents\skills\operate-comfy2-install\scripts\comfy_gpu_preflight.ps1"
& $preflight
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$image = "nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c"
& docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Required local image is missing: $image (pull it explicitly before running this script)"
}

$mount = "{0}:/workspace/h3" -f $h3Root
& docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    --pull never --rm -v $mount $image `
    python /workspace/h3/benchmarks/bench_te_fp8_mlp.py `
    --i-understand-this-uses-gpu @args
exit $LASTEXITCODE
