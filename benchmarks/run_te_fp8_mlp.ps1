$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$h3Root = (Resolve-Path (Join-Path $scriptRoot ".." )).Path
$comfyRoot = (Resolve-Path (Join-Path $h3Root "..\.." )).Path
$preflight = Join-Path $comfyRoot ".agents\skills\operate-comfy2-install\scripts\comfy_gpu_preflight.ps1"
& $preflight
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$image = "h3-te-fp8-convrot:ck-0.2.28"
& docker image inspect $image *> $null
if ($LASTEXITCODE -ne 0) {
    $dockerfile = Join-Path $scriptRoot "Dockerfile.te-fp8-convrot"
    & docker build -t $image -f $dockerfile $scriptRoot
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

$mount = "{0}:/workspace/h3" -f $h3Root
$models = "D:\AI\ComfyUI\Models:/models:ro"
& docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 `
    --pull never --rm -v $mount -v $models $image `
    python /workspace/h3/benchmarks/bench_te_fp8_mlp.py `
    --i-understand-this-uses-gpu @args
exit $LASTEXITCODE
