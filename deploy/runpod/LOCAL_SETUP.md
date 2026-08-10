# Local RunPod control

The local machine has two separate responsibilities:

1. configure the reusable Serverless endpoint once, and
2. submit jobs to that endpoint whenever inference is wanted.

The local machine does **not** create and destroy a Pod for every generation. The endpoint is configured with `workers-min=0`; submitting a queued job is what causes RunPod to allocate a matching worker. When the worker becomes idle, RunPod scales it back to zero.

## 1. Install and authenticate `runpodctl`

Use RunPod's current CLI and configure the API key once:

```bash
runpodctl config --apiKey YOUR_RUNPOD_API_KEY
runpodctl version
```

`setup_endpoint.py` shells out to `runpodctl` because RunPod's current CLI exposes the cached-model attachment directly through `serverless create --model-reference`.

## 2. Create the reusable H3 endpoint

From this repository:

```bash
python deploy/runpod/setup_endpoint.py
```

The default configuration creates a private Serverless template using the stock image:

```text
runpod/worker-comfyui:5.8.6-base-cuda12.8.1
```

and a queue-based endpoint configured as:

```text
GPU profile: rtx4090
GPU: NVIDIA GeForce RTX 4090 (SM89, 24 GB)
GPU count: 1
workers min: 0
workers max: 1
idle timeout: 5 s
execution timeout: 7200 s
minimum CUDA: 12.8
FlashBoot: enabled
network volume: none
cached model: https://huggingface.co/Comfy-Org/MiniMax-H3:main
```

The template's start command downloads `bootstrap.sh` from this repository. The bootstrap updates ComfyUI, installs H3-Extended into `custom_nodes`, detects the worker's CUDA architecture, compiles the matching pinned SpargeAttention extension in the stock `/opt/venv` when the marker/ABI preflight does not prove it is present, links the RunPod Cached Model files into Comfy's model directories, installs the H3-aware handler, and then hands control to the stock worker `/start.sh`. No custom image is required; each newly allocated worker may pay the CUDA 12.8 compiler install and source-build cost.

The setup helper is idempotent by endpoint name. If `h3-extended-rtx4090` already exists, it prints the existing endpoint instead of creating another one. Each named GPU profile gets its own default endpoint and template name, so changing profiles is explicit and does not alter an existing endpoint. Use `--force-new-endpoint` only when a genuinely separate endpoint with the same name is wanted.

The final line printed by the helper is suitable for the shell:

```bash
export RUNPOD_ENDPOINT_ID=...
```

Useful overrides include:

```bash
python deploy/runpod/setup_endpoint.py --list-gpu-profiles
python deploy/runpod/setup_endpoint.py --gpu-profile l40s

python deploy/runpod/setup_endpoint.py \
  --h3-ref <commit-or-branch> \
  --comfy-ref v0.31.0 \
  --gpu "NVIDIA Future GPU" \
  --model-reference "https://huggingface.co/Comfy-Org/MiniMax-H3:main"
```

The named profiles are `rtx4090` (default), `rtx5090`, `rtx6000-ada`, `l40s`, and `a100-80gb`. A raw `--gpu` value lets the same deployment path target another exact RunPod GPU ID; the bootstrap still checks the actual runtime capability. The currently supported sparse architectures are SM80, SM86, SM87, SM89, SM90, and SM120. B200/SM100 is intentionally rejected until the H3 backend supports it.

Once the deployment is validated, prefer an immutable H3-Extended commit SHA for `--h3-ref` rather than the moving experimental branch.

## 3. Submit a job

Set the API key for the HTTP job client and export the endpoint ID returned above:

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
```

Then submit a Comfy API-format workflow:

```bash
python deploy/runpod/run_h3.py workflow_api_export.json
```

`run_h3.py` walks backward from connected `SaveVideo` nodes, reads the filenames saved in connected image/audio/video loaders, resolves them beneath the local Comfy input directory, and uploads them inline. Unconnected loader nodes are ignored. It also keeps `MiniMaxH3HybridSparseAttentionZi` enabled, replaces only the custom preview sampler, bypasses the local Ziro image scaler, and applies the cached-model and inline-I/O placeholders. The worker selects fused QKV mode on SM89 and portable `sage128` on other supported architectures.

The input directory is normally inferred from a workflow stored below `ComfyUI/User`. Use `COMFY_INPUT_DIR` or `--input-root` for a different layout. The optional `--input [NAME=]PATH` argument overrides one of the connected saved inputs; it is not required for the normal path.

The `run_h3.py` operation creates demand on the endpoint. If no worker is alive, RunPod schedules the GPU type configured on that endpoint. Because the endpoint has the H3 Hugging Face repository attached as a cached-model reference, RunPod selects a host with that cache when possible; otherwise RunPod populates the cache before the worker starts. The worker then runs the bootstrap (including a compiler install and architecture-specific source build when that worker has no valid marker), links the cached files, launches ComfyUI, executes the workflow, returns artifacts inline, becomes idle, and scales back to zero.

The 4090 default is a pricing choice, not a proven capacity claim. A live cold-worker run must still show that this workflow fits its 24 GB and that bootstrap plus inference complete successfully. If it does not fit, select a larger profile without changing the workflow runner.

The recurring local workflow is therefore only:

```text
workflow + local input files
        |
        v
run_h3.py
        |
        v
RunPod endpoint queue
        |
        v
worker scales 0 -> 1
        |
        v
H3 inference
        |
        v
worker scales 1 -> 0
```

There is no explicit local `rent_gpu()` or `stop_gpu()` call in the Serverless path. Inputs and outputs are inline, so the endpoint also needs no S3 credentials or persistent volume.
