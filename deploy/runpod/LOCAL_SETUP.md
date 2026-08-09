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

The default configuration creates a private Serverless template using:

```text
runpod/worker-comfyui:5.8.6-base
```

and a queue-based endpoint configured as:

```text
GPU: NVIDIA RTX 6000 Ada Generation
GPU count: 1
workers min: 0
workers max: 1
idle timeout: 5 s
execution timeout: 7200 s
minimum CUDA: 12.6
FlashBoot: enabled
network volume: none
cached model: https://huggingface.co/Comfy-Org/MiniMax-H3:main
```

The template's start command downloads `bootstrap.sh` from this repository. The bootstrap updates ComfyUI, installs H3-Extended into `custom_nodes`, links the RunPod Cached Model files into Comfy's model directories, installs the H3-aware handler, and then hands control to the stock worker `/start.sh`.

The setup helper is idempotent by endpoint name. If `h3-extended-6000-ada` already exists, it prints the existing endpoint instead of creating another one. Use `--force-new-endpoint` only when a genuinely separate endpoint is wanted.

The final line printed by the helper is suitable for the shell:

```bash
export RUNPOD_ENDPOINT_ID=...
```

Useful overrides include:

```bash
python deploy/runpod/setup_endpoint.py \
  --h3-ref <commit-or-branch> \
  --comfy-ref v0.31.0 \
  --gpu "NVIDIA RTX 6000 Ada Generation" \
  --model-reference "https://huggingface.co/Comfy-Org/MiniMax-H3:main"
```

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

`run_h3.py` walks backward from connected `SaveVideo` nodes, reads the filenames saved in connected image/audio/video loaders, resolves them beneath the local Comfy input directory, and uploads them inline. Unconnected loader nodes are ignored. It also keeps `MiniMaxH3HybridSparseAttentionZi` enabled, replaces only the custom preview sampler, bypasses the local Ziro image scaler, and applies the cached-model and inline-I/O placeholders.

The input directory is normally inferred from a workflow stored below `ComfyUI/User`. Use `COMFY_INPUT_DIR` or `--input-root` for a different layout. The optional `--input [NAME=]PATH` argument overrides one of the connected saved inputs; it is not required for the normal path.

The `run_h3.py` operation creates demand on the endpoint. If no worker is alive, RunPod schedules a matching RTX 6000 Ada worker. Because the endpoint has the H3 Hugging Face repository attached as a cached-model reference, RunPod selects a host with that cache when possible; otherwise RunPod populates the cache before the worker starts. The worker then runs the bootstrap, links the cached files, launches ComfyUI, executes the workflow, returns artifacts inline, becomes idle, and scales back to zero.

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
