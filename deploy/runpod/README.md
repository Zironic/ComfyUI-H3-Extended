# RunPod Serverless deployment

This branch contains an experimental RunPod deployment path that deliberately uses RunPod's stock ComfyUI worker image instead of building and distributing a custom Docker image.

The runtime sequence is:

1. boot RunPod's stock `worker-comfyui` image,
2. fetch the requested ComfyUI revision into the existing runtime,
3. fetch this repository into `custom_nodes`,
4. detect the worker GPU and compile the pinned SpargeAttention extension for its CUDA architecture in the stock venv,
5. symlink selected MiniMax H3 files from RunPod Cached Models into Comfy's model directories,
6. replace the stock image-only RunPod handler with the H3-aware handler in this directory,
7. hand control back to RunPod's stock `/start.sh`.

The deployment code lives here because it is source-controlled deployment logic. Runtime state is written beneath `/comfyui/user/runpod`, while per-job inputs and outputs use `/comfyui/input/runpod/<job_id>` and `/comfyui/output/runpod/<job_id>`.

## RunPod endpoint

Default smoke-test configuration:

- GPU profile: `rtx4090` (`NVIDIA GeForce RTX 4090`, SM89, 24 GB)
- active workers: 0
- max workers: 1
- GPUs per worker: 1
- idle timeout: 5 seconds
- execution timeout: 7200 seconds
- network volume: none
- Cached Model: `Comfy-Org/MiniMax-H3`
- Hybrid Sparse Attention: enabled by the submitted workflow

The endpoint uses the stock RunPod image `runpod/worker-comfyui:5.8.6-base-cuda12.8.1`, which contains `/start.sh`, `/opt/venv`, `uv`, `git`, and `wget`. No custom image or Dockerfile is required. Each newly allocated worker may pay the CUDA compiler install and source-build cost before ComfyUI starts. A still-running worker reuses its marker only when the GPU architecture, pinned ref, Torch/CUDA build identity, and extension symbols all match.

`setup_endpoint.py` keeps GPU choice outside the worker code. `rtx4090` is the default, while `rtx5090`, `rtx6000-ada`, `l40s`, and `a100-80gb` are named alternatives. Use `--gpu` for an exact RunPod GPU ID that is not yet in the profile list. Profiles create distinct default endpoint/template names, so switching GPU does not silently mutate an existing endpoint.

```bash
python deploy/runpod/setup_endpoint.py --list-gpu-profiles
python deploy/runpod/setup_endpoint.py
python deploy/runpod/setup_endpoint.py --gpu-profile l40s
python deploy/runpod/setup_endpoint.py --gpu "NVIDIA Future GPU"
```

The bootstrap currently supports SM80, SM86, SM87, SM89, SM90, and SM120. B200 is deliberately rejected because it is SM100, which the current H3 Hybrid Sparse backend does not support.

## Container start command

Use the stock worker image and override the container start command with a tiny bootstrap download. Replace `<REF>` with this branch or, preferably once validated, an immutable commit SHA.

```bash
wget -qO /tmp/h3-runpod-bootstrap.sh \
  https://raw.githubusercontent.com/Zironic/ComfyUI-H3-Extended/<REF>/deploy/runpod/bootstrap.sh \
&& chmod +x /tmp/h3-runpod-bootstrap.sh \
&& H3_EXTENDED_REF=<REF> /tmp/h3-runpod-bootstrap.sh
```

For this experimental branch:

```bash
wget -qO /tmp/h3-runpod-bootstrap.sh \
  https://raw.githubusercontent.com/Zironic/ComfyUI-H3-Extended/main/deploy/runpod/bootstrap.sh \
&& chmod +x /tmp/h3-runpod-bootstrap.sh \
&& H3_EXTENDED_REF=main /tmp/h3-runpod-bootstrap.sh
```

Useful bootstrap environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFYUI_REF` | `v0.31.0` | ComfyUI tag, branch, or commit fetched at startup |
| `H3_EXTENDED_REF` | `main` | H3-Extended branch/tag/commit fetched at startup |
| `RUNPOD_SKIP_COMFY_UPDATE` | `0` | Set to `1` to retain the ComfyUI version bundled in the stock image |
| `RUNPOD_H3_MODEL_REPO` | `Comfy-Org/MiniMax-H3` | Hugging Face repo configured as RunPod Cached Model |
| `RUNPOD_H3_REQUIRED` | `0` | Set to `1` to fail startup if no expected cached H3 file is linked |
| `RUNPOD_BOOTSTRAP_SMOKE_TEST` | `0` | Set to `1` to run Comfy's CPU import quick test before worker startup |
| `SPARGE_REPO` | `https://github.com/woct0rdho/SpargeAttn.git` | SpargeAttention source repository |
| `SPARGE_REF` | `067d80cb6b76345c7b8be40e86c7d19a3cf7c4eb` | Pinned portable SpargeAttention commit |
| `SPARGE_BUILD_JOBS` | empty | Optional source-build parallelism override |

The bootstrap records the resolved ComfyUI, H3-Extended, and SpargeAttention refs in:

```text
/comfyui/user/runpod/deployment.json
```

The install marker is `/comfyui/user/runpod/sparge-attn.json`. It is never used by itself: the bootstrap also matches the detected architecture, Torch and CUDA versions, and H3's exact extension ABI before skipping a build.

## Cached H3 models

`link_h3_models.py` searches RunPod's Hugging Face cache under:

```text
/runpod-volume/huggingface-cache/hub/models--Comfy-Org--MiniMax-H3/snapshots/
```

It currently links these filenames when present:

```text
minimax_h3_fl2va_pruned_int8_convrot.safetensors
minimax_h3_ref2va_pruned_int8_convrot.safetensors
qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
minimax_h3_video_vae_fp16.safetensors
minimax_h3_audio_vae_fp32.safetensors
```

The script copies no model bytes. It creates symlinks into Comfy's `diffusion_models`, `text_encoders`, and `vae` directories.

The exact filenames are deployment policy, not an H3 core requirement. Update `MODEL_TARGETS` if the workflow uses another quantization.

## Input assets

Input files are embedded directly in the RunPod request as base64. No object storage or signed URL is required.

Request schema:

```json
{
  "input": {
    "workflow": { "...": "ComfyUI API-format workflow" },
    "assets": [
      {
        "name": "reference.mp4",
        "content_type": "video/mp4",
        "base64": "AAAA..."
      }
    ]
  }
}
```

The handler decodes each asset into:

```text
/comfyui/input/runpod/<job_id>/<name>
```

A workflow may use placeholders instead of knowing the generated job ID:

```text
{{ASSET:reference.mp4}}
{{RUNPOD_JOB_ID}}
{{RUNPOD_OUTPUT_PREFIX}}
```

They are replaced recursively in all string values in the API-format workflow before it is sent to ComfyUI.

`{{RUNPOD_OUTPUT_PREFIX}}` becomes:

```text
runpod/<job_id>/result
```

Use that as the filename prefix in save/video-combine nodes where possible. This makes output discovery deterministic.

## Output artifacts

The stock RunPod Comfy handler only treats image outputs specially. `handler.py` instead gathers:

1. files explicitly reported by ComfyUI history, and
2. every file created below `/comfyui/output/runpod/<job_id>/`.

This allows MP4, WAV, PNG, JSON diagnostics, sparse-attention reports, and similar artifacts to be returned from the same job.

Artifacts are returned inline as base64 and decoded by `run_h3.py`. RunPod currently limits `/runsync` payloads to 20 MB, so this transport is intended for reference files and outputs that remain comfortably below that limit after base64 and JSON overhead. The client rejects encoded requests above 19 MB, and the handler applies the same guard to its result.

A successful result has the form:

```json
{
  "status": "success",
  "job_id": "...",
  "prompt_id": "...",
  "output_prefix": "runpod/.../result",
  "artifacts": [
    {
      "name": "result_00001.mp4",
      "bytes": 3456789,
      "content_type": "video/mp4",
      "base64": "AAAA..."
    }
  ]
}
```

## Submit a job

Export the Comfy workflow in API format, then pass that export directly to `run_h3.py`:

```bash
python deploy/runpod/run_h3.py workflow_api_export.json
```

The runner normalizes the workflow in memory. It keeps `MiniMaxH3HybridSparseAttentionZi` enabled, replaces the custom preview sampler with core `SamplerCustomAdvanced`, bypasses `ZiroScaleImageToNativeCanvas`, and selects the cached-model filenames. Starting at each connected `SaveVideo`, it walks upstream and automatically collects saved files from connected core `LoadImage`, `LoadImageMask`, `LoadAudio`, and `LoadVideo` nodes. Disconnected loaders are ignored.

When the workflow lives under the normal `ComfyUI/User/.../workflows` directory, the local `ComfyUI/Input` directory is inferred. Otherwise set `COMFY_INPUT_DIR` or pass `--input-root`. `--input [NAME=]PATH` remains available only to override an automatically discovered asset.

To inspect or save the normalized workflow without submitting it, use:

```bash
python deploy/runpod/prepare_workflow.py workflow_api_export.json workflow_runpod.json
```

Credentials can be supplied with environment variables:

```text
RUNPOD_API_KEY
RUNPOD_ENDPOINT_ID
```

`run_h3.py` uses `/runsync` with the maximum five-minute wait, then polls the returned job ID when inference takes longer. It decodes returned artifacts into `runpod-output` by default. `submit_job.py` remains as a compatibility entry point for the same client.

## First smoke test

Keep `MiniMaxH3HybridSparseAttentionZi` enabled in the target workflow. The bootstrap detects the actual worker capability and builds the matching Sparse Sage extension in `/opt/venv` when needed. At submission time the handler selects `sage128_fused_qkv` on SM89 GPUs (4090, RTX 6000 Ada, and L40S profiles) and portable `sage128` on the other supported architectures.

The first test should establish, in order:

1. the stock worker boots,
2. ComfyUI updates successfully,
3. H3-Extended imports,
4. cached H3 files are visible and linked,
5. an API workflow validates,
6. H3 fits and runs on the selected worker,
7. MP4/audio output is produced,
8. the MP4 is returned inline and decoded locally,
9. the worker scales back to zero.

## Current limitations

- No automatic workflow conversion is performed. Submit ComfyUI API-format JSON.
- Inline requests and results must remain below RunPod's payload limits after base64 expansion.
- The handler assumes a single ComfyUI process on `127.0.0.1:8188`, matching RunPod's stock worker architecture.
- Cached-model file discovery is filename-based and deliberately fails on duplicate matches.
- The branch has not yet been exercised on a live RunPod endpoint; the first live run is the validation step. In particular, the default 4090 has 24 GB, so CPU tests and shell syntax checks cannot prove that this exact workflow fits, that the worker can install the CUDA toolkit, or that the extension compiles successfully.
