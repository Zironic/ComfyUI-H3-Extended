# RunPod Serverless deployment

This branch contains an experimental RunPod deployment path that deliberately uses RunPod's stock ComfyUI worker image instead of building and distributing a custom Docker image.

The runtime sequence is:

1. boot RunPod's stock `worker-comfyui` image,
2. fetch the requested ComfyUI revision into the existing runtime,
3. fetch this repository into `custom_nodes`,
4. symlink selected MiniMax H3 files from RunPod Cached Models into Comfy's model directories,
5. replace the stock image-only RunPod handler with the H3-aware handler in this directory,
6. hand control back to RunPod's stock `/start.sh`.

The deployment code lives here because it is source-controlled deployment logic. Runtime state is written beneath `/comfyui/user/runpod`, while per-job inputs and outputs use `/comfyui/input/runpod/<job_id>` and `/comfyui/output/runpod/<job_id>`.

## RunPod endpoint

Recommended first smoke-test configuration:

- GPU class: L40 / L40S / RTX 6000 Ada 48 GB
- active workers: 0
- max workers: 1
- GPUs per worker: 1
- idle timeout: 5 seconds
- execution timeout: 7200 seconds
- network volume: none
- Cached Model: `Comfy-Org/MiniMax-H3`
- Hybrid Sparse Attention: disabled for the first smoke test

The endpoint must use a RunPod `worker-comfyui` base image that contains `/start.sh`, `/opt/venv`, `uv`, `git`, and `wget`. The deployment currently targets the `5.8.6` worker layout.

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
  https://raw.githubusercontent.com/Zironic/ComfyUI-H3-Extended/agent/runpod-serverless/deploy/runpod/bootstrap.sh \
&& chmod +x /tmp/h3-runpod-bootstrap.sh \
&& H3_EXTENDED_REF=agent/runpod-serverless /tmp/h3-runpod-bootstrap.sh
```

Useful bootstrap environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `COMFYUI_REF` | `v0.31.0` | ComfyUI tag, branch, or commit fetched at startup |
| `H3_EXTENDED_REF` | `agent/runpod-serverless` | H3-Extended branch/tag/commit fetched at startup |
| `RUNPOD_SKIP_COMFY_UPDATE` | `0` | Set to `1` to retain the ComfyUI version bundled in the stock image |
| `RUNPOD_H3_MODEL_REPO` | `Comfy-Org/MiniMax-H3` | Hugging Face repo configured as RunPod Cached Model |
| `RUNPOD_H3_REQUIRED` | `0` | Set to `1` to fail startup if no expected cached H3 file is linked |
| `RUNPOD_BOOTSTRAP_SMOKE_TEST` | `0` | Set to `1` to run Comfy's CPU import quick test before worker startup |

The bootstrap records the resolved ComfyUI and H3-Extended commits in:

```text
/comfyui/user/runpod/deployment.json
```

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

Large reference images, audio, and video are not embedded in the RunPod JSON request. Put them in HTTP-accessible object storage and submit signed URLs.

Request schema:

```json
{
  "input": {
    "workflow": { "...": "ComfyUI API-format workflow" },
    "assets": [
      {
        "name": "reference.mp4",
        "url": "https://signed.example/reference.mp4"
      }
    ]
  }
}
```

The handler downloads each asset into:

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

Configure RunPod's S3-compatible bucket environment variables supported by the RunPod SDK:

```text
BUCKET_ENDPOINT_URL
BUCKET_ACCESS_KEY_ID
BUCKET_SECRET_ACCESS_KEY
BUCKET_NAME        # optional; otherwise RunPod's helper uses its default bucket naming behavior
```

The handler uses `runpod.serverless.utils.rp_upload.upload_file_to_bucket`, which performs multipart upload for arbitrary files and returns a presigned URL.

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
      "bytes": 12345678,
      "content_type": "video/mp4",
      "url": "https://..."
    }
  ]
}
```

## Submit a job

Export the Comfy workflow in API format, then run:

```bash
python deploy/runpod/submit_job.py workflow_api.json \
  --endpoint "$RUNPOD_ENDPOINT_ID" \
  --asset reference.mp4="https://signed.example/reference.mp4"
```

Credentials can be supplied with environment variables:

```text
RUNPOD_API_KEY
RUNPOD_ENDPOINT_ID
```

`submit_job.py` uses RunPod's asynchronous `/run` API and polls the job until it reaches a terminal state. The default requested execution timeout is two hours.

## First smoke test

Do not test the custom SM89 Sparse Sage path at the same time as the serverless deployment plumbing. Start with a known-good dense or ordinary Sage H3 workflow on the RTX 6000 Ada.

The first test should establish, in order:

1. the stock worker boots,
2. ComfyUI updates successfully,
3. H3-Extended imports,
4. cached H3 files are visible and linked,
5. an API workflow validates,
6. H3 runs on the 48 GB worker,
7. MP4/audio output is produced,
8. the artifact uploads to object storage,
9. the worker scales back to zero.

After that works, install a precompiled SM89 `spas_sage_attn` wheel or move that dependency to a dedicated image revision and test `MiniMaxH3HybridSparseAttentionZi` separately.

## Current limitations

- No automatic workflow conversion is performed. Submit ComfyUI API-format JSON.
- Asset URLs must be HTTP(S) URLs accessible from the worker.
- Input assets are downloaded serially. This is intentional for the first implementation and can be parallelized later.
- The handler assumes a single ComfyUI process on `127.0.0.1:8188`, matching RunPod's stock worker architecture.
- Cached-model file discovery is filename-based and deliberately fails on duplicate matches.
- The branch has not yet been exercised on a live RunPod endpoint; the first live run is the validation step.
