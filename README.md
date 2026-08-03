# ComfyUI-MiniMax

Private fork of ComfyUI's built-in MiniMax H3 nodes (`comfy_extras/nodes_minimax_h3.py`),
so changes survive ComfyUI updates.

Forked at ComfyUI v0.30.1, including the local `raw_latent_t` addition on
`MiniMaxH3ImageToVideo`.

## Nodes

Node ids and display names carry a `Zi` suffix so both these and the stock nodes
can be loaded at the same time:

| node id | display name |
| --- | --- |
| `EmptyMiniMaxH3LatentAVZi` | Empty MiniMax H3 AV Latent (Zi) |
| `MiniMaxH3ImageToVideoZi` | MiniMax H3 Image to Video (Zi) |
| `MiniMaxH3ReferenceToVideoZi` | MiniMax H3 Reference to Video (Zi) |
| `MiniMaxH3SigmaShiftZi` | MiniMax H3 Sigma Shift (Zi) |

Existing workflows still point at the stock ids; re-add the `(Zi)` nodes to use
this copy.

## Notes

The nodes only produce conditioning + latents — all model-side behaviour
(`comfy/ldm/minimax/`, the minimax CLIP/tokenizer, the VAEs) still lives in core
and is *not* forked here. If a core update changes those conditioning keys
(`minimax_keyframes`, `minimax_refs`, `minimax_h3_sigma_shift_*`), this fork
needs the matching update.
