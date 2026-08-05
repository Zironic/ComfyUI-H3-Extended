"""Import bootstrap smoke test for minimax VRAM probe path resolution."""

import os


def _resolve_comfyui_root(path):
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(path)), "..", ".."))


def test_probe_root_from_nested_custom_node_path():
    # Probe path as launched by:
    #   python custom_nodes/ComfyUI-H3-Extended/minimax_vram_probe.py ...
    probe_path = os.path.join(
        "C:", "Users", "johan", "ComfyUI-Installs", "Comfy2", "ComfyUI",
        "custom_nodes", "ComfyUI-H3-Extended", "minimax_vram_probe.py"
    )
    expected_root = os.path.abspath(os.path.join(
        "C:", "Users", "johan", "ComfyUI-Installs", "Comfy2", "ComfyUI"
    ))
    assert os.path.basename(_resolve_comfyui_root(probe_path)) == "ComfyUI"
    assert _resolve_comfyui_root(probe_path) == expected_root


def test_base_module_root_from_nested_custom_node_path():
    base_path = os.path.join(
        "C:", "Users", "johan", "ComfyUI-Installs", "Comfy2", "ComfyUI",
        "custom_nodes", "ComfyUI-H3-Extended", "_minimax_vram_probe_base.py"
    )
    expected_root = os.path.abspath(os.path.join(
        "C:", "Users", "johan", "ComfyUI-Installs", "Comfy2", "ComfyUI"
    ))
    assert _resolve_comfyui_root(base_path) == expected_root
