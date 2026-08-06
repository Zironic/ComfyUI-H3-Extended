"""Comfy-native output contract for disk-backed long-form video nodes.

The long-form engines must write a backing file while they stream decoded chunks,
but that file is an implementation detail.  Graph users should receive a normal
VIDEO value and connect it to ComfyUI's Save Video node, which owns final naming,
container, codec, metadata, and output preview behavior.
"""

from __future__ import annotations

import os
from dataclasses import is_dataclass, replace

from comfy_api.latest import InputImpl, io

_LONGFORM_NODE_IDS = {
    "MiniMaxH3LongFormRef2VZi",
    "MiniMaxH3LongFormReferenceVideoZi",
}


def _replace_schema(schema, *, inputs, outputs):
    updates = {
        "inputs": inputs,
        "outputs": outputs,
        "is_output_node": False,
    }
    if is_dataclass(schema):
        return replace(schema, **updates)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update=updates)
    schema.inputs = inputs
    schema.outputs = outputs
    schema.is_output_node = False
    return schema


def _video_from_path(output_path):
    """Return a public Comfy VIDEO object backed by the streamed output file."""
    if not output_path or not os.path.isfile(output_path):
        raise RuntimeError(
            "long-form generation completed without a backing video file"
        )
    return InputImpl.VideoFromFile(output_path)


def _video_result(output_path):
    """Compatibility hook for the existing node implementations.

    They still call ``_video_result`` internally.  Replacing that helper keeps
    their bounded-memory writer intact while removing use of the private
    ``comfy_api.latest._input_impl`` module and terminal preview UI.
    """
    return _video_from_path(output_path), None


def install_public_video_factory():
    """Patch every imported long-form helper binding to the public API."""
    from . import nodes, reference_nodes, reference_preview_nodes

    nodes._video_result = _video_result
    reference_nodes._video_result = _video_result
    reference_preview_nodes._video_result = _video_result


def adapt_longform_node(node):
    """Expose a long-form node as a normal VIDEO-producing graph node."""
    schema = node.define_schema()
    if schema.node_id not in _LONGFORM_NODE_IDS:
        return node

    class ComfyVideoOutputNode(node):
        @classmethod
        def define_schema(cls):
            original = super().define_schema()
            inputs = [
                item
                for item in original.inputs
                if getattr(item, "id", getattr(item, "name", None))
                != "output_video"
            ]
            video_outputs = [
                item for item in original.outputs if item.io_type == "VIDEO"
            ]
            other_outputs = [
                item for item in original.outputs if item.io_type != "VIDEO"
            ]
            if len(video_outputs) != 1:
                raise RuntimeError(
                    "%s must define exactly one VIDEO output" % original.node_id
                )
            return _replace_schema(
                original,
                inputs=inputs,
                outputs=video_outputs + other_outputs,
            )

        @classmethod
        def execute(cls, *args, **kwargs):
            # A VIDEO-producing node must always produce its backing stream.
            kwargs["output_video"] = True
            original = super().execute(*args, **kwargs)
            if not isinstance(original, io.NodeOutput) or len(original.args) != 5:
                raise RuntimeError(
                    "unexpected long-form output contract from %s" % cls.__name__
                )

            preview, run_directory, video_path, report, _legacy_video = original.args
            video = _video_from_path(video_path)
            return io.NodeOutput(
                video,
                preview,
                run_directory,
                video_path,
                report,
            )

    ComfyVideoOutputNode.__name__ = node.__name__
    ComfyVideoOutputNode.__qualname__ = node.__qualname__
    ComfyVideoOutputNode.__module__ = node.__module__
    return ComfyVideoOutputNode


install_public_video_factory()


__all__ = ["adapt_longform_node", "install_public_video_factory"]
