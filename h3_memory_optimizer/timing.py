"""Dense memory-optimizer request timing and report listener."""

from datetime import datetime
import json
import logging
import os

try:
    from ..h3_attention.hybrid.stats import DeferredCudaTiming
    from ..h3_runtime.timing import publish_timing
except ImportError:
    from h3_attention.hybrid.stats import DeferredCudaTiming
    from h3_runtime.timing import publish_timing


LOG_PREFIX = "[H3 memory optimizer]"


class MemoryOptimizerTimingListener:
    """Generic runtime listener for dense attention/MLP timing."""

    def __init__(self, report_directory, selected_attention, reason="", timing=None):
        self.report_directory = str(report_directory).strip()
        self.selected_attention = str(selected_attention)
        self.reason = str(reason)
        self.timing = timing if timing is not None else DeferredCudaTiming(enabled=True)
        self._timestamp = None
        self.last_report_directory = None

    def on_request_reset(self, request_id):
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.timing.on_request_reset(request_id)

    def before_forward(self, snapshot, transformer_options, payload):
        device = getattr(snapshot, "device", None)
        cuda = getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"
        self.timing.begin_request(snapshot.request_id, cuda=cuda, snapshot=snapshot)
        publish_timing(transformer_options, self.timing)

    @staticmethod
    def _render(payload):
        timing = payload["timing"]
        lines = [
            "H3 Memory Optimizer timing",
            "selected attention: %s" % payload["selected_attention"],
            "attention reason: %s" % payload["attention_reason"],
            "request wall seconds: %s" % timing.get("request_wall_seconds"),
            "measured attention CUDA seconds: %.6f" % timing.get(
                "total_measured_attention_cuda_seconds", 0.0),
            "measured DiT block CUDA seconds: %.6f" % timing.get(
                "total_measured_dit_block_cuda_seconds", 0.0),
            "model forward CUDA seconds: %.6f (%d calls)" % (
                timing.get("total_model_forward_cuda_seconds", 0.0),
                timing.get("model_forward_call_count", 0),
            ),
        ]
        for stage, values in timing.get("stages", {}).items():
            lines.append(
                "%s: count=%d sum_ms=%.3f mean_ms=%.3f"
                % (
                    stage,
                    values.get("count", 0),
                    values.get("sum_ms", 0.0),
                    values.get("mean_ms", 0.0),
                )
            )
        for step in timing.get("per_step", ()):
            lines.append(
                "step %d (ordinal %d): attention %.6f s, DiT block %.6f s, "
                "model forward %.6f s"
                % (
                    step["step_index"], step["ordinal"],
                    step.get("total_measured_attention_cuda_seconds", 0.0),
                    step.get("total_measured_dit_block_cuda_seconds", 0.0),
                    step.get("total_model_forward_cuda_seconds", 0.0),
                )
            )
            for branch in step.get("branches", ()):
                lines.append(
                    "  branch %s: attention %.6f s, DiT block %.6f s, "
                    "model forward %.6f s"
                    % (
                        ",".join(str(value) for value in branch.get("branch", ())),
                        branch.get("total_measured_attention_cuda_seconds", 0.0),
                        branch.get("total_measured_dit_block_cuda_seconds", 0.0),
                        branch.get("total_model_forward_cuda_seconds", 0.0),
                    )
                )
        lines.append("Stage times are nested/overlapping; do not sum stage totals.")
        return "\n".join(lines) + "\n"

    def on_request_end(self, request_id, seconds=None):
        timing = self.timing.resolve(seconds)
        timestamp = self._timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        directory = os.path.join(self.report_directory, "timing_%s" % timestamp)
        os.makedirs(directory, exist_ok=False)
        payload = {
            "status": "complete",
            "mode": "dense_memory_optimizer",
            "request_id": int(request_id),
            "selected_attention": self.selected_attention,
            "attention_reason": self.reason,
            "timing": timing,
        }
        with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        with open(os.path.join(directory, "report.txt"), "w", encoding="utf-8") as handle:
            handle.write(self._render(payload))
        self.last_report_directory = directory
        logging.info("%s wrote timing report to %s", LOG_PREFIX, directory)
        return directory

    def as_status(self):
        return {
            "enabled": True,
            "selected_attention": self.selected_attention,
            "report_directory": self.report_directory,
            "last_report_directory": self.last_report_directory,
        }
