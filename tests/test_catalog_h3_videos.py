import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import h3_test_tempfile as tempfile  # noqa: E402
from benchmarks import catalog_h3_videos as catalog


def prompt_graph(vector=None, active_sampler="res_multistep"):
    sampler = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["noise", 0], "guider": ["guider", 0], "sampler": ["ks", 0],
        "sigmas": ["sched", 0], "latent_image": ["latent", 0],
    }}
    if vector:
        sampler = {"class_type": "SamplerCustomAdvanced", "inputs": {
            **sampler["inputs"], "sampler": ["vec", 0],
        }}
    return {
        "out": {"class_type": "SaveVideo", "inputs": {"images": ["sample", 0]}},
        "sample": sampler,
        "ks": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": active_sampler}},
        "vec": {"class_type": "MiniMaxH3VectorAccelSamplerZi", "inputs": vector or {}},
        "sched": {"class_type": "BasicScheduler", "inputs": {"scheduler": "normal", "steps": 20}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "guider": {"class_type": "BasicGuider", "inputs": {"model": ["model", 0]}},
        "latent": {"class_type": "EmptyLatent", "inputs": {}},
        "unused": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "text": {"class_type": "PrimitiveStringMultiline", "inputs": {"value": "same prompt"}},
    }


class CatalogTests(unittest.TestCase):
    def test_active_link_and_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h3_00002.mp4"
            path.write_bytes(b"not media")
            tags = {"prompt": json.dumps(prompt_graph())}
            with patch.object(catalog, "probe_tags", return_value=tags):
                row = catalog.catalog_file(path, "ffprobe")
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["active_sampler"]["name"], "res_multistep")
        self.assertEqual(row["active_sampler"]["source_kind"], "KSamplerSelect")
        self.assertEqual(row["classification"], "stock_res20")
        self.assertEqual(row["seed"], 42)
        self.assertEqual(row["video_number"], 2)
        graph = prompt_graph()
        self.assertIsNotNone(row["prompt_digest"])
        self.assertEqual(row["prompt_digest"], catalog._text_digest(dict(reversed(list(graph.items())))))

    def test_vector_and_adaptive_settings_are_preserved(self):
        vector = {"method": "res_multistep", "evaluation_profile": "adaptive_embedded_res_v1", "embedded_video_tolerance": 0.05}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h3_00003.mp4"
            path.write_bytes(b"not media")
            with patch.object(catalog, "probe_tags", return_value={"prompt": json.dumps(prompt_graph(vector))}):
                row = catalog.catalog_file(path, "ffprobe")
        self.assertEqual(row["evaluation_profile"], "adaptive_embedded_res_v1")
        self.assertEqual(row["classification"], "vector")
        self.assertEqual(row["vector_settings"]["embedded_video_tolerance"], 0.05)
        self.assertIn("vec", row["node_settings"])

    def test_vector_native_20_is_not_stock_res(self):
        vector = {"method": "linear_velocity", "evaluation_profile": "native_20"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "h3_00059.mp4"
            path.write_bytes(b"not media")
            with patch.object(catalog, "probe_tags", return_value={"prompt": json.dumps(prompt_graph(vector))}):
                row = catalog.catalog_file(path, "ffprobe")
        self.assertEqual(row["classification"], "vector")
        self.assertEqual(row["sampler_method"], "linear_velocity")
        self.assertEqual(row["evaluation_profile"], "native_20")

    def test_failure_isolated_and_order_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.mp4").write_bytes(b"")
            (root / "a.mp4").write_bytes(b"")
            def tags(path, _probe):
                if path.name == "b.mp4":
                    raise RuntimeError("broken")
                return {"prompt": json.dumps(prompt_graph())}
            with patch.object(catalog, "probe_tags", side_effect=tags):
                rows = catalog.catalog(root, ffprobe="ffprobe")
        self.assertEqual([row["filename"] for row in rows], ["a.mp4", "b.mp4"])
        self.assertEqual(rows[1]["status"], "error")
        self.assertEqual(catalog.render_text(rows), catalog.render_text(rows))


if __name__ == "__main__":
    unittest.main()
