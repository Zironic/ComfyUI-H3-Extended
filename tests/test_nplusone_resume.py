"""Resume geometry and the chunk-validity rule.

No torch tensors are needed for the scan itself, so these run on CPU in
milliseconds and pin the two rules that are expensive to get wrong: which
reference lengths decode exactly, and which stored chunks may be reused.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "chunked_ref2v"))
sys.path.insert(0, os.path.join(ROOT, "chunked_ref2v", "longform"))

import nplusone_resume as resume  # noqa: E402


class GroupAlignmentTest(unittest.TestCase):
    def test_legal_reference_frames_for_the_production_chunk(self):
        self.assertEqual(
            resume.legal_reference_frames(141),
            [5, 22, 39, 56, 73, 90, 107, 124],
        )

    def test_the_two_values_we_actually_use_are_aligned(self):
        # R=90 is what the reference run used; R=39 is the cheap option at 1MP.
        self.assertEqual(resume.group_aligned_slice(141, 90), (15, 27))
        self.assertEqual(resume.group_aligned_slice(141, 39), (30, 12))

    def test_51_is_rejected_with_the_legal_list(self):
        # 51 lands on a latent boundary but *not* a VAE group boundary, so it
        # would decode a partial group and silently return wrong frames.
        with self.assertRaises(ValueError) as ctx:
            resume.group_aligned_slice(141, 51)
        self.assertIn("group aligned", str(ctx.exception))
        self.assertIn("90", str(ctx.exception))

    def test_every_legal_value_round_trips_to_its_frame_count(self):
        for chunk_frames in (73, 90, 141, 192):
            for r in resume.legal_reference_frames(chunk_frames):
                start, count = resume.group_aligned_slice(chunk_frames, r)
                self.assertEqual(start % 5, 0, "must start on a group boundary")
                self.assertEqual(start + count,
                                 resume.group_aligned_slice(chunk_frames, r)[0] + count)

    def test_rejects_out_of_range(self):
        for bad in (0, -1, 142):
            with self.assertRaises(ValueError):
                resume.group_aligned_slice(141, bad)


class _FakeStore:
    """Writes only the metadata sidecars; the scan reads tensors lazily."""

    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, resume.SAMPLES_DIR), exist_ok=True)

    def write(self, index, *, prompt, seed, parent_sha, video_sha,
              audio_sha=None,
              chunk_frames=141, reference_frames=90, schema=None):
        audio_sha = audio_sha or "audio-%d" % index
        meta = {
            "schema": resume.CHUNK_SCHEMA if schema is None else schema,
            "index": index,
            "seed": seed,
            "prompt_sha256": resume.prompt_digest(prompt),
            "parent_sha256": parent_sha,
            "reference_frames": reference_frames,
            "chunk_frames": chunk_frames,
            "video_sha256": video_sha,
            "audio_sha256": audio_sha,
            "chunk_sha256": resume.chunk_digest_from_shas(video_sha, audio_sha),
        }
        with open(resume.chunk_path(self.root, index, ".json"), "w",
                  encoding="utf-8") as fh:
            json.dump(meta, fh)
        # a stand-in for the latent payload; load_chunk only needs the key
        with open(resume.chunk_path(self.root, index), "wb") as fh:
            fh.write(b"stub")


class ResumeScanTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="h3resume")
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.prompts = ["chunk %d" % i for i in range(5)]
        self.seeds = [1000 + i for i in range(5)]
        self.store = _FakeStore(self.root)
        # a valid chain: each chunk records the previous chunk's video digest
        parent = None
        for i in range(5):
            self.store.write(i, prompt=self.prompts[i], seed=self.seeds[i],
                             parent_sha=parent, video_sha="sha-%d" % i)
            parent = resume.chunk_digest_from_shas(
                "sha-%d" % i, "audio-%d" % i,
            )

        # load_chunk normally reads real tensors; stub it for the scan tests
        self._real_load = resume.load_chunk

        def fake_load(root, index):
            path = resume.chunk_path(root, index, ".json")
            if not os.path.exists(path):
                return None, None
            with open(path, encoding="utf-8") as fh:
                meta = json.load(fh)
            return {
                "video_latent": meta["video_sha256"],
                "audio_latent": meta["audio_sha256"],
            }, meta

        resume.load_chunk = fake_load
        self.addCleanup(setattr, resume, "load_chunk", self._real_load)
        self._real_digest = resume.latent_digest
        resume.latent_digest = lambda value: str(value)
        self.addCleanup(setattr, resume, "latent_digest", self._real_digest)

    def scan(self):
        return resume.resume_point(
            self.root, chunk_count=5,
            chunk_digests=[resume.prompt_digest(text) for text in self.prompts],
            chunk_seeds=self.seeds, reference_frames=90, chunk_frames=141)

    def test_untouched_run_needs_no_sampling(self):
        self.assertEqual(self.scan(), 5)

    def test_edited_prompt_resumes_at_that_chunk(self):
        self.prompts[2] = "rewritten"
        self.assertEqual(self.scan(), 2)

    def test_changed_seed_resumes_at_that_chunk(self):
        self.seeds[3] += 1
        self.assertEqual(self.scan(), 3)

    def test_changed_run_seed_resumes_at_zero(self):
        self.seeds = [s + 999 for s in self.seeds]
        self.assertEqual(self.scan(), 0)

    def test_missing_chunk_resumes_there(self):
        os.remove(resume.chunk_path(self.root, 3, ".json"))
        self.assertEqual(self.scan(), 3)

    def test_broken_parent_chain_is_caught(self):
        # chunk 3 claims a parent that is not what sits at index 2
        self.store.write(3, prompt=self.prompts[3], seed=self.seeds[3],
                         parent_sha="sha-from-another-run", video_sha="sha-3")
        self.assertEqual(self.scan(), 3)

    def test_tampered_video_payload_is_caught(self):
        real_load = resume.load_chunk

        def tampered(root, index):
            tensors, meta = real_load(root, index)
            if index == 2:
                tensors["video_latent"] = "different-payload"
            return tensors, meta

        resume.load_chunk = tampered
        self.assertEqual(self.scan(), 2)

    def test_tampered_audio_payload_is_caught(self):
        real_load = resume.load_chunk

        def tampered(root, index):
            tensors, meta = real_load(root, index)
            if index == 2:
                tensors["audio_latent"] = "different-audio"
            return tensors, meta

        resume.load_chunk = tampered
        self.assertEqual(self.scan(), 2)

    def test_changed_reference_frames_invalidates_everything(self):
        got = resume.resume_point(
            self.root, chunk_count=5,
            chunk_digests=[resume.prompt_digest(text) for text in self.prompts],
            chunk_seeds=self.seeds, reference_frames=39, chunk_frames=141)
        self.assertEqual(got, 0)

    def test_old_schema_invalidates(self):
        self.store.write(0, prompt=self.prompts[0], seed=self.seeds[0],
                         parent_sha=None, video_sha="sha-0", schema=0)
        self.assertEqual(self.scan(), 0)

    def test_invalidate_from_removes_the_whole_suffix(self):
        resume.invalidate_from(self.root, 2, 5)
        for i in (0, 1):
            self.assertTrue(os.path.exists(resume.chunk_path(self.root, i, ".json")))
        for i in (2, 3, 4):
            self.assertFalse(os.path.exists(resume.chunk_path(self.root, i, ".json")))


if __name__ == "__main__":
    unittest.main()
