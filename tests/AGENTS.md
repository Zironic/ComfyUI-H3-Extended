# H3 test guidance

- Prefer a direct test script from the ComfyUI root over a broad suite. Use the
  managed interpreter and report the exact file and result.
- Establish CPU mode before the first Comfy import. On Windows use
  `CUDA_VISIBLE_DEVICES=-1`; an empty value does not reliably mask CUDA. Tests
  using Comfy argument parsing must call `comfy.options.enable_args_parsing()`
  before importing model-management paths and supply `--cpu` as required.
- Tests importing package code must place the H3 pack and ComfyUI root on
  `sys.path`. N+1 direct scripts may also require `chunked_ref2v` and
  `chunked_ref2v\longform`. Make the test self-contained instead of requiring a
  hidden `PYTHONPATH` from the current agent session.
- Use `h3_test_tempfile` for `TemporaryDirectory`, `mkdtemp`, and test-owned
  directory roots. It creates inherited-ACL directories under
  `.agent\tmp\tests`. Do not reintroduce Python 3.13 `tempfile` directory APIs,
  probe alternate temp roots, repair ACLs, or delete legacy ACL-locked folders
  as part of an unrelated test run.
- Inspect CUDA behavior before execution. `test_attention_backend.py` can run
  real kernels when CUDA/SageAttention are present, and `test_probe.py`
  constructs CUDA tensors despite mocked driver decisions. These require
  explicit GPU permission. `test_vram_guard.py` forces CPU mode and mocks its
  CUDA and VBAR calls.
- Distinguish contract coverage from live evidence. Mocked driver calls,
  synthetic tensors, fake CUDA events, imports, and compilation do not establish
  real kernel selection, VRAM behavior, model loading, inference, or media
  output.
- Keep test artifacts in `.agent\tmp`, clean only directories created by the
  current test, and preserve all pre-existing dirty/untracked work.
