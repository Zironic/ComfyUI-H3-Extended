# Live rerun after integration fixes

After updating to the commits following PR #41, restart ComfyUI so cached model-patch node outputs and wrapper objects are discarded.

For the first measurement run keep Hybrid Sparse activation memory disabled, keep Chipmunk in `measure` mode, and use the same 25% / refresh-6 settings. The timing-stage crash and duplicate-runtime-wrapper composition are fixed in the follow-up changes.
