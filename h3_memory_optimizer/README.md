# MiniMax H3 Memory Optimizer

One model-patch node that composes the existing attention and activation-memory
implementations without merging their copied core methods.

## First release behavior

- **SM89 with a compatible SageAttention 2.2 build and Triton:** installs the
  prepared-QKV efficient Sage adapter and BF16/native MLP chunking.
- **Every other environment:** preserves the incoming attention backend and can
  still install portable BF16 MLP chunking.
- Capability failures are handled before attention patch installation. Unexpected
  runtime CUDA failures remain hard errors.

## Extension model

Add future SM80/86/90/120 support by registering another adapter in
`attention.ADAPTERS`. The node, activation patch, and status format do not need to
change.
