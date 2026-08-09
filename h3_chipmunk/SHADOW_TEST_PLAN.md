# H3 Chipmunk shadow validation

This file marks the implementation branch for the output-exact depth-profile validation test.

The validator keeps the dense H3 MLP result in the real model path and computes a Chipmunk delta approximation only on sampled video rows/layers. No approximate tensor is fed forward.
