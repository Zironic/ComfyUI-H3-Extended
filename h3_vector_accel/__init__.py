"""CPU-safe H3 vector acceleration sampler primitives."""

from .config import SamplerConfig, actual_mask, profile_mask
from .fingerprint import configuration_fingerprint, sigma_hash
from .predictor import (
    HoldPredictor,
    LinearVelocityPredictor,
    Prediction,
    VDEPredictor,
)
from .repairability import (
    NativeTrajectory,
    RepairabilityProfile,
    capture_native_trajectory,
    run_repairability_sweep,
)
from .sampler import H3SamplingContext, resolve_h3_sampling, sample_vector_accel

__all__ = [
    "SamplerConfig",
    "actual_mask",
    "profile_mask",
    "Prediction",
    "HoldPredictor",
    "LinearVelocityPredictor",
    "VDEPredictor",
    "NativeTrajectory",
    "RepairabilityProfile",
    "capture_native_trajectory",
    "run_repairability_sweep",
    "H3SamplingContext",
    "resolve_h3_sampling",
    "sample_vector_accel",
    "configuration_fingerprint",
    "sigma_hash",
]
