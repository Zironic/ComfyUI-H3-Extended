"""The experiment catalog and the named suites that select from it.

An `ExperimentSpec` is data. It names a carry strategy and four policies, and
that is the whole description of an arm - the runner reads nothing else.
"""

from dataclasses import dataclass

from .strategies import get_strategy

PROMPT_POLICIES = ("original", "keyframe_completion", "video2", "composite")
POSITION_POLICIES = ("copy_target", "stock", "none")
SOURCE_REFERENCE_POLICIES = ("original", "composite", "plus_generated_video2")
TARGET_POLICIES = ("sample_all", "clamp_overlap")


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    display_name: str

    carry_strategy: str
    prompt_policy: str = "original"
    position_policy: str = "copy_target"
    source_reference_policy: str = "original"
    target_policy: str = "sample_all"

    enabled: bool = True
    notes: str = ""

    def strategy(self):
        return get_strategy(self.carry_strategy)

    def dependencies(self):
        return self.strategy().dependencies()

    def as_dict(self):
        return {
            "experiment_id": self.experiment_id,
            "display_name": self.display_name,
            "carry_strategy": self.carry_strategy,
            "prompt_policy": self.prompt_policy,
            "position_policy": self.position_policy,
            "source_reference_policy": self.source_reference_policy,
            "target_policy": self.target_policy,
            "notes": self.notes,
        }


CATALOG = {}


def _register(spec):
    if spec.prompt_policy not in PROMPT_POLICIES:
        raise ValueError("bad prompt policy on %s" % spec.experiment_id)
    if spec.position_policy not in POSITION_POLICIES:
        raise ValueError("bad position policy on %s" % spec.experiment_id)
    if spec.source_reference_policy not in SOURCE_REFERENCE_POLICIES:
        raise ValueError("bad source reference policy on %s" % spec.experiment_id)
    if spec.target_policy not in TARGET_POLICIES:
        raise ValueError("bad target policy on %s" % spec.experiment_id)
    CATALOG[spec.experiment_id] = spec
    return spec


_register(ExperimentSpec(
    experiment_id="baseline_none",
    display_name="Baseline (no carried state)",
    carry_strategy="none",
    position_policy="none",
    notes="Control for every anchor and overlap-adherence measurement.",
))

_register(ExperimentSpec(
    experiment_id="frame_reencode_corrected",
    display_name="Re-encoded frame, target-aligned",
    carry_strategy="frame_reencode",
    notes="The original Stage-0 concept: decode frame S, VAE-encode, condition "
          "at target position 0.",
))

_register(ExperimentSpec(
    experiment_id="frame_direct_corrected",
    display_name="Direct latent frame, target-aligned",
    carry_strategy="direct_latent_frame",
    notes="Same placement as frame_reencode_corrected, no VAE round trip - the "
          "difference between them is the round trip.",
))

_register(ExperimentSpec(
    experiment_id="frame_direct_stock_position",
    display_name="Direct latent frame, stock (pre-reference) position",
    carry_strategy="direct_latent_frame",
    position_policy="stock",
    notes="Diagnostic for the MM-RoPE placement correction. Identical latent "
          "source to frame_direct_corrected.",
))

_register(ExperimentSpec(
    experiment_id="frame_direct_prompted",
    display_name="Direct latent frame + keyframe-completion prompt",
    carry_strategy="direct_latent_frame",
    prompt_policy="keyframe_completion",
    notes="Deferred until the unmodified prompt has been tested.",
))

_register(ExperimentSpec(
    experiment_id="aligned_overlap_direct",
    display_name="Direct target-aligned overlap",
    carry_strategy="direct_latent_overlap",
    notes="Primary candidate for the production mechanism.",
))

_register(ExperimentSpec(
    experiment_id="aligned_overlap_stock_position",
    display_name="Target-aligned overlap, stock position",
    carry_strategy="direct_latent_overlap",
    position_policy="stock",
    notes="Only useful if the corrected full-overlap arm behaves unexpectedly.",
))

_register(ExperimentSpec(
    experiment_id="aligned_overlap_prompted",
    display_name="Target-aligned overlap + continuation prompt",
    carry_strategy="direct_latent_overlap",
    prompt_policy="keyframe_completion",
    notes="Optional follow-up to aligned_overlap_direct.",
))

_register(ExperimentSpec(
    experiment_id="generated_overlap_video2",
    display_name="Generated overlap as <Video 2>",
    carry_strategy="generated_overlap_video2",
    prompt_policy="video2",
    position_policy="none",
    source_reference_policy="plus_generated_video2",
    notes="No target-aligned rows; the overlap is an ordinary reference block "
          "Qwen can see.",
))

_register(ExperimentSpec(
    experiment_id="composite_source",
    display_name="Composite source reference",
    carry_strategy="composite_source",
    prompt_policy="composite",
    position_policy="none",
    source_reference_policy="composite",
    notes="One continuous <Video 1>; no extra transformer rows; original source "
          "geometry is unavailable inside the overlap.",
))

_register(ExperimentSpec(
    experiment_id="target_overlap_clamped",
    display_name="Sigma-correct clamped target overlap",
    carry_strategy="clamped_target_overlap",
    position_policy="none",
    target_policy="clamp_overlap",
    enabled=False,
    notes="Needs a sampler intervention (milestone 5). Refuses to run until "
          "that exists.",
))


SUITES = {
    "minimal": [
        "baseline_none",
        "frame_reencode_corrected",
        "frame_direct_corrected",
        "frame_direct_stock_position",
    ],
    "aligned": [
        "baseline_none",
        "frame_direct_corrected",
        "aligned_overlap_direct",
    ],
    "prompt": [
        "frame_direct_corrected",
        "frame_direct_prompted",
        "aligned_overlap_direct",
        "aligned_overlap_prompted",
    ],
    "reference": [
        "baseline_none",
        "aligned_overlap_direct",
        "generated_overlap_video2",
        "composite_source",
    ],
    "clamp": [
        "aligned_overlap_direct",
        "target_overlap_clamped",
    ],
    "all": [k for k, v in CATALOG.items() if v.enabled],
}

SUITE_NAMES = list(SUITES) + ["custom"]

# Lower-memory arms first, so a later OOM cannot destroy an earlier artifact.
RUN_ORDER = [
    "baseline_none",
    "frame_reencode_corrected",
    "frame_direct_corrected",
    "frame_direct_stock_position",
    "frame_direct_prompted",
    "aligned_overlap_direct",
    "aligned_overlap_stock_position",
    "aligned_overlap_prompted",
    "composite_source",
    "generated_overlap_video2",
    "target_overlap_clamped",
]


def resolve_suite(suite, custom_experiments=""):
    """Experiment ids for a suite name, in run order.

    `custom` reads a comma-separated list; every id is validated against the
    catalog so a typo fails before Chunk A is generated rather than after.
    """
    if suite == "custom":
        ids = [x.strip() for x in custom_experiments.split(",") if x.strip()]
        if not ids:
            raise ValueError("experiment_suite is 'custom' but custom_experiments is empty")
    else:
        try:
            ids = list(SUITES[suite])
        except KeyError:
            raise ValueError("unknown suite %r (known: %s)"
                             % (suite, ", ".join(sorted(SUITES)))) from None

    unknown = [i for i in ids if i not in CATALOG]
    if unknown:
        raise ValueError("unknown experiment id(s): %s (known: %s)"
                         % (", ".join(unknown), ", ".join(sorted(CATALOG))))

    order = {name: i for i, name in enumerate(RUN_ORDER)}
    return sorted(dict.fromkeys(ids), key=lambda i: order.get(i, len(order)))


def union_dependencies(experiment_ids):
    """Everything Phase D must prepare for the selected experiments, combined."""
    from .strategies import StrategyDependencies

    deps = StrategyDependencies()
    for experiment_id in experiment_ids:
        deps = deps.union(CATALOG[experiment_id].dependencies())
    return deps
