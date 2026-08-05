"""Carry strategies: how Chunk B is told what Chunk A generated.

Each strategy declares what it needs from the harness (`dependencies()`) and
then, given the prepared common assets, produces whatever combination of prompt,
Qwen presentation, DiT reference blocks and target-aligned conditions it wants
(`prepare()`).

The runner asks the strategy; it never assumes an experiment "has a keyframe".
That is the difference between this and a set of hard-coded arms: adding an
eighth idea means adding a class, not editing the sampling loop.
"""

from dataclasses import dataclass, field

from .layout_ops import TargetAlignedCondition


@dataclass(frozen=True)
class StrategyDependencies:
    """What a strategy needs before Chunk B can be sampled.

    The harness unions these across the selected experiments and loads each
    heavy model at most once (`Phase D`), so selecting five strategies that all
    want Qwen costs one Qwen residency, not five.
    """

    needs_chunk_a_pixels: bool = False
    needs_chunk_a_latent: bool = False
    needs_anchor_reencode: bool = False

    needs_dynamic_video_vae: bool = False
    needs_dynamic_qwen: bool = False
    needs_sampler_intervention: bool = False

    def union(self, other):
        return StrategyDependencies(**{
            f: getattr(self, f) or getattr(other, f)
            for f in self.__dataclass_fields__
        })

    def as_dict(self):
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


@dataclass
class PreparedStrategy:
    """Everything one Chunk B run needs, assembled by the strategy.

    Every field is independent. A strategy may change only the prompt, only the
    conditions, only the reference blocks, or any combination - the runner does
    not care which.
    """

    prompt: str = ""
    conditioning: object = None                      # Qwen output for this arm
    qwen_ref_items: list = field(default_factory=list)
    dit_ref_blocks: list = field(default_factory=list)

    target_conditions: list = field(default_factory=list)
    position_policy: str = "copy_target"
    target_initializer: object = None
    sampler_intervention: object = None

    metadata: dict = field(default_factory=dict)


class StrategyUnavailable(RuntimeError):
    """A strategy cannot run: unaligned profile, missing asset, or not implemented."""


class CarryStrategy:
    """Base class. `strategy_id` is what an `ExperimentSpec` names."""

    strategy_id = ""

    def dependencies(self):
        return StrategyDependencies()

    def prepare(self, context, spec):
        raise NotImplementedError

    # --- shared helpers -------------------------------------------------

    @staticmethod
    def _base(context, spec):
        """The unmodified Chunk B presentation: original prompt, original refs."""
        return PreparedStrategy(
            prompt=context.prompt_for(spec.prompt_policy),
            conditioning=context.conditioning_for(spec.prompt_policy, "chunk_b"),
            qwen_ref_items=list(context.qwen_ref_items_b),
            dit_ref_blocks=list(context.dit_ref_blocks_b),
            position_policy=spec.position_policy,
        )


class NoCarry(CarryStrategy):
    """Control: Chunk B is generated as if Chunk A never happened."""

    strategy_id = "none"

    def prepare(self, context, spec):
        prepared = self._base(context, spec)
        prepared.metadata["carry"] = "none"
        return prepared


class DirectLatentFrame(CarryStrategy):
    """One target-aligned latent position, copied straight from Chunk A's output.

    No decode/re-encode round trip, so any difference against
    `frame_reencode` isolates the VAE round trip rather than the conditioning.
    """

    strategy_id = "direct_latent_frame"

    def dependencies(self):
        return StrategyDependencies(needs_chunk_a_latent=True)

    def prepare(self, context, spec):
        # `require` rather than attribute access: on a profile whose overlap does
        # not land on a latent boundary this asset is deliberately absent, and
        # refusing is the whole point of computing the mapping.
        latent = context.require("direct_frame_latent")
        prepared = self._base(context, spec)
        prepared.target_conditions = [TargetAlignedCondition(
            latent=latent,
            target_latent_start=0,
            label="previous frame (direct)",
            position_policy=spec.position_policy,
        )]
        prepared.metadata["carry"] = "direct latent position %d" % context.geometry.overlap_slice()[0]
        return prepared


class ReencodedFrame(CarryStrategy):
    """The Stage-0 concept: decode Chunk A frame S, VAE-encode it back."""

    strategy_id = "frame_reencode"

    def dependencies(self):
        return StrategyDependencies(
            needs_chunk_a_pixels=True,
            needs_anchor_reencode=True,
            needs_dynamic_video_vae=True,
        )

    def prepare(self, context, spec):
        latent = context.require("reencoded_frame_latent")
        prepared = self._base(context, spec)
        prepared.target_conditions = [TargetAlignedCondition(
            latent=latent,
            target_latent_start=0,
            label="previous frame (re-encoded)",
            position_policy=spec.position_policy,
        )]
        prepared.metadata["carry"] = "decoded frame %d, re-encoded" % context.geometry.stride_frames
        return prepared


class DirectLatentOverlap(CarryStrategy):
    """The full generated overlap, losslessly copied from Chunk A's output.

    The primary candidate for the production mechanism: the target is still
    fully sampled, and the overlap rides as a separate fixed condition stream,
    so this constrains Chunk B without pinning it.
    """

    strategy_id = "direct_latent_overlap"

    def dependencies(self):
        return StrategyDependencies(needs_chunk_a_latent=True)

    def prepare(self, context, spec):
        latent = context.require("overlap_latent")
        start, count = context.geometry.overlap_slice()
        prepared = self._base(context, spec)
        prepared.target_conditions = [TargetAlignedCondition(
            latent=latent,
            target_latent_start=0,
            label="previous overlap",
            position_policy=spec.position_policy,
        )]
        prepared.metadata["carry"] = (
            "direct latent positions %d-%d (T=%d) onto target 0-%d"
            % (start, start + count - 1, count, count - 1))
        return prepared


class GeneratedOverlapAsVideo2(CarryStrategy):
    """The generated overlap presented to Qwen as an ordinary second reference.

    No target-aligned rows at all: `<Video 2>` is a normal reference block, so
    this tests whether the model can use the carried state when it can *see* it
    described, rather than only feel it at fixed target coordinates.
    """

    strategy_id = "generated_overlap_video2"

    def dependencies(self):
        return StrategyDependencies(
            needs_chunk_a_pixels=True,
            needs_dynamic_video_vae=True,
            needs_dynamic_qwen=True,
        )

    def prepare(self, context, spec):
        block = context.require("video2_ref_block")
        prepared = self._base(context, spec)
        prepared.conditioning = context.conditioning_for(spec.prompt_policy, "video2")
        prepared.prompt = context.prompt_for(spec.prompt_policy)
        prepared.dit_ref_blocks = list(context.dit_ref_blocks_b) + [block]
        prepared.metadata["carry"] = "generated overlap as <Video 2> (%d frames)" % (
            context.geometry.overlap_frames)
        return prepared


class CompositeSource(CarryStrategy):
    """One source reference whose opening is replaced by the generated overlap.

    Zero extra transformer rows - Qwen sees a single continuous `<Video 1>` - at
    the price of losing the original source geometry inside the overlap.
    """

    strategy_id = "composite_source"

    def dependencies(self):
        return StrategyDependencies(
            needs_chunk_a_pixels=True,
            needs_dynamic_video_vae=True,
            needs_dynamic_qwen=True,
        )

    def prepare(self, context, spec):
        block = context.require("composite_ref_block")
        prepared = self._base(context, spec)
        prepared.conditioning = context.conditioning_for(spec.prompt_policy, "composite")
        prepared.prompt = context.prompt_for(spec.prompt_policy)
        prepared.dit_ref_blocks = [
            block if b is context.source_ref_block_b else b
            for b in context.dit_ref_blocks_b
        ]
        prepared.metadata["carry"] = (
            "composite <Video 1>: generated frames %d-%d then source frames %d-%d"
            % (context.geometry.stride_frames, context.geometry.chunk_frames - 1,
               context.geometry.overlap_frames, context.geometry.chunk_frames - 1))
        prepared.metadata["source_reference"] = "composite"
        return prepared


class ClampedTargetOverlap(CarryStrategy):
    """Sigma-correct clamping of the target overlap. Not implemented.

    This one cannot be expressed as condition rows: the known overlap has to be
    re-noised to the current sigma with one fixed noise tensor at every step,
    which is a sampler intervention. Inserting clean latents during high-noise
    steps instead - the obvious shortcut - produces a latent the model was never
    trained to see, so it is refused rather than approximated.
    """

    strategy_id = "clamped_target_overlap"

    def dependencies(self):
        return StrategyDependencies(
            needs_chunk_a_latent=True,
            needs_sampler_intervention=True,
        )

    def prepare(self, context, spec):
        raise StrategyUnavailable(
            "clamped_target_overlap needs a sampler intervention that is not "
            "implemented yet (harness milestone 5). Run 'aligned' first - the "
            "clamp is only worth building if the direct overlap under-constrains.")


STRATEGIES = {
    s.strategy_id: s for s in (
        NoCarry(),
        DirectLatentFrame(),
        ReencodedFrame(),
        DirectLatentOverlap(),
        GeneratedOverlapAsVideo2(),
        CompositeSource(),
        ClampedTargetOverlap(),
    )
}


def get_strategy(strategy_id):
    try:
        return STRATEGIES[strategy_id]
    except KeyError:
        raise StrategyUnavailable(
            "unknown carry strategy %r (known: %s)"
            % (strategy_id, ", ".join(sorted(STRATEGIES)))) from None
