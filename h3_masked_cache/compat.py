"""Cross-feature guards for output-neutral H3 masked measurements."""


def approximation_contamination_reason(transformer_options):
    """Return why approximate optimizer state invalidates a Stage-0 trajectory."""
    status = transformer_options.get("minimax_h3_memory_optimizer")
    if isinstance(status, dict):
        if status.get("attention_approximate") or status.get("attention_selected") == "sol_attn":
            return "Sol-Attn is active; sparse attention contaminates the Stage-0 trajectory"
        cache_mode = status.get("block_cache_mode")
        if status.get("block_cache_approximate") or cache_mode not in (None, "off"):
            return "FirstBlockCache is active; block reuse contaminates the Stage-0 trajectory"
    if transformer_options.get("minimax_h3_sol_backend") is not None:
        return "Sol-Attn is active; sparse attention contaminates the Stage-0 trajectory"
    if transformer_options.get("minimax_h3_first_block_cache") is not None:
        return "FirstBlockCache is active; block reuse contaminates the Stage-0 trajectory"
    return None
