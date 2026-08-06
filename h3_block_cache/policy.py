def decide_action(config, entry, step, total_steps):
    if config.mode in ("observe", "shadow"):
        return "refresh", config.mode
    if not entry.valid:
        return "refresh", "cache-miss"
    if step < config.warmup_steps:
        return "refresh", "warmup"
    if total_steps and config.force_refresh_last_steps:
        if step >= max(0, total_steps - config.force_refresh_last_steps):
            return "refresh", "final-refresh"
    if entry.reuse_count >= config.max_reuse_span:
        return "refresh", "max-reuse-span"
    if step % config.refresh_interval == 0:
        return "refresh", "interval"
    return "reuse", "fixed-schedule"
