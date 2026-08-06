import logging
from .session import LOG_PREFIX


def make_outer_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        session.begin()
        try:
            return executor(*args, **kwargs)
        finally:
            path = session.end()
            if path:
                logging.info("%s report -> %s", LOG_PREFIX, path)
    return wrapper


def make_diffusion_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options")
        if transformer_options is None:
            if session.config.strict:
                raise RuntimeError(f"{LOG_PREFIX}: missing transformer_options")
            return executor(*args, **kwargs)
        session.prepare_forward(transformer_options)
        return executor(*args, **kwargs)
    return wrapper


def make_block_replacement(session, index):
    def replacement(args, extra):
        return {"img": session.block(index, args, extra["original_block"])}
    replacement._h3_block_cache = True
    replacement._h3_block_index = index
    return replacement
