
def build_wrapper(task, cfg, name=""):
    if cfg._target_ == "al4pde.models.fno_wrapper.FNOWrapper":
        from al4pde.models.fno_wrapper import build_fno_wrapper
        return build_fno_wrapper(task, cfg, name)
    elif cfg._target_ == "al4pde.models.arena_wrapper.ArenaWrapper":
        from al4pde.models.arena_wrapper import build_arena_wrapper
        return build_arena_wrapper(task, cfg, name)
    else:
        raise ValueError(cfg._target_)
