from al4pde.prob_models.ensemble import build_ensemble


def build_prob_model(task, cfg):
    if cfg._target_ == "al4pde.prob_models.ensemble.Ensemble":
        return build_ensemble(task, cfg)
    else:
        raise ValueError
