import hydra
from al4pde.acquisition.distance_pool_based import build_distance_pool_based


def build_strategy(task, cfg):
    if cfg._target_ == "al4pde.acquisition.distance_pool_based.DistancePoolBased":
        return build_distance_pool_based(task, cfg)
    else:
        return hydra.utils.instantiate(cfg, task=task)
