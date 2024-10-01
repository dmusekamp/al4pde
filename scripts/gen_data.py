import os
import hydra
from omegaconf import DictConfig, OmegaConf


def generate_data(task, path, num_batches, al_iter_id, batch_size):
    os.makedirs(path, exist_ok=True)
    for idx in range(num_batches):
        print(f"\ngenerating  batch {idx + 1}")
        ic_params = task.get_ic_params(batch_size)
        pde_params_normed = task.get_pde_params_normed(batch_size)
        pde_params = task.get_pde_params(pde_params_normed)
        ic = task.get_ic(ic_params, pde_params)
        u_trajectories, u_grid_coords, u_tcoords = task.evolve_ic(ic, pde_params)
        task.save_trajectories(u_trajectories, pde_params, u_grid_coords, u_tcoords, al_iter_id, idx, save_path=path,
                               ic_params=ic_params, pde_params_normed=pde_params_normed)


@hydra.main(version_base="1.3.2", config_path="../config", config_name="main")
def main(cfg: DictConfig):
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".10"
    import wandb
    import hydra
    import jax.numpy as jnp
    jnp.arange(0, 100)  # Start jax before loading torch, otherwise my setup fails
    import torch
    from al4pde.utils import set_current_seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from jax.lib import xla_bridge

    task = hydra.utils.instantiate(cfg.task)

    run = wandb.init(
        project=cfg.wandb.project,
        group="gen_data",
        name=task.pde_name,
        config=OmegaConf.to_container(cfg)
    )
    print(device)

    print("jax_dev", xla_bridge.get_backend().platform, flush=True)

    set_current_seed(42, 0, False, task, False)  # can just be fixed

    # test data on which the efficacy of active learning will be tested
    print("generate val data", flush=True)
    batch_size = cfg.task.data_gen.batch_size
    path = os.path.join(task.data_path, task.pde_name, "val")
    generate_data(task, path, cfg.task.data_gen.num_test_batches, "val", batch_size)

    # initial data to train ensemble
    print("generate initial data", flush=True)
    path = os.path.join(task.data_path, task.pde_name, "initial")
    generate_data(task, path, cfg.task.data_gen.num_initial_batches, "init", batch_size)

    print("generate test data", flush=True)
    batch_size = cfg.task.data_gen.batch_size
    path = os.path.join(task.data_path, task.pde_name, "test")
    generate_data(task, path, cfg.task.data_gen.num_test_batches, "test", batch_size)
    

if __name__ == "__main__":
    main()
    print("\nDone with generating data", flush=True)
