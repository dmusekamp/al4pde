import shutil
import hydra
from omegaconf import OmegaConf, DictConfig
import wandb
import os
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".10"
import torch
import jax.numpy as jnp
jnp.arange(0, 100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from jax.lib import xla_bridge

from al4pde.utils import load_checkpoint, save_checkpoint, set_current_seed
from al4pde.evaluation.visualization import end_of_al_iter_plots
from scripts.gen_data import generate_data
from al4pde.prob_models.build_prob_model import build_prob_model
from al4pde.acquisition.build_selection import build_strategy


@hydra.main(version_base="1.3.2", config_path="../config", config_name="main")
def main(cfg: DictConfig):

    run = wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg)
    )

    if cfg.restore_checkpoint:
        run_id = cfg.checkpoint_id
    else:
        run_id = run.id

    print("run_id", run_id, flush=True)
    print("torch_device", device)
    print("jax_dev", xla_bridge.get_backend().platform, flush=True)

    use_test = cfg.task.use_test
    if use_test:
        print("using test set")

    # init components
    set_current_seed(cfg.seed, 0, sampling_finished=True, task=None, use_test=use_test)
    run_save_path = os.path.join(cfg.task.run_save_path, run_id)
    task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)
    acq_strat = build_strategy(task, cfg.acquisition)
    print("acq", acq_strat, flush=True)
    prob_model = build_prob_model(task, cfg.prob_model)

    if cfg.wandb.name is None:
        run.name = acq_strat.name + "_" + str(cfg.seed)

    num_epoch = cfg.num_epochs
    first_al_iter = -1
    if cfg.restore_checkpoint:

        first_al_iter, sampling_finished = load_checkpoint(run_save_path, cfg.checkpoint_file_name, prob_model)

        if not sampling_finished:
            set_current_seed(cfg.seed, first_al_iter, False, task=task)
            acq_strat.generate(prob_model, first_al_iter)
            save_checkpoint(run_save_path, prob_model, first_al_iter, True)
            end_of_al_iter_plots(task, prob_model, first_al_iter)
            first_al_iter += 1

    # generate initial data:
    if first_al_iter == -1:
        set_current_seed(cfg.seed, -1, sampling_finished=False, task=task, use_test=use_test)
        if "pregen_initial" in cfg and cfg.pregen_initial:
            print("USING PREGENERATED INITIAL DATA - ONLY FOR DEBUGGING")
            assert cfg.seed == 23  # To avoid forgetting regenerating if using multiple seeds

            initial_data_path = os.path.join(task.data_path, task.pde_name, "initial")
            shutil.copytree(initial_data_path, task.traj_save_path, dirs_exist_ok=True)

        else:
            generate_data(task, task.traj_save_path, cfg.task.data_gen.num_initial_batches, "init",
                          cfg.task.data_gen.batch_size)
        first_al_iter = 0

    for al_iter in range(first_al_iter, cfg.num_al_iter):
        is_last = al_iter == cfg.num_al_iter - 1

        print("\nactive learning iteration " + str(al_iter), flush=True)

        # retrain ensemble model
        set_current_seed(cfg.seed, al_iter, sampling_finished=True, task=task,  use_test=use_test)
        prob_model.train_n_epoch(al_iter, num_epoch, num_epoch * al_iter, is_last=is_last)

        wandb.log({"al/al_iter": al_iter, "al/num_data": len(prob_model.train_loader_full_traj.dataset)})
        save_checkpoint(run_save_path, prob_model, al_iter, False)
        
        # generate data
        set_current_seed(cfg.seed, al_iter, sampling_finished=False, task=task, use_test=use_test)

        if not is_last:
            acq_strat.generate(prob_model, al_iter, prob_model.train_loader_full_traj)
            save_checkpoint(run_save_path, prob_model, al_iter, True)

        end_of_al_iter_plots(task, prob_model, al_iter, last=is_last)


if __name__ == "__main__":
    main()
    print("Done with active learning experiment.", flush=True)
