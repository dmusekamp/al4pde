import os
import torch
import glob
import numpy as np
import random


def bxtc_to_btcx(traj):
    # pdebench to pdearena
    return traj.permute([0, -2, -1] + list(range(1, len(traj.shape) - 2)))


def btcx_to_bxtc(traj):
    # pdearena to pdebench
    return traj.permute([0,] + list(range(3, len(traj.shape))) + [1, 2])


def subsample_trajectory(traj, reduced_resolution, reduced_resolution_t):
    if len(traj.shape) == 4:  # 1d
        return traj[:, ::reduced_resolution, ::reduced_resolution_t]
    elif len(traj.shape) == 5:  # 2d
        return traj[:, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution_t]
    elif len(traj.shape) == 6:  # 3d
        return traj[:, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution_t]
    raise ValueError("Only up to 3d supported")


def subsample_grid(grid, reduced_resolution):
    if len(grid.shape) == 3:  # 1d
        return grid[:, ::reduced_resolution]
    elif len(grid.shape) == 4:  # 2d
        return grid[:, ::reduced_resolution, ::reduced_resolution]
    elif len(grid.shape) == 5:  # 3d
        return grid[:, ::reduced_resolution, ::reduced_resolution, ::reduced_resolution]
    raise ValueError("Only up to 3d supported")


def subsample(traj, grid, reduced_resolution, reduced_resolution_t):
    traj = subsample_trajectory(traj, reduced_resolution, reduced_resolution_t)
    grid = subsample_grid(grid, reduced_resolution)
    return traj, grid


def set_random_seed(seed: int = 23) -> int:
    """Set seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"\nThe random seed is set as : {seed}")

    return seed


def save_checkpoint(run_save_path, model, al_iter, sampling_finished):
    checkpoint_dir = os.path.join(run_save_path, "checkpoints",)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir,  str(al_iter) + ".pt")
    save_dict = {"model": model.state_dict(), "al_iter": al_iter, "sampling_finished": sampling_finished}
    torch.save(save_dict, checkpoint_path)


def load_checkpoint(run_save_path, checkpoint_file_name, model, train=True, load_train_data=True):
    save_dict = torch.load(os.path.join(run_save_path, "checkpoints", checkpoint_file_name))
    model.init_training(load_train_data=load_train_data)
    model.load_state_dict(save_dict['model'])
    data_path = os.path.join(run_save_path, "data")
    npy_filenames = glob.glob("*_alstp" + "*param.npy", root_dir=data_path)
    al_iter = [int(f.split("_")[-3][5:]) for f in npy_filenames]

    data_iter = max(al_iter) if len(al_iter) > 0 else 0
    if train:
        if data_iter > save_dict["al_iter"]:
            raise Exception("current dataset contains data generated with later models")
        if not save_dict["sampling_finished"]:
            npy_filenames = glob.glob("*_alstp" + str(save_dict["al_iter"]) + "*.npy",
                                      root_dir=data_path)
            for file in npy_filenames:
                del_path = os.path.join(data_path, file)
                print("delete data", del_path)
                os.remove(os.path.join(data_path, file))
    print("restored checkpoint", checkpoint_file_name,
          "from active learning iteration", save_dict["al_iter"],
          ".Sampling finished: ", save_dict["sampling_finished"])
    return save_dict["al_iter"], save_dict["sampling_finished"]


def get_current_seed(start_seed, al_iter, sampling_finished):
    rng = torch.Generator().manual_seed(start_seed)
    for i in range(0, al_iter + 2): # iterate so that changig num_al_iter does not change seeding of the first iters
        # + 2 because al_iter of initial data generation == -1
        seeds = torch.randint(0, torch.iinfo(torch.int).max, (2, 2), generator=rng)
    return int(seeds[int(sampling_finished), 0]), int(seeds[int(sampling_finished), 1])


def set_current_seed(start_seed, al_iter, sampling_finished, task, use_test):
    if use_test:
        start_seed *= 333
    global_seed, task_seed = get_current_seed(start_seed, al_iter, sampling_finished)
    print(task_seed)
    set_random_seed(global_seed)

    if task is not None:
        task.set_seed(task_seed)