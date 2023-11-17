import os
import random

import numpy as np
import torch
from pettingzoo.mpe import simple_spread_v3


def make_mpe_env(env_id, seed):
    """
    Generate environment with seeding/wrapping.

    Args:
        env_id: ID of the environment to use.
        seed: Seed to set for environment.
        max_episode_len: Episode timeout length.

    Returns:
        Generated environment.

    """
    if env_id == "Simple_Spread":
        env = simple_spread_v3.parallel_env(N=2)
    else:
        raise ValueError("Unrecognized MPE!")

    return env


def save(
    run_id,
    checkpoint_dir,
    global_step,
    models,
    optimizers,
    replay_buffer,
    rng_states,
):
    """
    Saves a checkpoint.

    Args:
        run_id: Wandb ID of run.
        checkpoint_dir: Directory to store checkpoint in.
        global_step: Timestep of training.
        models: State dict of models.
        optimizers: State dict of optimizers.
        replay_buffer: Replay buffer.
        rng_states: RNG states.
    """
    save_dir = checkpoint_dir + run_id + "/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # Prevent permission issues when writing to this directory
        # after resuming a training job
        os.chmod(save_dir, 0o777)

    save_path = save_dir + "global_step_" + str(global_step) + ".pth"
    print("Saving checkpoint: " + save_path, flush=True)
    torch.save(
        {
            "global_step": global_step,
            "model_state_dict": models,
            "optimizer_state_dict": optimizers,
            "replay_buffer": replay_buffer,
            "rng_states": rng_states,
        },
        save_path,
    )


def set_seed(seed, device):
    """
    Sets seeding for experiment.

    Args:
        seed: Seed.
        device: Device being used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
