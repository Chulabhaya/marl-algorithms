import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import wandb
from common.models import DiscreteCritic
from common.replay_buffer import ReplayBuffer
from common.utils import make_mpe_env as make_env, save, set_seed


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--exp-group", type=str, default=None,
        help="the group under which this experiment falls")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project", type=str, default="idqn",
        help="wandb project name")
    parser.add_argument("--wandb-dir", type=str, default="./",
        help="the wandb directory")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Simple_Spread",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=100500,
        help="total timesteps of the experiments")
    parser.add_argument("--num-agents", type=int, default=2,
        help="number of agents")
    parser.add_argument("--buffer-size", type=int, default=int(1e3),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network optimizer")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target networks")
    parser.add_argument("--epsilon", type=float, default=0.05,
        help="percentage of time to take random action for exploration")

    # Checkpointing specific arguments
    parser.add_argument("--save", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="checkpoint saving during training")
    parser.add_argument("--save-checkpoint-dir", type=str, default="./trained_models/",
        help="path to directory to save checkpoints in")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
        help="how often to save checkpoints during training (in timesteps)")
    parser.add_argument("--resume", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to resume training from a checkpoint")
    parser.add_argument("--resume-checkpoint-path", type=str, default="./trained_models/CartPole-v0__sac_discrete_action__1__1680268581__07jx3mba/global_step_35000.pth",
        help="path to checkpoint to resume training from")
    parser.add_argument("--run-id", type=str, default="CartPole-v0__sac_discrete_action__1__1680268581__07jx3mba",
        help="wandb unique run id for resuming")

    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}"
    wandb_id = wandb.util.generate_id()
    run_id = f"{run_name}_{wandb_id}"

    # If a unique wandb run id is given, then resume from that, otherwise
    # generate new run for resuming
    if args.resume and args.run_id is not None:
        run_id = args.run_id
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            resume="must",
            mode="offline",
        )
    else:
        wandb.init(
            id=run_id,
            dir=args.wandb_dir,
            project=args.wandb_project,
            config=vars(args),
            name=run_name,
            save_code=True,
            settings=wandb.Settings(code_dir="."),
            mode="online",
        )

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Running on the following device: " + device.type, flush=True)

    # Set seeding
    set_seed(args.seed, device)

    # Load checkpoint if resuming
    if args.resume:
        print("Resuming from checkpoint: " + args.resume_checkpoint_path, flush=True)
        checkpoint = torch.load(args.resume_checkpoint_path)

    # Set RNG state for seeds if resuming
    if args.resume:
        random.setstate(checkpoint["rng_states"]["random_rng_state"])
        np.random.set_state(checkpoint["rng_states"]["numpy_rng_state"])
        torch.set_rng_state(checkpoint["rng_states"]["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["rng_states"]["torch_cuda_rng_state"])
            torch.cuda.set_rng_state_all(
                checkpoint["rng_states"]["torch_cuda_rng_state_all"]
            )

    # Env setup
    env = make_env(args.env_id, args.seed)

    # Initialize models and optimizers
    model_config = {
        "input_size": env.observation_space("agent_0").shape[0],
        "output_size": env.action_space("agent_0").n
    }
    critics = []
    targets = []
    optimizers = []
    for agent_idx in range(args.num_agents):
        critics.append(DiscreteCritic(model_config).to(device))
        targets.append(DiscreteCritic(model_config).to(device))
        targets[agent_idx].load_state_dict(critics[agent_idx].state_dict())
        optimizers.append(optim.Adam(list(critics[agent_idx].parameters()), lr=args.q_lr))

    # # If resuming training, load models and optimizers
    # if args.resume:
    #     qf1.load_state_dict(checkpoint["model_state_dict"]["qf1_state_dict"])
    #     qf2.load_state_dict(checkpoint["model_state_dict"]["qf2_state_dict"])
    #     qf1_target.load_state_dict(
    #         checkpoint["model_state_dict"]["qf1_target_state_dict"]
    #     )
    #     qf2_target.load_state_dict(
    #         checkpoint["model_state_dict"]["qf2_target_state_dict"]
    #     )
    #     q1_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["q1_optimizer"])
    #     q2_optimizer.load_state_dict(checkpoint["optimizer_state_dict"]["q1_optimizer"])

    # Initialize replay buffers
    replay_buffers = []
    for agent_idx in range(args.num_agents):
        replay_buffers.append(
            ReplayBuffer(
                args.buffer_size,
                episodic=False,
                stateful=False,
                device=device,
            )
        )
    # # If resuming training, then load previous replay buffer
    # if args.resume:
    #     rb1_data = checkpoint["replay_buffer_1"]
    #     rb2_data = checkpoint["replay_buffer_2"]
    #     rb1.load_buffer(rb1_data)
    #     rb2.load_buffer(rb2_data)

    # Start time tracking for run
    start_time = time.time()

    # Start the game
    start_global_step = 0
    # If resuming, update starting step
    if args.resume:
        start_global_step = checkpoint["global_step"] + 1
    # Set RNG state for env
    if args.resume:
        env.np_random.bit_generator.state = checkpoint["rng_states"]["env_rng_state"]
        env.action_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_action_space_rng_state"
        ]
        env.observation_space.np_random.bit_generator.state = checkpoint["rng_states"][
            "env_obs_space_rng_state"
        ]

    # Store episodic returns
    all_episodic_returns = [0, 0]
    all_episodic_lengths = [0, 0]
    all_obs, info = env.reset(seed=args.seed)
    for global_step in range(start_global_step, args.total_timesteps):
        # Store values for data logging for each global step
        data_log = {}
        # Calculate actions for each agent
        all_actions = {}
        for agent_idx, agent in enumerate(env.agents):
            # With some percentage pick random action,
            # otherwise use Q-network
            if np.random.rand(1) < args.epsilon:
                action = env.action_space(agent).sample()
                all_actions[agent] = action
            else:
                obs = all_obs[agent]
                with torch.no_grad():
                    qf = critics[agent_idx]
                    q_values = qf(torch.tensor(obs).to(device).unsqueeze(0))
                action = torch.argmax(q_values, dim=1)
                action = action.detach().cpu().numpy()[0]
                all_actions[agent] = action

        # Take step in environment.
        all_next_obs, all_reward, all_terminated, all_truncated, all_info = env.step(all_actions)

        # Save data to replay buffer, iterate by obs keys in case agent died
        # in step so we can get final obs
        for agent_idx, agent in enumerate(all_next_obs.keys()):
            obs = all_obs[agent]
            action = all_actions[agent]
            next_obs = all_next_obs[agent]
            reward = all_reward[agent]
            terminated = all_terminated[agent]
            truncated = all_truncated[agent]
            replay_buffers[agent_idx].add(obs, action, next_obs, reward, terminated, truncated)

            all_episodic_returns[agent_idx] += reward
            all_episodic_lengths[agent_idx] += 1

        # Update next obs
        all_obs = all_next_obs

        # Handle episode end for each agent, record rewards for plotting purposes
        for agent_idx, agent in enumerate(all_next_obs.keys()):
            terminated = all_terminated[agent]
            truncated = all_truncated[agent]
            if terminated or truncated:
                print(
                    f"global_step={global_step}, agent={agent_idx}, episodic_return={all_episodic_returns[agent_idx]}, episodic_length={all_episodic_lengths[agent_idx]}",
                    flush=True,
                )
                data_log[f"misc/agent_{agent_idx}/episodic_return"] = all_episodic_returns[agent_idx]
                data_log[f"misc/agent_{agent_idx}/episodic_length"] = all_episodic_lengths[agent_idx]

        # If all agents are done, then start new episode
        if all(all_terminated.values()) or all(all_truncated.values()):
            all_obs, info = env.reset()
            all_episodic_returns = [0, 0]
            all_episodic_lengths = [0, 0]

        # ALGO LOGIC: training.
        for agent_idx in range(args.num_agents):
            # Sample data from replay buffer
            observations, actions, next_observations, rewards, terminateds = replay_buffers[agent_idx].sample(
                args.batch_size
            )
            # ---------- update critic ---------- #
            with torch.no_grad():
                # Calculate target value
                qf_target = targets[agent_idx]
                q_next_target_values = qf_target(next_observations)
                max_q_next_target_values, _ = torch.max(q_next_target_values, dim=1, keepdim=True)
                next_q_values = rewards + (
                    (1 - terminateds)
                    * args.gamma
                    * max_q_next_target_values
                )

            # calculate eq. 5 in updated SAC paper
            qf = critics[agent_idx]
            q_a_values = qf(observations).gather(1, actions)
            qf_loss = F.mse_loss(q_a_values, next_q_values)

            # calculate eq. 6 in updated SAC paper
            optimizers[agent_idx].zero_grad()
            qf_loss.backward()
            optimizers[agent_idx].step()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(
                    critics[agent_idx].parameters(), targets[agent_idx].parameters()
                ):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                data_log[f"losses/agent_{agent_idx}/qf_values"] = q_a_values.mean().item()
                data_log[f"losses/agent_{agent_idx}/qf_loss"] = qf_loss.item()
                data_log["misc/steps_per_second"] = int(
                    global_step / (time.time() - start_time)
                )
        print("SPS:", int(global_step / (time.time() - start_time)), flush=True)

        data_log["misc/global_step"] = global_step
        wandb.log(data_log, step=global_step)

        # # Save checkpoints during training
        # if args.save:
        #     if global_step % args.checkpoint_interval == 0:
        #         # Save models
        #         models = {
        #             "actor_state_dict": actor.state_dict(),
        #             "qf1_state_dict": qf1.state_dict(),
        #             "qf2_state_dict": qf2.state_dict(),
        #             "qf1_target_state_dict": qf1_target.state_dict(),
        #             "qf2_target_state_dict": qf2_target.state_dict(),
        #         }
        #         # Save optimizers
        #         optimizers = {
        #             "q_optimizer": q_optimizer.state_dict(),
        #             "actor_optimizer": actor_optimizer.state_dict(),
        #         }
        #         if args.autotune:
        #             optimizers["a_optimizer"] = a_optimizer.state_dict()
        #             models["log_alpha"] = log_alpha
        #         # Save replay buffer
        #         rb_data = rb.save_buffer()
        #         # Save random states, important for reproducibility
        #         rng_states = {
        #             "random_rng_state": random.getstate(),
        #             "numpy_rng_state": np.random.get_state(),
        #             "torch_rng_state": torch.get_rng_state(),
        #             "env_rng_state": env.np_random.bit_generator.state,
        #             "env_action_space_rng_state": env.action_space.np_random.bit_generator.state,
        #             "env_obs_space_rng_state": env.observation_space.np_random.bit_generator.state,
        #         }
        #         if device.type == "cuda":
        #             rng_states["torch_cuda_rng_state"] = torch.cuda.get_rng_state()
        #             rng_states[
        #                 "torch_cuda_rng_state_all"
        #             ] = torch.cuda.get_rng_state_all()

        #         save(
        #             run_id,
        #             args.save_checkpoint_dir,
        #             global_step,
        #             models,
        #             optimizers,
        #             rb_data,
        #             rng_states,
        #         )

    env.close()
