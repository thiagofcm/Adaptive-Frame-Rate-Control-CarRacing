"""
eval_nav_vector.py

Evaluate a checkpoint trained by train_nav_vector.py: the plain-MLP
navigation policy that consumes the 11-dim cautious-variables vector
(no CNN, no images). Modeled on utils/cautious_variables.py's own CLI
(reuses its Agent-free plot_track_trajectory helper and debug-text overlay
pattern) and eval_adaptive_fps_track_aware.py's conventions.

Usage:
    python -m experiments.navigation_vector.eval_nav_vector --ckpt runs/navigation_vector/<run>/final.pt
"""

import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

import envs.car_racing_random_spawn  # noqa: F401 -- registers "CarRacing_RandomSpawn"
from wrappers.cautious_vars_wrapper import CautiousVarsWrapper
from utils.cautious_variables import plot_track_trajectory, draw_debug_text


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Must match experiments/navigation_vector/train_nav_vector.py's Agent exactly."""

    def __init__(self, obs_dim=11, n_actions=5):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_action_and_value(self, x, deterministic=False):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def make_eval_env(env_id, constant_speed, max_episode_steps):
    env = gym.make(env_id, continuous=False, render_mode="rgb_array", constant_speed=constant_speed)
    env = CautiousVarsWrapper(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    p.add_argument("--env-id", default="CarRacing_RandomSpawn")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--constant-velocity", action="store_true")
    p.add_argument("--constant-speed-value", type=float, default=30.0)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--out-dir", default="experiments/navigation_vector/eval")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent().to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = checkpoint.get("agent_state_dict", checkpoint)
    agent.load_state_dict(sd)
    agent.eval()
    print(f"Loaded checkpoint from {args.ckpt} "
          f"(iteration {checkpoint.get('iteration', '?')}, global_step {checkpoint.get('global_step', '?')})")

    constant_speed = args.constant_speed_value if args.constant_velocity else None
    returns, lengths = [], []

    for ep in range(args.episodes):
        env = make_eval_env(args.env_id, constant_speed, args.max_episode_steps)
        obs, _ = env.reset(seed=args.seed + ep)

        positions, speeds, frames, cautious_log = [], [], [], []
        ep_return, ep_len, done = 0.0, 0, False

        while not done:
            x, y = env.unwrapped.car.hull.position
            vx, vy = env.unwrapped.car.hull.linearVelocity
            positions.append((x, y))
            speeds.append(float(np.hypot(vx, vy)))

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t, deterministic=args.deterministic)

            obs, reward, terminated, truncated, info = env.step(action.item())
            cautious_log.append(obs.copy())
            if args.save_video:
                frames.append(env.unwrapped.render())

            ep_return += reward
            ep_len += 1
            done = terminated or truncated

        print(f"episode {ep}: return={ep_return:7.1f}  length={ep_len}")
        returns.append(ep_return)
        lengths.append(ep_len)

        plot_track_trajectory(
            env, positions, speeds,
            save_path=os.path.join(args.out_dir, f"traj_ep{ep}_seed{args.seed + ep}_ret{int(ep_return)}.png"),
            title=f"episode={ep}  return={ep_return:.0f}  len={ep_len}",
        )

        if args.save_video and frames:
            import imageio.v2 as imageio

            annotated = []
            for idx, (frame, c) in enumerate(zip(frames, cautious_log)):
                (vx_n, vy_n, dist_n, severity_n, heading_n, cross_n,
                 cross_rate_n, off_n, time_off_n, completion_n, curves_n) = c
                annotated.append(draw_debug_text(frame, {
                    "vx": vx_n, "vy": vy_n, "dist_to_curve": dist_n,
                    "cross_track": cross_n, "off_track": off_n,
                }, idx + 1))
            video_path = os.path.join(args.out_dir, f"nav_vector_ep{ep}_ret{int(ep_return)}.mp4")
            imageio.mimsave(video_path, annotated, fps=30, codec="libx264",
                             pixelformat="yuv420p", macro_block_size=1)
            print(f"Saved video → {video_path}")

        env.close()

    print(f"\nMean return: {np.mean(returns):.1f} ± {np.std(returns):.1f}  over {args.episodes} episodes")
    print(f"Mean length: {np.mean(lengths):.0f}")


if __name__ == "__main__":
    main()
