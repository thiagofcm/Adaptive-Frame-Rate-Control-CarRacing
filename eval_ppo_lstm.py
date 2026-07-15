import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from ppo_lstm import Agent
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.adaptive_fps import AdaptiveFPSWrapper

FPS_CHOICES = [1, 5, 10, 25, 50]

def make_eval_env(env_id, nav_model_path, frame_cost, budget, max_episode_steps):
    env = gym.make(env_id, continuous=False, render_mode="rgb_array", max_episode_steps=None)
    from gymnasium.wrappers import TimeLimit as _TL
    while isinstance(env, _TL):
        env = env.env
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    env = AdaptiveFPSWrapper(env, nav_model_path, frame_cost, budget)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def plot_track_fps(env, positions, fps_values, save_path, title=None):
    track = np.asarray(env.unwrapped.track, dtype=np.float32)
    tx, ty = track[:, 2], track[:, 3]
    traj = np.asarray(positions, dtype=np.float32)
    fps  = np.asarray(fps_values, dtype=np.float32)

    idx_map = {v: i for i, v in enumerate(FPS_CHOICES)}
    idx = np.array([idx_map[int(f)] for f in fps])
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(FPS_CHOICES))))

    fig, ax = plt.subplots(figsize=(7, 7))
    txc, tyc = np.append(tx, tx[0]), np.append(ty, ty[0])
    ax.plot(txc, tyc, color="0.75", lw=8, solid_capstyle="round", zorder=1)
    ax.plot(txc, tyc, "k--", lw=1, zorder=2)
    ax.scatter(traj[:, 0], traj[:, 1], c=idx, cmap=cmap,
               vmin=-0.5, vmax=len(FPS_CHOICES) - 0.5, s=12, zorder=3)
    goal_xy = track[int(0.95 * len(track)), 2:4]
    ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=20,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6)
    ax.plot(traj[0, 0], traj[0, 1], "o", color="lime", ms=11, zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], "X", color="red", ms=11, zorder=5)
    handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                      markerfacecolor=cmap(i), markeredgecolor="none",
                      label=f"{FPS_CHOICES[i]} FPS") for i in range(len(FPS_CHOICES))]
    handles += [
        Line2D([0], [0], marker="o", linestyle="", markersize=8, markerfacecolor="lime", label="start"),
        Line2D([0], [0], marker="X", linestyle="", markersize=8, markerfacecolor="red", label="end"),
        Line2D([0], [0], marker="*", linestyle="", markersize=10, markerfacecolor="gold", label="goal"),
    ]
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.0), frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env-id", default="CarRacing-v3")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--force-fps", type=int, default=None, help="Force FPS index 0-4")
    p.add_argument("--frame-cost", type=float, default=0.4)
    p.add_argument("--budget", type=float, default=80.0)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--out-dir", default="eval_fps_plots")
    p.add_argument("--save-video", action="store_true", help="Save an MP4 of the episode")
    p.add_argument("--video-fps", type=int, default=30, help="Playback FPS of the saved video")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nav_model_path = "runs/CarRacing-v3__ppo__1__1781901069/final.pt"

    class _Dummy:
        single_action_space = gym.spaces.Discrete(5)
    agent = Agent(_Dummy()).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    agent.load_state_dict(sd)
    agent.eval()

    env = make_eval_env(args.env_id, nav_model_path, args.frame_cost, args.budget, args.max_episode_steps)
    obs, _ = env.reset(seed=args.seed)
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size).to(device),
    )
    done_t = torch.zeros(1).to(device)

    positions, fps_values, frames = [], [], []
    ep_return, ep_len, done = 0.0, 0, False

    while not done:
        with torch.no_grad():
            if args.force_fps is not None:
                action = torch.tensor([args.force_fps], device=device)
                _, _, _, _, lstm_state = agent.get_action_and_value(obs_t, lstm_state, done_t, action)
            else:
                action, _, _, _, lstm_state = agent.get_action_and_value(obs_t, lstm_state, done_t, deterministic=True)

        obs, r, term, trunc, info = env.step(action.item())
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        x, y = env.unwrapped.car.hull.position
        positions.append((x, y))
        fps_values.append(info["chosen_fps"])

        if args.save_video:
            frame = env.unwrapped.render()          # full-res RGB frame from base env
            frames.append(frame)

        ep_return += r
        ep_len += 1
        done = term or trunc
        done_t = torch.tensor([float(done)], device=device)

    tag = f"forced_fps{FPS_CHOICES[args.force_fps]}" if args.force_fps is not None else "learned"
    print(f"{tag}: return={ep_return:.1f}  length={ep_len}  reached_goal={info.get('reached_goal')}")

    plot_track_fps(env, positions, fps_values,
                   save_path=os.path.join(args.out_dir, f"fps_traj_{tag}_seed{args.seed}_ret{int(ep_return)}.png"),
                   title=f"{tag}  return={ep_return:.0f}  len={ep_len}")

    if args.save_video and frames:
        video_path = os.path.join(args.out_dir, f"video_{tag}_seed{args.seed}_ret{int(ep_return)}.mp4")
        imageio.mimsave(video_path, frames, fps=args.video_fps,
                        codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        print(f"  saved video → {video_path}  ({len(frames)} frames)")

    env.close()

if __name__ == "__main__":
    main()