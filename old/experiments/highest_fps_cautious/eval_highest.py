import os
from tkinter import Image
os.environ["SDL_VIDEODRIVER"] = "dummy"
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import imageio.v2 as imageio
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from old.experiments.highest_fps_cautious.train_highest import Agent
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.highest_fps_cautious import HighestFPS_Cautious_Wrapper
from PIL import Image, ImageDraw, ImageFont

FPS_CHOICES = [1, 5, 10, 25, 50]
NAV_MODEL_PATH = "experiments/navigation/runs/CarRacing-v3__ppo__1__1781901069/final.pt"


def make_eval_env(env_id, nav_model_path, max_episode_steps):
    env = gym.make(env_id, continuous=False, render_mode="rgb_array", max_episode_steps=None)
    while isinstance(env, TimeLimit):          # strip any built-in TimeLimit
        env = env.env
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    env = HighestFPS_Cautious_Wrapper(env, nav_model_path, device="cpu")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def load_agent(ckpt_path, device):
    class _Dummy:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        single_action_space = gym.spaces.Discrete(5)
    agent = Agent(_Dummy()).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["agent_state_dict"] if "agent_state_dict" in ckpt else ckpt
    agent.load_state_dict(sd)
    agent.eval()
    return agent


def annotate(frame, step, fps, cross_track, time_to_curve, ret):
    """Draw eval info onto an RGB frame (returns a new array)."""
    img = frame.copy()
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    lines = [
        f"step: {step}",
        f"FPS: {fps}",
        f"cross_track: {cross_track:+.3f}",
        f"time_to_curve: {time_to_curve:.3f}",
        f"return: {ret:.1f}",
    ]

    x, y = 10, 10
    line_height = 18

    # black background box
    box_w = 230
    box_h = line_height * len(lines) + 10
    draw.rectangle([x - 5, y - 5, x + box_w, y + box_h], fill=(0, 0, 0))

    for i, line in enumerate(lines):
        draw.text((x, y + i * line_height), line, fill=(255, 255, 255))

    return np.array(img)

    # for i, txt in enumerate(lines):
    #     cv2.putText(img, txt, (5, 15 + i * 14),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    # return img


def plot_track_fps(env, positions, fps_values, save_path, title=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    ax.scatter(traj[:, 0], traj[:, 1], c=idx, cmap=cmap,vmin=-0.5, vmax=len(FPS_CHOICES) - 0.5, s=12, zorder=3)
    goal_xy = track[int(0.95 * len(track)), 2:4]
    ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=20,markeredgecolor="black", markeredgewidth=0.8, zorder=6)
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
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--out-dir", default="experiments/highest_fps_cautious/eval")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = load_agent(args.ckpt, device)
    env = make_eval_env(args.env_id, NAV_MODEL_PATH, args.max_episode_steps)
    obs, _ = env.reset(seed=args.seed)

    positions, fps_values, frames = [], [], []
    fps_counter = {f: 0 for f in FPS_CHOICES}     # tally of how often each FPS chosen
    ep_return, ep_len, done = 0.0, 0, False

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if args.force_fps is not None:
            action = args.force_fps
        else:
            with torch.no_grad():
                a, _, _, _ = agent.get_action_and_value(obs_t, deterministic=True)
            action = int(a.item())

        obs, r, term, trunc, info = env.step(action)
        ep_return += r
        ep_len += 1
        done = term or trunc

        chosen = info["chosen_fps"]
        x, y = env.unwrapped.car.hull.position
        positions.append((x, y))
        fps_values.append(chosen)
        fps_counter[int(chosen)] += 1

        print(f"step {ep_len:4d} | FPS {chosen:2d} | return {ep_return:7.1f}")

        if args.save_video:
            frame = env.unwrapped.render()
            frames.append(annotate(
                frame, ep_len, chosen,
                cross_track=obs[6],          # index 6 in your 8-vector
                time_to_curve=obs[5],        # index 5
                ret=ep_return,
            ))

    tag = f"forced_fps{FPS_CHOICES[args.force_fps]}" if args.force_fps is not None else "highest_cautious"
    print(f"\n{tag}: return={ep_return:.1f}  length={ep_len}  reached_goal={info.get('reached_goal')}")
    print("FPS usage:", {f: fps_counter[f] for f in FPS_CHOICES})
    total = sum(fps_counter.values())
    print("FPS fraction:", {f: f"{fps_counter[f]/total:.1%}" for f in FPS_CHOICES})

    plot_track_fps(env, positions, fps_values,
                   save_path=os.path.join(args.out_dir, "plots", f"plot_fps_{tag}_seed{args.seed}_ret{int(ep_return)}.png"),
                   title=f"{tag}  return={ep_return:.0f}  len={ep_len}")
    
    if args.save_video and frames:
        video_path = os.path.join(args.out_dir, "videos", f"video_{tag}_seed{args.seed}_ret{int(ep_return)}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=args.video_fps,
                        codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        print(f"saved video → {video_path}  ({len(frames)} frames)")

    env.close()


if __name__ == "__main__":
    main()