import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import argparse
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# import from THIS training script (the highest-FPS one)
from experiments.highest_fps.train_highest import Agent
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.highest_fps import HighestFPSWrapper

FPS_CHOICES = [1, 5, 10, 25, 50]
NAV_MODEL_PATH = "experiments/navigation/runs/CarRacing-v3__ppo__1__1781901069/final.pt"


def make_eval_env(env_id, max_episode_steps=1000):
    from gymnasium.wrappers import TimeLimit
    env = gym.make(env_id, continuous=False, render_mode="rgb_array", max_episode_steps=None)
    while isinstance(env, TimeLimit):
        env = env.env
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    env = HighestFPSWrapper(env, NAV_MODEL_PATH, device="cpu")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def load_agent(ckpt_path, device):
    class _Dummy:
        single_observation_space = gym.spaces.Box(0, 255, shape=(4, 96, 96), dtype=np.float32)
        single_action_space = gym.spaces.Discrete(5)
    agent = Agent(_Dummy()).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["agent_state_dict"] if isinstance(ckpt, dict) and "agent_state_dict" in ckpt else ckpt
    agent.load_state_dict(sd)
    agent.eval()
    return agent


def annotate(frame, step, fps, ret):
    import cv2
    img = frame.copy()
    lines = [f"step: {step}", f"FPS: {fps}", f"return: {ret:.1f}"]
    box_h = 14 * len(lines) + 8
    max_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0] for t in lines)
    cv2.rectangle(img, (2, 2), (2 + max_w + 8, 2 + box_h), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (6, 18 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def run_episode(env, agent, device, seed, save_video):
    obs, _ = env.reset(seed=seed)
    ep_return, ep_len, done = 0.0, 0, False
    positions, fps_values, frames = [], [], []

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            #action, _, _, _ = agent.get_action_and_value(obs_t, deterministic=True)
            action = 4
        obs, r, term, trunc, info = env.step(action)
        ep_return += r
        ep_len += 1
        done = term or trunc

        x, y = env.unwrapped.car.hull.position
        positions.append((x, y))
        fps_values.append(info["chosen_fps"])
        if save_video:
            frames.append(annotate(env.unwrapped.render(), ep_len, info["chosen_fps"], ep_return))

    return ep_return, ep_len, positions, fps_values, frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env-id", default="CarRacing-v3")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="experiments/highest_fps/eval")
    p.add_argument("--save-video", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(args.ckpt, device)
    env = make_eval_env(args.env_id)

    ret, length, positions, fps_values, frames = run_episode(env, agent, device, args.seed, args.save_video)

    fps_arr = np.array(fps_values)
    counts = {f: int((fps_arr == f).sum()) for f in FPS_CHOICES}
    print(f"return={ret:.1f}  length={length}")
    print("FPS usage:", counts)

    # trajectory colored by chosen FPS
    track = np.asarray(env.unwrapped.track, dtype=np.float32)
    tx, ty = np.append(track[:, 2], track[0, 2]), np.append(track[:, 3], track[0, 3])
    traj = np.asarray(positions)
    idx_map = {v: i for i, v in enumerate(FPS_CHOICES)}
    idx = np.array([idx_map[int(f)] for f in fps_values])
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(FPS_CHOICES))))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(tx, ty, color="0.75", lw=8, solid_capstyle="round", zorder=1)
    ax.plot(tx, ty, "k--", lw=1, zorder=2)
    ax.scatter(traj[:, 0], traj[:, 1], c=idx, cmap=cmap, vmin=-0.5, vmax=len(FPS_CHOICES)-0.5, s=12, zorder=3)
    ax.plot(traj[0, 0], traj[0, 1], "o", color="lime", ms=11, zorder=5)
    ax.plot(traj[-1, 0], traj[-1, 1], "X", color="red", ms=11, zorder=5)
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=cmap(i),
                      label=f"{FPS_CHOICES[i]} FPS") for i in range(len(FPS_CHOICES))]
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"return={ret:.0f}  len={length}")
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    plot_path = os.path.join(args.out_dir, f"traj_seed{args.seed}_ret{int(ret)}.png")
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {plot_path}")

    env.close()

    if args.save_video and frames:
        vp = os.path.join(args.out_dir, f"video_seed{args.seed}_ret{int(ret)}.mp4")
        imageio.mimsave(vp, frames, fps=30, codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        print(f"saved {vp}")


if __name__ == "__main__":
    main()