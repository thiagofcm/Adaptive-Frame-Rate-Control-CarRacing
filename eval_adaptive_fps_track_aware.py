"""
eval_adaptive_fps_track_aware.py

Evaluator for the AdaptiveFPS_TrackAware_Wrapper pipeline (env: CarRacing_VarFramerate,
wrapper: wrappers/adaptive_fps_track_aware_wrapper.py). Two modes:

  --fixed-fps {1,5,10,25,50}   run the frozen NavModel at a single, fixed FPS
  --adaptive-ckpt PATH         run a trained FPS-selection policy checkpoint
                                (train_adaptive_fps_track_aware_lstm.py's Agent)

wrapper.step() is one *physics tick* (not a decision window), and the FPS action
is only actually consumed by the wrapper at sampling instants -- in between, the
11-dim cautious-var block of the observation is held stale (unchanged) rather than
recomputed. At --fixed-fps 1 the vast majority of on-screen ticks will show a
"STALE" cautious reading, by design -- that's the sensing-frequency effect this
whole pipeline is about, not a bug in this eval script.

Freshness is read from info["episode_frame_count"]: it only increments on ticks
where the wrapper actually sampled a new observation, so comparing it to the
previous tick's value is a reliable fresh/stale signal from outside the wrapper.

Reference/pattern followed: old/experiments/highest_fps_cautious/eval_highest.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import envs.car_racing_var_fps  # noqa: F401 -- registers "CarRacing_VarFramerate"
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.adaptive_fps_track_aware_wrapper import AdaptiveFPS_TrackAware_Wrapper
from training.train_adaptive_fps_track_aware_lstm import Agent as AdaptiveAgent

FPS_CHOICES = [1, 5, 10, 25, 50]
NAV_MODEL_PATH = "old/experiments/navigation/runs/CarRacing-v3__ppo__1__1781901069/final.pt"

# Must match the 11-dim layout returned by CautiousVars.get_cautious_var() -- note
# frame_counter is NOT in here, it's in the wrapper's augmented block (last 3 obs dims).
CAUTIOUS_LABELS = [
    "vx", "vy", "dist_to_curve", "curve_severity", "heading_alignment", "cross_track",
    "cross_track_rate", "off_track", "time_off_track",
    "episode_completion", "curves_passed",
]


def make_eval_env(env_id, nav_model_path, frame_cost, budget, max_episode_steps):
    env = gym.make(env_id, continuous=False, render_mode="rgb_array")
    while isinstance(env, TimeLimit):          # strip any built-in TimeLimit
        env = env.env
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    # NavModel stays on CPU regardless of --device: single-sample inference every
    # physics tick is dominated by call overhead, not compute, so CPU is faster here.
    env = AdaptiveFPS_TrackAware_Wrapper(env, nav_model_path, device="cpu",
                                          frame_cost=frame_cost, budget=budget)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def load_adaptive_agent(ckpt_path, device):
    class _Dummy:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        single_action_space = gym.spaces.Discrete(len(FPS_CHOICES))
    agent = AdaptiveAgent(_Dummy()).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    agent.load_state_dict(sd)
    agent.eval()
    return agent


def draw_cautious_overlay(frame_rgb, step, fps, fresh, cautious_vec, ret):
    """cv2-based overlay: fresh/stale banner + all 12 cautious values. Returns a new RGB array."""
    img = frame_rgb.copy()
    h, w = img.shape[:2]

    n_lines = 3 + len(CAUTIOUS_LABELS)
    line_h = 16
    box_h = n_lines * line_h + 12
    box_w = 210
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + box_h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.65, img, 0.35, 0)

    x0, y0 = 10, 20
    fresh_color = (0, 255, 0) if fresh else (0, 165, 255)   # green=fresh, orange=stale (RGB order)
    cv2.putText(img, f"{'FRESH' if fresh else 'STALE'}  fps={fps}", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fresh_color, 1, cv2.LINE_AA)
    cv2.putText(img, f"step={step}  return={ret:.1f}", (x0, y0 + line_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (x0, y0 + line_h + 4), (x0 + box_w - 10, y0 + line_h + 4), (120, 120, 120), 1)

    for i, (label, val) in enumerate(zip(CAUTIOUS_LABELS, cautious_vec)):
        y = y0 + (i + 2) * line_h + 6
        cv2.putText(img, f"{label:<15s}{val:+.3f}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, fresh_color if fresh else (255, 255, 255),
                    1, cv2.LINE_AA)
    return img


def plot_track_fps(env, positions, fps_values, save_path, title=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    track = np.asarray(env.unwrapped.track, dtype=np.float32)
    tx, ty = track[:, 2], track[:, 3]
    traj = np.asarray(positions, dtype=np.float32)
    fps = np.asarray(fps_values, dtype=np.float32)

    idx_map = {v: i for i, v in enumerate(FPS_CHOICES)}
    idx = np.array([idx_map[int(f)] for f in fps])
    cmap = ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(FPS_CHOICES))))

    fig, ax = plt.subplots(figsize=(7, 7))
    txc, tyc = np.append(tx, tx[0]), np.append(ty, ty[0])
    ax.plot(txc, tyc, color="0.75", lw=8, solid_capstyle="round", zorder=1)
    ax.plot(txc, tyc, "k--", lw=1, zorder=2)
    ax.scatter(traj[:, 0], traj[:, 1], c=idx, cmap=cmap, vmin=-0.5, vmax=len(FPS_CHOICES) - 0.5, s=12, zorder=3)
    goal_xy = track[int(0.95 * len(track)), 2:4]
    ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=20, markeredgecolor="black", markeredgewidth=0.8, zorder=6)
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
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixed-fps", type=int, choices=FPS_CHOICES,
                       help="run the whole episode at this fixed FPS")
    mode.add_argument("--adaptive-ckpt", type=str,
                       help="path to a train_adaptive_fps_track_aware_lstm.py checkpoint")

    p.add_argument("--env-id", default="CarRacing_VarFramerate")
    p.add_argument("--nav-model-path", default=NAV_MODEL_PATH)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--frame-cost", type=float, default=0.0)
    p.add_argument("--budget", type=float, default=150)
    p.add_argument("--max-episode-steps", type=int, default=1000)
    p.add_argument("--out-dir", default="eval_adaptive_fps_track_aware")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--display", action="store_true", help="show a live cv2 window while evaluating")
    p.add_argument("--no-cuda", action="store_true", help="force CPU for the adaptive-FPS agent")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    adaptive_agent = None
    next_lstm_state = None
    if args.adaptive_ckpt is not None:
        adaptive_agent = load_adaptive_agent(args.adaptive_ckpt, device)
        next_lstm_state = (
            torch.zeros(adaptive_agent.lstm.num_layers, 1, adaptive_agent.lstm.hidden_size).to(device),
            torch.zeros(adaptive_agent.lstm.num_layers, 1, adaptive_agent.lstm.hidden_size).to(device),
        )
        next_done = torch.zeros(1).to(device)

    env = make_eval_env(args.env_id, args.nav_model_path, args.frame_cost, args.budget, args.max_episode_steps)
    obs, _ = env.reset(seed=args.seed)

    positions, fps_values, frames = [], [], []
    fps_counter = {f: 0 for f in FPS_CHOICES}
    ep_return, ep_len, done = 0.0, 0, False
    prev_episode_frame_count = 0

    if args.fixed_fps is not None:
        fixed_action = FPS_CHOICES.index(args.fixed_fps)
        tag = f"fixed_fps{args.fixed_fps}"
    else:
        tag = f"adaptive_{os.path.splitext(os.path.basename(args.adaptive_ckpt))[0]}"

    if args.display:
        cv2.namedWindow("adaptive-fps eval", cv2.WINDOW_NORMAL)

    while not done:
        if args.fixed_fps is not None:
            action = fixed_action
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, _, _, _, next_lstm_state = adaptive_agent.get_action_and_value(
                    obs_t, next_lstm_state, next_done, deterministic=True
                )
            action = int(a.item())

        obs, r, term, trunc, info = env.step(action)
        ep_return += r
        ep_len += 1
        done = term or trunc

        chosen = int(info["chosen_fps"])
        fresh = info["episode_frame_count"] != prev_episode_frame_count
        prev_episode_frame_count = info["episode_frame_count"]

        x, y = env.unwrapped.car.hull.position
        positions.append((x, y))
        fps_values.append(chosen)
        fps_counter[chosen] += 1

        print(f"step {ep_len:4d} | fps {chosen:2d} | {'FRESH' if fresh else 'STALE'} | return {ep_return:7.1f}")

        if args.display or args.save_video:
            frame = env.unwrapped.render()
            annotated = draw_cautious_overlay(frame, ep_len, chosen, fresh, obs[:11], ep_return)
            if args.save_video:
                frames.append(annotated)
            if args.display:
                cv2.imshow("adaptive-fps eval", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    done = True

    if args.display:
        cv2.destroyAllWindows()

    print(f"\n{tag}: return={ep_return:.1f}  length={ep_len}  reached_goal={info.get('reached_goal')}")
    print("FPS usage:", {f: fps_counter[f] for f in FPS_CHOICES})
    total = sum(fps_counter.values())
    print("FPS fraction:", {f: f"{fps_counter[f] / total:.1%}" for f in FPS_CHOICES})

    plot_track_fps(
        env, positions, fps_values,
        save_path=os.path.join(args.out_dir, "plots", f"plot_fps_{tag}_seed{args.seed}_ret{int(ep_return)}.png"),
        title=f"{tag}  return={ep_return:.0f}  len={ep_len}",
    )

    if args.save_video and frames:
        video_path = os.path.join(args.out_dir, "videos", f"video_{tag}_seed{args.seed}_ret{int(ep_return)}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=args.video_fps,
                         codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        print(f"saved video → {video_path}  ({len(frames)} frames)")

    env.close()


if __name__ == "__main__":
    main()
