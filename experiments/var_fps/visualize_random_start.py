"""
visualize_random_start.py

Visual smoke-test for AdaptiveFPS_Random_Initial_Pos + CarRacing_VarFramerate_RandomStart.
Runs several episodes (different seeds) and plots, one panel per episode (each episode
regenerates its own procedural track, so they can't be usefully overlaid on one outline):
the track, the spawn point with a heading arrow, the goal point, and a short trajectory
right after spawn. If the car is correctly oriented, the trajectory moves along the
track direction immediately -- not sideways or backward -- which is the main thing to
visually check here.

--save-video-seed additionally renders a real video (car's-eye rendered frames, not the
top-down track plot) for one specific seed -- useful for diagnosing a panel that shows
little/no trajectory movement: is the car actually stuck, or did the frozen nav model
just pick a non-moving action (e.g. steer-only, no throttle) at that particular spawn?

Usage:
  python experiments/var_fps/visualize_random_start.py \\
      --nav-model-path experiments/navigation/CarRacing-v3__train__1__1785768450/final.pt

  python experiments/var_fps/visualize_random_start.py \\
      --nav-model-path experiments/navigation/CarRacing-v3__train__1__1785768450/final.pt \\
      --save-video-seed 1 --video-max-ticks 300
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import argparse
import math

import numpy as np
import cv2
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym

import envs.car_racing_var_fps_random_start  # noqa: F401 -- registers "CarRacing_VarFramerate_RandomStart"
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.adaptive_fps_random_initial_pos import AdaptiveFPS_Random_Initial_Pos

FPS_CHOICES = [1, 5, 10, 25, 50]
# Discrete CarRacing-v3 action labels -- see gymnasium's car_racing.py docstring:
# 0: do nothing, 1: steer right, 2: steer left, 3: gas, 4: brake. Only 3 (gas) applies
# any throttle -- 0/1/2/4 alone, with the car starting at rest, produce zero net motion.
DISCRETE_ACTION_LABELS = {0: "noop", 1: "steer_right", 2: "steer_left", 3: "gas", 4: "brake"}


def make_env(nav_model_path, frame_cost, budget):
    # No TimeLimit here -- run_episode() caps its own loop via trajectory_ticks, and
    # TimeLimit would sit between us and AdaptiveFPS_Random_Initial_Pos, blocking
    # direct attribute access to its goal_xy/spawn_idx-derived fields below.
    env = gym.make("CarRacing_VarFramerate_RandomStart", continuous=False, render_mode="rgb_array")
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    env = AdaptiveFPS_Random_Initial_Pos(env, nav_model_path, device="cpu",
                                          frame_cost=frame_cost, budget=budget)
    return env


def run_episode(env, seed, fps, trajectory_ticks):
    obs, info = env.reset(seed=seed)
    action = FPS_CHOICES.index(fps)

    spawn_idx = int(env.unwrapped.spawn_idx)
    beta = float(env.unwrapped.track[spawn_idx][1])
    # true heading = beta + 90deg -- see utils/cautious_variables.py's heading_alignment
    # comment: car.hull.angle spawns equal to beta directly, but beta itself is in the
    # track generator's frame, rotated -90deg from the car's true direction of travel.
    heading = beta + math.pi / 2

    positions = [tuple(env.unwrapped.car.hull.position)]
    done = False
    tick = 0
    while not done and tick < trajectory_ticks:
        obs, r, term, trunc, info = env.step(action)
        positions.append(tuple(env.unwrapped.car.hull.position))
        done = term or trunc
        tick += 1

    return {
        "seed": seed,
        "spawn_idx": spawn_idx,
        "heading": heading,
        "goal_xy": tuple(env.goal_xy),
        "goal_radius": float(env.goal_radius),
        "positions": positions,
        "track": np.asarray(env.unwrapped.track, dtype=np.float32),
    }


def plot_episode(ax, ep, arrow_len=25.0):
    track = ep["track"]
    tx, ty = track[:, 2], track[:, 3]
    txc, tyc = np.append(tx, tx[0]), np.append(ty, ty[0])
    ax.plot(txc, tyc, color="0.75", lw=6, solid_capstyle="round", zorder=1)
    ax.plot(txc, tyc, "k--", lw=0.8, zorder=2)

    traj = np.asarray(ep["positions"], dtype=np.float32)
    ax.plot(traj[:, 0], traj[:, 1], color="crimson", lw=2, zorder=3)

    sx, sy = traj[0]
    ax.plot(sx, sy, "o", color="lime", ms=10, mec="black", mew=1, zorder=5, label="spawn")
    dx, dy = arrow_len * math.cos(ep["heading"]), arrow_len * math.sin(ep["heading"])
    ax.annotate("", xy=(sx + dx, sy + dy), xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color="lime", lw=2), zorder=6)

    gx, gy = ep["goal_xy"]
    ax.plot(gx, gy, "*", color="gold", ms=16, mec="black", mew=0.8, zorder=5, label="goal")
    ax.add_patch(plt.Circle((gx, gy), ep["goal_radius"], fill=False, color="gold",
                             lw=1, ls="--", zorder=4))

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"seed={ep['seed']}  spawn_idx={ep['spawn_idx']}", fontsize=9)


def draw_debug_overlay(frame_rgb, tick, action, ep_return, x, y, speed):
    img = frame_rgb.copy()
    box_w, line_h, n_lines = 230, 18, 5
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (5 + box_w, 5 + n_lines * line_h + 10), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
    x0, y0 = 10, 20
    lines = [
        f"tick={tick}",
        f"action={action} ({DISCRETE_ACTION_LABELS.get(action, '?')})",
        f"pos=({x:.1f},{y:.1f})",
        f"speed={speed:.3f}",
        f"return={ep_return:.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x0, y0 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save_episode_video(env, seed, fps, max_ticks, video_path, video_fps=30):
    obs, info = env.reset(seed=seed)
    action = FPS_CHOICES.index(fps)
    spawn_idx = int(env.unwrapped.spawn_idx)
    print(f"[video] seed={seed} spawn_idx={spawn_idx} recording up to {max_ticks} ticks...")

    frames = []
    ep_return, done, tick = 0.0, False, 0
    while not done and tick < max_ticks:
        # navigation_action is what the frozen nav model actually chose this tick --
        # the thing to look at if the car isn't moving despite a "gas"-capable action
        # space, since a discrete steer-only/noop/brake choice with the car at rest
        # produces zero motion regardless of anything about spawn position/orientation.
        nav_action, _ = env.navigation_model.predict(env.last_sampled_obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        ep_return += r
        x, y = env.unwrapped.car.hull.position
        vx, vy = env.unwrapped.car.hull.linearVelocity
        frame = env.unwrapped.render()
        frames.append(draw_debug_overlay(frame, tick, int(nav_action), ep_return, x, y, float(np.hypot(vx, vy))))
        done = term or trunc
        tick += 1

    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    imageio.mimsave(video_path, frames, fps=video_fps, codec="libx264",
                     pixelformat="yuv420p", macro_block_size=1)
    print(f"[video] wrote {video_path}  ({len(frames)} frames, final_return={ep_return:.1f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nav-model-path", required=True)
    p.add_argument("--n-episodes", type=int, default=6)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--fixed-fps", type=int, default=25, choices=FPS_CHOICES)
    p.add_argument("--frame-cost", type=float, default=0.02)
    p.add_argument("--budget", type=float, default=100)
    p.add_argument("--trajectory-ticks", type=int, default=150,
                    help="how many physics ticks after spawn to draw per episode -- "
                         "kept short so the plot shows spawn orientation, not full laps")
    p.add_argument("--out-dir", default="experiments/var_fps/visualize_random_start")
    p.add_argument("--save-video-seed", type=int, default=None,
                    help="if set, skip the panel plot and instead render a debug-overlaid "
                         "video for this one seed (nav model's chosen action, position, "
                         "speed each tick) -- e.g. to see why a panel showed no movement")
    p.add_argument("--video-max-ticks", type=int, default=300)
    p.add_argument("--video-fps", type=int, default=30)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = make_env(args.nav_model_path, args.frame_cost, args.budget)

    if args.save_video_seed is not None:
        video_path = os.path.join(args.out_dir, "videos",
                                   f"random_start_seed{args.save_video_seed}_fps{args.fixed_fps}.mp4")
        save_episode_video(env, args.save_video_seed, args.fixed_fps, args.video_max_ticks, video_path,
                            video_fps=args.video_fps)
        env.close()
        return

    episodes = []
    for i in range(args.n_episodes):
        seed = args.seed_start + i
        ep = run_episode(env, seed, args.fixed_fps, args.trajectory_ticks)
        print(f"seed={seed:3d}  spawn_idx={ep['spawn_idx']:4d}  "
              f"spawn={ep['positions'][0]}  goal={ep['goal_xy']}")
        episodes.append(ep)
    env.close()

    ncols = min(3, args.n_episodes)
    nrows = math.ceil(args.n_episodes / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, ep in zip(axes, episodes):
        plot_episode(ax, ep)
    for ax in axes[len(episodes):]:
        ax.axis("off")

    fig.suptitle(f"Random spawn check -- {args.n_episodes} episodes, fixed FPS={args.fixed_fps}\n"
                 f"green arrow = spawn heading, gold star = goal (should always point along the track)")
    fig.tight_layout()
    plot_path = os.path.join(args.out_dir, f"random_start_fps{args.fixed_fps}_n{args.n_episodes}.png")
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {plot_path}")


if __name__ == "__main__":
    main()
