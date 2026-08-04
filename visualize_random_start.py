"""
visualize_random_start.py

Visual smoke-test for AdaptiveFPS_Random_Initial_Pos + CarRacing_VarFramerate_RandomStart.
Runs several episodes (different seeds) and plots, one panel per episode (each episode
regenerates its own procedural track, so they can't be usefully overlaid on one outline):
the track, the spawn point with a heading arrow, the goal point, and a short trajectory
right after spawn. If the car is correctly oriented, the trajectory moves along the
track direction immediately -- not sideways or backward -- which is the main thing to
visually check here.

Usage:
  python experiments/var_fps/visualize_random_start.py \\
      --nav-model-path experiments/navigation/CarRacing-v3__train__1__1785768450/final.pt
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym

import envs.car_racing_var_fps_random_start  # noqa: F401 -- registers "CarRacing_VarFramerate_RandomStart"
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.adaptive_fps_random_initial_pos import AdaptiveFPS_Random_Initial_Pos

FPS_CHOICES = [1, 5, 10, 25, 50]


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nav-model-path", required=True)
    p.add_argument("--n-episodes", type=int, default=6)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--fixed-fps", type=int, default=50, choices=FPS_CHOICES)
    p.add_argument("--frame-cost", type=float, default=0.02)
    p.add_argument("--budget", type=float, default=100)
    p.add_argument("--trajectory-ticks", type=int, default=1000,
                    help="how many physics ticks after spawn to draw per episode -- "
                         "kept short so the plot shows spawn orientation, not full laps")
    p.add_argument("--out-dir", default="experiments/var_fps/visualize_random_start")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = make_env(args.nav_model_path, args.frame_cost, args.budget)

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
