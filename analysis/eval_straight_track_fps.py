"""
eval_straight_track_fps.py

Evaluates each fixed FPS choice (1, 5, 10, 25, 50) on the synthetic
CarRacing_StraightTrack stadium track (envs/car_racing_straight_track.py), with
the wrapper's goal_distance set strictly within the length of one straight
segment -- so the whole evaluated episode never has to navigate a curve. This
directly tests whether a high sensing rate (FPS) is actually needed on straight
sections, decoupled entirely from the confound of curve navigation.

frame_cost=0.0 and a generous budget are used throughout -- this experiment is
about raw driving quality (nav_reward/off-track/cross-track) per FPS, not about
frame_cost economics (that's what frame_cost_calibration.py already covers).

Usage:
  python analysis/eval_straight_track_fps.py --n-seeds 8 --out-dir eval_straight_track
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import envs.car_racing_straight_track as straight_track_mod  # noqa: F401 -- registers the env
from envs.car_racing_var_fps import FPS as RAW_FPS  # 50 -- the base env's own physics rate
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.adaptive_fps_track_aware_wrapper import AdaptiveFPS_TrackAware_Wrapper
from utils.cautious_variables import OFF_TRACK_WHEEL_THRESHOLD
from eval_adaptive_fps_track_aware import FPS_CHOICES, NAV_MODEL_PATH

STRAIGHT_LENGTH = straight_track_mod.STRAIGHT_LENGTH
# Strictly less than STRAIGHT_LENGTH so the goal point lands on the SAME straight
# segment as the start line -- the episode then ends via the wrapper's own
# reached_goal mechanism without ever needing to navigate a turn.
GOAL_DISTANCE = 400.0
CROSS_TRACK_IDX = 5  # augmented-obs layout: [...11 cautious dims (index 5 = cross_track)..., obs_age_ratio, fps_ratio, frame_counter]


def make_straight_env(nav_model_path, budget, max_episode_steps):
    env = gym.make("CarRacing_StraightTrack", continuous=False, render_mode="rgb_array")
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    env = AdaptiveFPS_TrackAware_Wrapper(env, nav_model_path, device="cpu",
                                          frame_cost=0.0, budget=budget,
                                          goal_distance=GOAL_DISTANCE,
                                          # This experiment is about raw driving quality --
                                          # the lump-sum goal bonus would otherwise dominate
                                          # the reward comparison rather than reflect it.
                                          goal_reward=0.0)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def run_episode(fps, seed, nav_model_path, budget, max_episode_steps):
    env = make_straight_env(nav_model_path, budget, max_episode_steps)
    obs, _ = env.reset(seed=seed)
    action = FPS_CHOICES.index(fps)
    # env.env is the AdaptiveFPS_TrackAware_Wrapper instance (env is TimeLimit(...))
    # -- goal_xy/goal_radius are set in its reset(), same fields eval_adaptive_fps_track_aware.py's
    # plot_track_fps() reads to draw the goal marker.
    goal_xy = tuple(env.env.goal_xy)
    goal_radius = float(env.env.goal_radius)

    positions, rewards, cum_rewards, in_curve_flags, cross_track_vals = [], [], [], [], []
    timesteps = []  # raw CarRacing_VarFramerate physics-tick count, see module docstring
    off_track_ticks = 0
    cum = 0.0
    nav_reward_cum = 0.0
    done = False
    tick = 0
    info = {}
    while not done:
        obs, r, term, trunc, info = env.step(action)
        tick += 1
        cum += r
        nav_reward_cum += info["nav_reward"]
        x, y = env.unwrapped.car.hull.position
        positions.append((x, y))
        rewards.append(r)
        cum_rewards.append(cum)
        in_curve_flags.append(bool(env.unwrapped.in_curve))
        off_wheels = sum(len(w.tiles) == 0 for w in env.unwrapped.car.wheels)
        if off_wheels >= OFF_TRACK_WHEEL_THRESHOLD:
            off_track_ticks += 1
        cross_track_vals.append(float(obs[CROSS_TRACK_IDX]))
        # env.unwrapped is the CarRacing_VarFramerate instance -- self.t accumulates real
        # sim time (self.t += 1.0/FPS every raw physics tick, car_racing_var_fps.py:617),
        # so this recovers the exact raw-tick count regardless of skip_frames grouping
        # (including a possibly-shorter final group if the episode ends mid-group).
        timesteps.append(round(env.unwrapped.t * RAW_FPS))
        done = term or trunc

    env.close()
    return {
        "fps": fps,
        "seed": seed,
        "positions": positions,
        "cum_rewards": cum_rewards,
        "in_curve_flags": in_curve_flags,
        "timesteps": timesteps,
        "reached_goal": bool(info.get("reached_goal", False)),
        "total_reward": cum,
        "total_nav_reward": nav_reward_cum,
        "off_track_rate": off_track_ticks / tick if tick > 0 else 0.0,
        "mean_cross_track_abs": float(np.mean(np.abs(cross_track_vals))) if cross_track_vals else 0.0,
        "tick_count": tick,
        "total_timesteps": timesteps[-1] if timesteps else 0,
        "goal_xy": goal_xy,
        "goal_radius": goal_radius,
    }


def _worker(job):
    fps, seed, nav_model_path, budget, max_episode_steps = job
    return run_episode(fps, seed, nav_model_path, budget, max_episode_steps)


def plot_trajectory(track, ep, save_path, title):
    tx, ty = track[:, 2], track[:, 3]
    txc, tyc = np.append(tx, tx[0]), np.append(ty, ty[0])
    traj = np.asarray(ep["positions"], dtype=np.float32)
    goal_xy, goal_radius = ep["goal_xy"], ep["goal_radius"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(txc, tyc, color="0.75", lw=8, solid_capstyle="round", zorder=1)
    ax.plot(txc, tyc, "k--", lw=1, zorder=2)
    ax.plot(traj[:, 0], traj[:, 1], color="crimson", lw=2, zorder=3)
    ax.plot(traj[0, 0], traj[0, 1], "o", color="lime", ms=10, zorder=4, label="start")
    ax.plot(traj[-1, 0], traj[-1, 1], "X", color="red", ms=10, zorder=4, label="end")
    # Goal marker -- same style as eval_adaptive_fps_track_aware.py's plot_track_fps()
    ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=20,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6, label="goal")
    ax.add_patch(plt.Circle(goal_xy, goal_radius, fill=False, color="gold", lw=1.2, ls="--", zorder=6))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)
    ax.legend(loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_reward_curve(ep, save_path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ep["timesteps"], ep["cum_rewards"], color="steelblue")
    ax.set_xlabel("raw physics timestep (CarRacing_VarFramerate, 1/50s each)")
    ax.set_ylabel("cumulative reward")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(episodes_by_fps, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(FPS_CHOICES)))
    for color, fps in zip(cmap, FPS_CHOICES):
        ep = episodes_by_fps[fps]
        ax.plot(ep["timesteps"], ep["cum_rewards"], color=color, lw=2, label=f"{fps} FPS")
    ax.set_xlabel("raw physics timestep (CarRacing_VarFramerate, 1/50s each)")
    ax.set_ylabel("cumulative reward")
    ax.set_title("Cumulative reward vs. raw timestep -- fixed FPS on a pure straight track")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nav-model-path", default=NAV_MODEL_PATH)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=300000,
                    help="distinct from other analyses' seed ranges used this session")
    p.add_argument("--budget", type=float, default=500,
                    help="generous on purpose -- this experiment is about driving quality, "
                         "not frame_cost/budget economics, so budget-overrun shouldn't fire")
    p.add_argument("--max-episode-steps", type=int, default=500)
    p.add_argument("--workers", type=int, default=40)
    p.add_argument("--out-dir", default="eval_straight_track")
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    jobs = [(fps, seed, args.nav_model_path, args.budget, args.max_episode_steps)
            for fps in FPS_CHOICES for seed in seeds]

    print(f"[eval_straight_track] running {len(jobs)} episodes across {args.workers} workers...")
    grouped = {fps: [] for fps in FPS_CHOICES}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            ep = fut.result()
            grouped[ep["fps"]].append(ep)
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[eval_straight_track] {done}/{len(jobs)} episodes done")

    # sort each fps's episodes by seed for reproducible "representative episode" (seed[0]) picks
    for fps in FPS_CHOICES:
        grouped[fps].sort(key=lambda e: e["seed"])

    # --- Mandatory verification: every logged tick must be geometrically off any curve ---
    print("\n=== Mandatory verification: episodes must never touch a curve ===")
    all_ok = True
    for fps in FPS_CHOICES:
        for ep in grouped[fps]:
            if any(ep["in_curve_flags"]):
                all_ok = False
                print(f"  [FAIL] fps={fps} seed={ep['seed']}: in_curve was True at some tick "
                      f"-- STRAIGHT_LENGTH/GOAL_DISTANCE need adjusting, results below are NOT trustworthy")
            xs = np.array([pos[0] for pos in ep["positions"]])
            if xs.min() < -STRAIGHT_LENGTH / 2 - 5 or xs.max() > STRAIGHT_LENGTH / 2 + 5:
                all_ok = False
                print(f"  [FAIL] fps={fps} seed={ep['seed']}: x-position left the straight's "
                      f"expected range (x in [{xs.min():.1f}, {xs.max():.1f}])")
    print("  all episodes verified on-straight" if all_ok else "  SOME EPISODES FAILED VERIFICATION -- see above")

    # --- Aggregate table ---
    # reward vs. nav_reward: identical every tick except the tick reached_goal fires,
    # where the wrapper overrides reward=150 outright (discarding nav_reward-frame_penalty
    # for that tick) -- see AdaptiveFPS_TrackAware_Wrapper.step(). So at frame_cost=0.0 with
    # a budget large enough to never overrun, total_reward and total_nav_reward should match
    # exactly for episodes that never reach the goal, and differ by exactly
    # (150 - that tick's true nav_reward) for episodes that do.
    print("\n=== Per-FPS aggregate results (pure straight track, frame_cost=0.0, goal_reward=0.0) ===")
    print(f"{'fps':>4} {'n':>3} {'mean_reward':>12} {'mean_nav_reward':>16} {'reward-nav_reward':>18} "
          f"{'reached_goal':>13} {'off_track_rate':>15} {'mean|cross_track|':>18} {'mean_ticks':>11} {'mean_timesteps':>15}")
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "straight_track_fps_comparison_2.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fps", "n_episodes", "mean_reward", "mean_nav_reward", "reward_minus_nav_reward",
                          "reached_goal_rate", "off_track_rate", "mean_cross_track_abs", "mean_ticks",
                          "mean_timesteps"])
        for fps in FPS_CHOICES:
            eps = grouped[fps]
            mean_reward = np.mean([e["total_reward"] for e in eps])
            mean_nav_reward = np.mean([e["total_nav_reward"] for e in eps])
            reached_rate = np.mean([e["reached_goal"] for e in eps])
            off_track_rate = np.mean([e["off_track_rate"] for e in eps])
            mean_cross = np.mean([e["mean_cross_track_abs"] for e in eps])
            mean_ticks = np.mean([e["tick_count"] for e in eps])
            mean_timesteps = np.mean([e["total_timesteps"] for e in eps])
            print(f"{fps:>4} {len(eps):>3} {mean_reward:>12.2f} {mean_nav_reward:>16.2f} "
                  f"{mean_reward - mean_nav_reward:>18.2f} {reached_rate:>13.0%} "
                  f"{off_track_rate:>15.2%} {mean_cross:>18.4f} {mean_ticks:>11.1f} {mean_timesteps:>15.1f}")
            writer.writerow([fps, len(eps), round(float(mean_reward), 3), round(float(mean_nav_reward), 3),
                              round(float(mean_reward - mean_nav_reward), 3), round(float(reached_rate), 4),
                              round(float(off_track_rate), 4), round(float(mean_cross), 4),
                              round(float(mean_ticks), 1), round(float(mean_timesteps), 1)])
    print(f"\nwrote {csv_path}")

    # --- Plots ---
    sample_env = gym.make("CarRacing_StraightTrack", continuous=False, render_mode="rgb_array")
    sample_env.reset(seed=seeds[0])
    track = np.asarray(sample_env.unwrapped.track, dtype=np.float32)
    sample_env.close()

    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    representative = {}
    for fps in FPS_CHOICES:
        ep0 = grouped[fps][0]  # seed[0], deterministic representative episode
        representative[fps] = ep0
        plot_trajectory(track, ep0, os.path.join(plot_dir, f"plot_straight_fixedfps{fps}_seed{ep0['seed']}_2.png"),
                         title=f"fixed FPS={fps}  seed={ep0['seed']}  reward={ep0['total_reward']:.0f}  "
                               f"timesteps={ep0['total_timesteps']}")
        plot_reward_curve(ep0, os.path.join(plot_dir, f"reward_curve_fixedfps{fps}_seed{ep0['seed']}_2.png"),
                           title=f"fixed FPS={fps}  cumulative reward vs. raw timestep")
        print(f"saved plots for fps={fps} -> {plot_dir}")

    plot_comparison(representative, os.path.join(plot_dir, "reward_comparison_all_fps.png"))
    print(f"saved comparison plot -> {os.path.join(plot_dir, 'reward_comparison_all_fps.png')}")


if __name__ == "__main__":
    main()
