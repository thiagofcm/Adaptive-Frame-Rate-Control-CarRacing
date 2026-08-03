"""
frame_cost_calibration.py

Diagnostic study (no training): quantifies the relative magnitude of the frame_cost
reward penalty against nav_reward, across fixed-FPS baselines and a hand-crafted
curve-aware adaptive heuristic, over a dense frame_cost grid. Goal: find frame_cost
values where genuine state-conditional FPS switching earns more net reward than any
single fixed FPS choice -- as opposed to values where one FPS extreme dominates
regardless of driving conditions (the corner-solution collapse observed in trained
adaptive policies: FPS=50 always, or FPS=1 always at frame_cost=10.0).

Key efficiency trick: under a FIXED FPS or a CautiousVars-only heuristic rule, the
env physics/nav trajectory never depends on frame_cost (the frozen NavModel's
actions don't see it, and neither policy's FPS choice is a function of reward). So
each (condition, seed) episode only needs to be simulated ONCE, at frame_cost=0.0 --
net_reward at every other frame_cost value is then reconstructed analytically:

    net_reward(fc) = nav_reward - fc*episode_frame_count
                      + 150*reached_goal_rate - 100*budget_overrun_rate

(Small known approximation: on the single tick where a reached_goal/budget_overrun
override fires, AdaptiveFPS_TrackAware_Wrapper.step() replaces that tick's reward
outright rather than also subtracting frame_cost -- so this can be off by at most
one tick's frame_cost per episode, negligible next to episode totals.)

Usage:
  python frame_cost_calibration.py --mode both --seeds 20 --workers 16
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
# Each of the --workers subprocesses runs a single episode's env.step()/NavModel
# forward loop -- no benefit from internal BLAS/OpenMP multithreading, only
# oversubscription against the other workers (same reasoning as the training
# script's identical guard). Must be set before numpy/torch import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
torch.set_num_threads(1)
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import envs.car_racing_var_fps  # noqa: F401 -- registers "CarRacing_VarFramerate"
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.adaptive_fps_track_aware_wrapper import AdaptiveFPS_TrackAware_Wrapper

FPS_CHOICES = [1, 5, 10, 25, 50]
NAV_MODEL_PATH = "old/experiments/navigation/runs/CarRacing-v3__ppo__1__1781901069/final.pt"
# Augmented-obs layout (see AdaptiveFPS_TrackAware_Wrapper): [vx, vy, dist_to_curve,
# curve_severity, heading_alignment, cross_track, cross_track_rate, off_track,
# time_off_track, episode_completion, curves_passed, obs_age_ratio, fps_ratio,
# frame_counter] -- both already normalized to roughly [0, 1].
DIST_TO_CURVE_IDX = 2
CURVE_SEVERITY_IDX = 3


def make_env(nav_model_path, budget, max_episode_steps, frame_cost=0.0):
    env = gym.make("CarRacing_VarFramerate", continuous=False, render_mode="rgb_array")
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    env = AdaptiveFPS_TrackAware_Wrapper(env, nav_model_path, device="cpu",
                                          frame_cost=frame_cost, budget=budget)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def fixed_fps_action_fn(fps):
    action = FPS_CHOICES.index(fps)
    return lambda obs: action


def make_heuristic_action_fn(dist_thresh, severity_thresh):
    # 50<->25 swing only (not down to 1): tests whether it's worth paying extra to
    # go from 25->50 specifically during curves, a much more plausible adaptive
    # tradeoff than the 50-vs-1 extreme swing tested in the first pass, which just
    # inherited FPS=1's poor driving quality whenever it wasn't near a curve.
    fast_idx = FPS_CHOICES.index(50)
    slow_idx = FPS_CHOICES.index(25)

    def action_fn(obs):
        close_to_curve = (obs[DIST_TO_CURVE_IDX] < dist_thresh) or (obs[CURVE_SEVERITY_IDX] > severity_thresh)
        return fast_idx if close_to_curve else slow_idx

    return action_fn


def run_episode(env, seed, action_fn):
    obs, _ = env.reset(seed=seed)
    nav_reward_sum = 0.0
    length = 0
    done = False
    info = {}
    fps_decision_counts = {fps: 0 for fps in FPS_CHOICES}
    while not done:
        action = action_fn(obs)
        obs, r, term, trunc, info = env.step(action)
        nav_reward_sum += info["nav_reward"]
        length += 1
        done = term or trunc
        # Only count *real* decisions (frame_consumed=True) -- matches the training
        # loop's own causal-tick masking, so this reflects how often each FPS was
        # actually chosen, not how many physics ticks happened to pass under it.
        if info["frame_consumed"]:
            fps_decision_counts[int(info["chosen_fps"])] += 1
    return {
        "nav_reward": nav_reward_sum,
        "length": length,
        "frame_count": info["episode_frame_count"],
        "reached_goal": bool(info["reached_goal"]),
        "budget_overrun": (not info["reached_goal"]) and info["episode_frame_count"] > info["budget"],
        "fps_decision_counts": fps_decision_counts,
    }


def _worker(job):
    kind, param, seed, nav_model_path, budget, max_episode_steps, dist_thresh, severity_thresh = job
    env = make_env(nav_model_path, budget, max_episode_steps)
    if kind == "fixed":
        action_fn = fixed_fps_action_fn(param)
    else:
        action_fn = make_heuristic_action_fn(dist_thresh, severity_thresh)
    stats = run_episode(env, seed, action_fn)
    env.close()
    return kind, param, seed, stats


def aggregate(episodes):
    nav_reward = np.array([e["nav_reward"] for e in episodes], dtype=np.float64)
    frame_count = np.array([e["frame_count"] for e in episodes], dtype=np.float64)
    length = np.array([e["length"] for e in episodes], dtype=np.float64)

    total_decisions = {fps: sum(e["fps_decision_counts"][fps] for e in episodes) for fps in FPS_CHOICES}
    grand_total = sum(total_decisions.values())
    fps_mix = {fps: (total_decisions[fps] / grand_total if grand_total > 0 else 0.0) for fps in FPS_CHOICES}

    return {
        "nav_reward_mean": nav_reward.mean(),
        "nav_reward_std": nav_reward.std(),
        "length_mean": length.mean(),
        "frame_count_mean": frame_count.mean(),
        "reached_goal_rate": np.mean([e["reached_goal"] for e in episodes]),
        "budget_overrun_rate": np.mean([e["budget_overrun"] for e in episodes]),
        "n_episodes": len(episodes),
        "fps_mix": fps_mix,
    }


def net_reward_curve(stats, frame_cost_grid):
    return (stats["nav_reward_mean"]
            - frame_cost_grid * stats["frame_count_mean"]
            + 150.0 * stats["reached_goal_rate"]
            - 100.0 * stats["budget_overrun_rate"])


def find_crossovers(labels, curves, frame_cost_grid):
    stacked = np.stack([curves[l] for l in labels], axis=0)
    argmax_idx = stacked.argmax(axis=0)
    crossovers = []
    for i in range(1, len(frame_cost_grid)):
        if argmax_idx[i] != argmax_idx[i - 1]:
            crossovers.append((frame_cost_grid[i], labels[argmax_idx[i - 1]], labels[argmax_idx[i]]))
    return crossovers


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["fixed", "adaptive", "both"], default="both")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--budget", type=float, default=180)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--nav-model-path", default=NAV_MODEL_PATH)
    p.add_argument("--out-dir", default="frame_cost_calibration")
    p.add_argument("--fc-max", type=float, default=10.0)
    p.add_argument("--fc-step", type=float, default=0.1)
    p.add_argument("--dist-thresh", type=float, default=0.1,
                    help="heuristic: sample at FPS=50 when normalized dist_to_curve is below this")
    p.add_argument("--severity-thresh", type=float, default=0.3,
                    help="heuristic: sample at FPS=50 when normalized curve_severity is above this")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    frame_cost_grid = np.arange(0.0, args.fc_max + 1e-9, args.fc_step)
    seeds = list(range(args.seeds))

    jobs = []
    if args.mode in ("fixed", "both"):
        for fps in FPS_CHOICES:
            for seed in seeds:
                jobs.append(("fixed", fps, seed, args.nav_model_path, args.budget,
                              args.max_episode_steps, args.dist_thresh, args.severity_thresh))
    if args.mode in ("adaptive", "both"):
        for seed in seeds:
            jobs.append(("heuristic", None, seed, args.nav_model_path, args.budget,
                          args.max_episode_steps, args.dist_thresh, args.severity_thresh))

    print(f"[calibration] running {len(jobs)} episodes across {args.workers} workers...")
    grouped = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        done_count = 0
        for fut in as_completed(futures):
            kind, param, seed, stats = fut.result()
            label = f"fps={param}" if kind == "fixed" else "heuristic-adaptive"
            grouped.setdefault(label, []).append(stats)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(jobs):
                print(f"[calibration] {done_count}/{len(jobs)} episodes done")

    labels = [f"fps={fps}" for fps in FPS_CHOICES if f"fps={fps}" in grouped]
    if "heuristic-adaptive" in grouped:
        labels.append("heuristic-adaptive")

    results = {label: aggregate(grouped[label]) for label in labels}

    print("\n=== Per-condition baseline stats (frame_cost=0.0) ===")
    for label in labels:
        s = results[label]
        mix_str = " ".join(f"fps{fps}={frac:.0%}" for fps, frac in s["fps_mix"].items() if frac > 0)
        print(f"[{label:20s}] nav_reward={s['nav_reward_mean']:8.1f}+/-{s['nav_reward_std']:6.1f}  "
              f"length={s['length_mean']:6.0f}  frame_count={s['frame_count_mean']:6.1f}  "
              f"reached_goal={s['reached_goal_rate']:5.0%}  budget_overrun={s['budget_overrun_rate']:5.0%}  "
              f"n={s['n_episodes']}  mix: {mix_str}")

    if "heuristic-adaptive" in results:
        h_mix = results["heuristic-adaptive"]["fps_mix"]
        dominant_frac = max(h_mix.values())
        if dominant_frac > 0.9:
            dominant_fps = max(h_mix, key=h_mix.get)
            print(f"\n  WARNING: heuristic-adaptive spent {dominant_frac:.0%} of decisions at fps={dominant_fps} "
                  f"-- it's not meaningfully mixing 25/50, so any comparison against fixed baselines below "
                  f"isn't a fair test of adaptiveness. Consider loosening --dist-thresh/--severity-thresh.")

    curves = {label: net_reward_curve(results[label], frame_cost_grid) for label in labels}

    fixed_labels = [l for l in labels if l.startswith("fps=")]
    recommended = []

    if len(fixed_labels) > 1:
        crossovers = find_crossovers(fixed_labels, curves, frame_cost_grid)
        print("\n=== Fixed-FPS crossovers (where the argmax constant choice changes) ===")
        if not crossovers:
            print(f"  none found in [0, {args.fc_max}] -- one FPS dominates the entire range")
        for fc, frm, to in crossovers:
            print(f"  at frame_cost~={fc:.2f}: {frm} -> {to}")
            for delta in (-0.3, -0.1, 0.0, 0.1, 0.3):
                cand = round(max(0.0, fc + delta), 2)
                if cand <= args.fc_max:
                    recommended.append(cand)

    if "heuristic-adaptive" in curves and fixed_labels:
        best_fixed = np.max([curves[l] for l in fixed_labels], axis=0)
        beats_fixed = curves["heuristic-adaptive"] > best_fixed
        if beats_fixed.any():
            beat_fc = frame_cost_grid[beats_fixed]
            print(f"\n=== heuristic-adaptive beats every fixed FPS choice for frame_cost in "
                  f"[{beat_fc.min():.2f}, {beat_fc.max():.2f}] ===")
            margin = curves["heuristic-adaptive"][beats_fixed] - best_fixed[beats_fixed]
            best_i = np.argmax(margin)
            print(f"  largest margin at frame_cost={beat_fc[best_i]:.2f} "
                  f"(+{margin[best_i]:.1f} net reward over the best fixed choice)")
            recommended.extend(round(float(v), 2) for v in beat_fc[::max(1, len(beat_fc) // 5)])
        else:
            print(f"\n=== heuristic-adaptive never beats the best fixed FPS choice anywhere in "
                  f"[0, {args.fc_max}] -- frame_cost scale alone likely isn't the whole problem ===")

    if recommended:
        recommended = sorted(set(recommended))
        print(f"\n=== Recommended frame_cost sweep candidates ===\n{recommended}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for label in fixed_labels:
        ax.plot(frame_cost_grid, curves[label], label=label)
    if "heuristic-adaptive" in curves:
        ax.plot(frame_cost_grid, curves["heuristic-adaptive"], "k--", lw=2.5, label="heuristic-adaptive")
    ax.set_xlabel("frame_cost")
    ax.set_ylabel("reconstructed net episode reward")
    ax.set_title("Net reward vs frame_cost -- fixed FPS baselines vs curve-aware heuristic")
    ax.legend()
    ax.grid(alpha=0.3)
    plot_path = os.path.join(args.out_dir, "net_reward_vs_frame_cost.png")
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved plot -> {plot_path}")


if __name__ == "__main__":
    main()
