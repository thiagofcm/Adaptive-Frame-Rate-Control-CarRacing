"""
collect_fixed_fps_segments.py

Step 2b of the FPS-policy investigation plan: the decoupled experiment. Runs the
frozen nav-controller OPEN-LOOP at fixed FPS (25 vs 50, the same pair the trained
policy actually chooses between -- see analyze_fps_vs_curvature.py's fps_mix
output), with no FPS-selection policy in the loop at all, and logs every physics
tick segmented by ground-truth curve/straight region. This isolates the frozen
controller's own staleness sensitivity from anything the FPS-selection policy
learned or didn't learn.

Per-tick columns logged: seed, fps (25 or 50 condition), tick, in_curve (binary,
reusing the base env's own `_is_curve_tile`/`in_curve` state -- the SAME ground
truth used for the CURVE_PASSED_BONUS reward, not re-derived), curv_abs
(continuous, from a fresh per-tick CautiousVars reading), off_track (binary,
>=OFF_TRACK_WHEEL_THRESHOLD wheels off), cross_track_abs (normalized), nav_reward
(this tick's raw task reward, NOT the FPS-cost-adjusted `reward` -- we want the
controller's task performance, decoupled from frame_cost/budget accounting).

At episode end, also records episode-level fields (reached_goal, curves_passed_rate,
budget_overrun) for supporting context.

Usage:
  python analysis/collect_fixed_fps_segments.py --n-episodes 150 --seed-start 200000 \
      --workers 100 --out-csv analysis_out/fixed_fps_segments.csv
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_adaptive_fps_track_aware import make_eval_env, FPS_CHOICES, NAV_MODEL_PATH
from utils.cautious_variables import CautiousVars, OFF_TRACK_WHEEL_THRESHOLD

CROSS_TRACK_IDX = 5  # CautiousVars.get_cautious_var() layout: see cautious_variables.py


def run_episode(seed, fps, nav_model_path, env_id, budget, max_episode_steps):
    # frame_cost=0.0: irrelevant here, we log raw nav_reward per tick, not the
    # FPS-cost-adjusted reward -- this experiment is about controller performance
    # under staleness, decoupled from frame_cost/budget economics entirely.
    env = make_eval_env(env_id, nav_model_path, frame_cost=0.0, budget=budget,
                         max_episode_steps=max_episode_steps)
    obs, _ = env.reset(seed=seed)

    cv = CautiousVars()
    cv.reset_track_reading(env)

    action = FPS_CHOICES.index(fps)
    rows = []
    tick = 0
    done = False
    info = {}
    while not done:
        obs, r, term, trunc, info = env.step(action)
        tick += 1
        done = term or trunc

        # Called every physics tick (dt_ticks=1 throughout) -- a fresh, continuously
        # updated ground-truth reading, decoupled from whatever staleness the FIXED
        # FPS condition itself imposes on the controller's own input.
        cvars = cv.get_cautious_var(env, dt_ticks=1)
        off_track_wheels = sum(len(w.tiles) == 0 for w in env.unwrapped.car.wheels)

        rows.append({
            "seed": seed,
            "fps": fps,
            "tick": tick,
            "in_curve": bool(env.unwrapped.in_curve),
            "curv_abs": abs(float(cv.curv[cv.last_idx])),
            "off_track": int(off_track_wheels >= OFF_TRACK_WHEEL_THRESHOLD),
            "cross_track_abs": abs(float(cvars[CROSS_TRACK_IDX])),
            "nav_reward": float(info["nav_reward"]),
            "track_hash": info["track_hash"],
        })

    episode_summary = {
        "seed": seed,
        "fps": fps,
        "reached_goal": bool(info.get("reached_goal", False)),
        "curves_passed_rate": float(env.unwrapped.curves_passed_count / env.unwrapped.total_curves_on_track),
        "budget_overrun": bool((not info.get("reached_goal", False))
                                and info.get("episode_frame_count", 0) > info.get("budget", budget)),
        "length": tick,
        "track_hash": info["track_hash"],
    }
    env.close()
    return rows, episode_summary


def _worker(job):
    seed, fps, nav_model_path, env_id, budget, max_episode_steps = job
    return run_episode(seed, fps, nav_model_path, env_id, budget, max_episode_steps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-id", default="CarRacing_VarFramerate")
    p.add_argument("--nav-model-path", default=NAV_MODEL_PATH)
    p.add_argument("--budget", type=float, default=180)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--n-episodes", type=int, default=150)
    p.add_argument("--seed-start", type=int, default=200000,
                    help="distinct from other analyses' seed ranges (100000+ for "
                         "collect_adaptive_rollouts.py, 0-19 for frame_cost_calibration.py, "
                         "42+ for eval_adaptive_fps_track_aware.py)")
    p.add_argument("--fixed-fps", type=int, nargs="+", default=[25, 50],
                    help="the FPS pair the trained policy actually toggles between "
                         "(see analyze_fps_vs_curvature.py's fps_mix output)")
    p.add_argument("--workers", type=int, default=100)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--episode-csv", default=None,
                    help="defaults to <out-csv dir>/fixed_fps_segments_episodes.csv")
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    jobs = [(s, fps, args.nav_model_path, args.env_id, args.budget, args.max_episode_steps)
            for fps in args.fixed_fps for s in seeds]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    tick_fields = ["seed", "fps", "tick", "in_curve", "curv_abs", "off_track",
                   "cross_track_abs", "nav_reward", "track_hash"]
    ep_fields = ["seed", "fps", "reached_goal", "curves_passed_rate", "budget_overrun",
                 "length", "track_hash"]

    print(f"[collect_fixed_fps_segments] running {len(jobs)} episodes "
          f"({args.n_episodes} seeds x {len(args.fixed_fps)} fps conditions) "
          f"across {args.workers} workers...")
    all_rows, all_summaries = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            rows, summary = fut.result()
            all_rows.extend(rows)
            all_summaries.append(summary)
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print(f"[collect_fixed_fps_segments] {done}/{len(jobs)} episodes done")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=tick_fields)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[collect_fixed_fps_segments] wrote {len(all_rows)} tick rows -> {args.out_csv}")

    episode_csv = args.episode_csv or os.path.join(
        os.path.dirname(args.out_csv) or ".", "fixed_fps_segments_episodes.csv")
    with open(episode_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ep_fields)
        writer.writeheader()
        writer.writerows(all_summaries)
    print(f"[collect_fixed_fps_segments] wrote {len(all_summaries)} episode summaries -> {episode_csv}")


if __name__ == "__main__":
    main()
