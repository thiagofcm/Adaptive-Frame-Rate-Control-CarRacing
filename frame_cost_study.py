"""
frame_cost_study.py

Built on frame_cost_calibration.py's fixed-FPS / heuristic-adaptive machinery. Two
modes:

  --mode sweep   (default) run every fixed-FPS choice + the heuristic-adaptive rule
                 over --seeds episodes each at frame_cost=0.0, analytically
                 reconstruct the return at every value in the --fc-max/--fc-step
                 grid (see frame_cost_calibration.py's docstring for why this
                 reconstruction is exact and doesn't need extra simulation), and
                 write one CSV row per (condition, frame_cost) with the
                 reconstructed mean return and mean frames_consumed
                 (episode_frame_count).

  --mode single  run ONE specific --seed at a REAL --frame-cost value (the env is
                 actually simulated at that cost, no reconstruction) for one
                 --condition (a fixed FPS or "heuristic"), producing a video and an
                 FPS-colored trajectory plot in the same style as
                 eval_adaptive_fps_track_aware.py's outputs.

Usage:
  python frame_cost_study.py --mode sweep --seeds 100 --workers 16
  python frame_cost_study.py --mode single --condition heuristic --seed 7 --frame-cost 2.2 --save-video
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
torch.set_num_threads(1)
import imageio.v2 as imageio

from frame_cost_calibration import (
    FPS_CHOICES, NAV_MODEL_PATH, make_env, fixed_fps_action_fn,
    make_heuristic_action_fn, run_episode, aggregate, net_reward_curve,
)
from eval_adaptive_fps_track_aware import draw_cautious_overlay, plot_track_fps

CONDITIONS = [f"fps{fps}" for fps in FPS_CHOICES] + ["heuristic"]


def action_fn_for(condition, dist_thresh, severity_thresh):
    if condition == "heuristic":
        return make_heuristic_action_fn(dist_thresh, severity_thresh)
    fps = int(condition[len("fps"):])
    return fixed_fps_action_fn(fps)


def fps_mix_str(fps_mix):
    return ",".join(f"fps{fps}={frac:.0%}" for fps, frac in fps_mix.items() if frac > 0)


def _sweep_worker(job):
    condition, seed, nav_model_path, budget, max_episode_steps, dist_thresh, severity_thresh = job
    env = make_env(nav_model_path, budget, max_episode_steps, frame_cost=0.0)
    action_fn = action_fn_for(condition, dist_thresh, severity_thresh)
    stats = run_episode(env, seed, action_fn)
    env.close()
    return condition, seed, stats


def run_sweep(args):
    frame_cost_grid = np.arange(0.0, args.fc_max + 1e-9, args.fc_step)
    seeds = list(range(args.seeds))

    jobs = [
        (condition, seed, args.nav_model_path, args.budget, args.max_episode_steps,
         args.dist_thresh, args.severity_thresh)
        for condition in CONDITIONS for seed in seeds
    ]

    print(f"[frame_cost_study] running {len(jobs)} episodes across {args.workers} workers...")
    grouped = {c: [] for c in CONDITIONS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_sweep_worker, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            condition, seed, stats = fut.result()
            grouped[condition].append(stats)
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"[frame_cost_study] {done}/{len(jobs)} episodes done")

    results = {c: aggregate(grouped[c]) for c in CONDITIONS}

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "frame_cost_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "frame_cost", "mean_return", "mean_frames_consumed",
                          "nav_reward_mean", "reached_goal_rate", "budget_overrun_rate",
                          "n_episodes", "fps_mix"])
        for condition in CONDITIONS:
            s = results[condition]
            curve = net_reward_curve(s, frame_cost_grid)
            mix = fps_mix_str(s["fps_mix"])
            for fc, ret in zip(frame_cost_grid, curve):
                writer.writerow([condition, round(float(fc), 4), round(float(ret), 3),
                                  round(s["frame_count_mean"], 3), round(s["nav_reward_mean"], 3),
                                  round(s["reached_goal_rate"], 4), round(s["budget_overrun_rate"], 4),
                                  s["n_episodes"], mix])

    print(f"[frame_cost_study] wrote {csv_path}  ({len(CONDITIONS) * len(frame_cost_grid)} rows)")


def run_single(args):
    env = make_env(args.nav_model_path, args.budget, args.max_episode_steps, frame_cost=args.frame_cost)
    action_fn = action_fn_for(args.condition, args.dist_thresh, args.severity_thresh)

    obs, _ = env.reset(seed=args.seed)
    positions, fps_values, frames = [], [], []
    ep_return, ep_len, done = 0.0, 0, False
    prev_episode_frame_count = 0
    info = {}

    while not done:
        action = action_fn(obs)
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

        if args.save_video:
            frame = env.unwrapped.render()
            annotated = draw_cautious_overlay(frame, ep_len, chosen, fresh, obs[:11], ep_return)
            frames.append(annotated)

    print(f"{args.condition}  seed={args.seed}  frame_cost={args.frame_cost}: "
          f"return={ep_return:.1f}  length={ep_len}  reached_goal={info.get('reached_goal')}")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.condition}_seed{args.seed}_fc{args.frame_cost}"
    plot_track_fps(
        env, env.env, positions, fps_values,
        save_path=os.path.join(args.out_dir, "plots", f"plot_{tag}_ret{int(ep_return)}.png"),
        title=f"{args.condition}  fc={args.frame_cost}  return={ep_return:.0f}  len={ep_len}",
    )

    if args.save_video and frames:
        video_path = os.path.join(args.out_dir, "videos", f"video_{tag}_ret{int(ep_return)}.mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        imageio.mimsave(video_path, frames, fps=args.video_fps,
                         codec="libx264", pixelformat="yuv420p", macro_block_size=1)
        print(f"saved video -> {video_path}  ({len(frames)} frames)")

    env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sweep", "single"], default="sweep")
    p.add_argument("--condition", choices=CONDITIONS, default="heuristic",
                    help="single mode only: which policy to run")
    p.add_argument("--seed", type=int, default=0, help="single mode only")
    p.add_argument("--seeds", type=int, default=100, help="sweep mode only")
    p.add_argument("--frame-cost", type=float, default=0.0, help="single mode only -- actual reward frame_cost")
    p.add_argument("--fc-max", type=float, default=10.0, help="sweep mode only")
    p.add_argument("--fc-step", type=float, default=0.1, help="sweep mode only")
    p.add_argument("--budget", type=float, default=180)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--nav-model-path", default=NAV_MODEL_PATH)
    p.add_argument("--dist-thresh", type=float, default=0.1,
                    help="heuristic: sample at FPS=50 when normalized dist_to_curve is below this")
    p.add_argument("--severity-thresh", type=float, default=0.3,
                    help="heuristic: sample at FPS=50 when normalized curve_severity is above this")
    p.add_argument("--out-dir", default="frame_cost_study")
    p.add_argument("--workers", type=int, default=16, help="sweep mode only")
    p.add_argument("--save-video", action="store_true", help="single mode only")
    p.add_argument("--video-fps", type=int, default=30, help="single mode only")
    args = p.parse_args()

    if args.mode == "sweep":
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
