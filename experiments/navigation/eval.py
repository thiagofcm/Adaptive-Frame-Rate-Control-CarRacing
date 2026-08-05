from datetime import datetime
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
# Each --workers subprocess runs a single episode's env.step()/Agent-forward loop --
# no benefit from internal BLAS/OpenMP multithreading, only oversubscription against
# the other workers. Must be set before numpy/torch import -- same pattern as
# experiments/var_fps/eval_adaptive_fps_track_aware.py.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
torch.set_num_threads(1)
import gymnasium as gym
from experiments.navigation.train import Agent, CarRacingPreprocessing
from utils.cautious_variables import CautiousVars, OFF_TRACK_WHEEL_THRESHOLD

SIM_FPS = 50
FPS_BASELINES = [50]

# Batch-eval seed group -- deliberately a plain constant, not a CLI flag, so results
# stay reproducible without needing a flag passed consistently between invocations.
# Seeds are RUN_SEED..RUN_SEED+n-1.
# Matches the convention in experiments/var_fps/eval_adaptive_fps_track_aware.py.
RUN_SEED = 42

# Field order returned by CautiousVars.get_cautious_var() -- see
# utils/cautious_variables.py. Column names below are prefixed with "mean_"/"std_"
# since these are per-tick in the raw sim and get aggregated to one value per episode.
CAUTIOUS_FIELDS = [
    "vx", "vy", "dist_to_curve", "curve_severity_norm", "heading_alignment",
    "cross_track", "cross_track_rate_norm", "off_track", "time_off_track_norm",
    "episode_completion", "curves_passed_norm",
]


def make_eval_env(env_id):
    env = gym.make(env_id, continuous=False, render_mode="rgb_array",
                   lap_complete_percent=0.95, max_episode_steps=4000)
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    return env


def load_agent(model_path, device):
    class _Dummy:
        single_action_space = gym.spaces.Discrete(5)
    agent = Agent(_Dummy()).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    sd = checkpoint["agent_state_dict"] if isinstance(checkpoint, dict) and "agent_state_dict" in checkpoint else checkpoint
    agent.load_state_dict(sd)
    agent.eval()
    return agent


def run_episode(env, agent, device, seed, fps):
    obs, _ = env.reset(seed=seed)
    obs_interval = int(SIM_FPS / fps)            # int: clean for 1,5,10,25,50
    last_sampled_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    ep_return, ep_len, done = 0.0, 0, False
    off_track_ticks = 0
    cautious_rows = []
    # Goal Settings:
    track = np.asarray(env.unwrapped.track, dtype=np.float32)
    goal_frac   = 0.95
    goal_xy     = track[int(goal_frac * len(track)), 2:4]
    goal_radius = 20.0
    min_steps   = 20

    cv = CautiousVars()
    cv.reset_track_reading(env)
    # curves_passed_norm reads env.unwrapped.curves_passed_count/total_curves_on_track,
    # which only CarRacing_VarFramerate (envs/car_racing_var_fps.py) tracks -- the stock
    # CarRacing-v3 this eval defaults to has no curve-passing concept. Shim zero/one so
    # get_cautious_var() doesn't AttributeError; curves_passed_norm then just reads 0.0
    # for every tick under the default --env-id, and reads real values if --env-id is
    # pointed at CarRacing_VarFramerate instead (hasattr already True there, no-op).
    if not hasattr(env.unwrapped, "curves_passed_count"):
        env.unwrapped.curves_passed_count = 0
        env.unwrapped.total_curves_on_track = 1

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        x, y = env.unwrapped.car.hull.position
        reached_goal = (np.hypot(x - goal_xy[0], y - goal_xy[1]) < goal_radius) \
                       and (env.step_counter > min_steps)

        if env.step_counter % obs_interval == 0:
            last_sampled_obs = obs_t

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(last_sampled_obs, deterministic=True)

        obs, r, term, trunc, _ = env.step(action.item())
        ep_return += r
        ep_len += 1
        done = term or trunc or reached_goal

        cautious_rows.append(cv.get_cautious_var(env, dt_ticks=1))
        off_track_wheel_count = sum(len(w.tiles) == 0 for w in env.unwrapped.car.wheels)
        if off_track_wheel_count >= OFF_TRACK_WHEEL_THRESHOLD:
            off_track_ticks += 1

    cautious_mean = np.mean(np.stack(cautious_rows), axis=0)

    return {
        "reward": ep_return,
        # No frame_cost/budget wrapper in this pipeline (unlike
        # AdaptiveFPS_TrackAware_Wrapper's info["nav_reward"]) -- reward IS the nav reward.
        "nav_reward": ep_return,
        "reached_goal": bool(reached_goal),
        "off_track_rate": off_track_ticks / ep_len if ep_len > 0 else 0.0,
        "cautious_mean": cautious_mean,
    }


def _worker(job):
    """Top-level (picklable) per-(fps, seed) job for ProcessPoolExecutor -- same
    flattening pattern as eval_adaptive_fps_track_aware.py's _metrics_worker. Each
    worker builds its own env + reloads its own copy of the model checkpoint. Stays on
    CPU regardless of GPU availability: single-sample inference every tick is dominated
    by call overhead, not compute, so CPU is faster here, and it sidesteps fork+CUDA
    hazards from ProcessPoolExecutor's default fork start method on Linux.
    """
    fps, seed, model_path, env_id = job
    device = torch.device("cpu")
    env = make_eval_env(env_id)
    agent = load_agent(model_path, device)
    result = run_episode(env, agent, device, seed, fps)
    env.close()
    return fps, seed, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--env-id", default="CarRacing-v3")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out-dir", default="experiments/navigation/eval")
    p.add_argument("--workers", type=int, default=32,
                    help="parallel worker processes, one per (fps, seed) episode -- "
                         "same pattern as eval_adaptive_fps_track_aware.py")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%d-%m-%H-%M-%S")
    tag = f"eval_nav_policy_{date_str}"
    print(f"Evaluating model={args.model}  env_id={args.env_id}  episodes={args.episodes}  (tag={tag})")

    seeds = [RUN_SEED + i for i in range(args.episodes)]
    jobs = [(fps, seed, args.model, args.env_id) for fps in FPS_BASELINES for seed in seeds]

    print(f"[eval] running {len(jobs)} episodes across {args.workers} workers...")
    grouped = {fps: [] for fps in FPS_BASELINES}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            fps, seed, result = fut.result()
            grouped[fps].append((seed, result))
            done += 1
            print(f"  [fps={fps:2d}] seed={seed} | reward={result['reward']:.2f} | "
                  f"reached_goal={result['reached_goal']}")
            if done % 25 == 0 or done == len(jobs):
                print(f"[eval] {done}/{len(jobs)} episodes done")

    # sort each fps's episodes by seed -- jobs complete out of order once
    # parallelized, this keeps the per-episode CSV rows in a stable, predictable order.
    for fps in FPS_BASELINES:
        grouped[fps].sort(key=lambda t: t[0])

    per_episode_fields = (["episode", "seed", "fps", "reward", "nav_reward",
                            "reached_goal", "off_track_rate"]
                           + [f"mean_{name}" for name in CAUTIOUS_FIELDS])
    summary_fields = (["fps", "n_episodes", "mean_reward", "std_reward",
                        "mean_nav_reward", "std_nav_reward",
                        "reached_goal_rate", "off_track_rate"])
    for name in CAUTIOUS_FIELDS:
        summary_fields += [f"mean_{name}", f"std_{name}"]

    per_episode_paths = []
    summary_rows = []

    for fps in FPS_BASELINES:
        episodes = grouped[fps]

        rows = []
        for episode_idx, (seed, result) in enumerate(episodes, start=1):
            row = {
                "episode": episode_idx, "seed": seed, "fps": fps,
                "reward": round(float(result["reward"]), 3),
                "nav_reward": round(float(result["nav_reward"]), 3),
                "reached_goal": result["reached_goal"],
                "off_track_rate": round(float(result["off_track_rate"]), 4),
            }
            for name, val in zip(CAUTIOUS_FIELDS, result["cautious_mean"]):
                row[f"mean_{name}"] = round(float(val), 4)
            rows.append(row)

        per_ep_path = os.path.join(args.out_dir, f"{tag}_fps{fps}_per_episode.csv")
        with open(per_ep_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=per_episode_fields)
            w.writeheader()
            w.writerows(rows)
        per_episode_paths.append(per_ep_path)

        rewards = np.array([r["reward"] for _, r in episodes])
        nav_rewards = np.array([r["nav_reward"] for _, r in episodes])
        reached = np.array([r["reached_goal"] for _, r in episodes])
        off_track = np.array([r["off_track_rate"] for _, r in episodes])
        cautious_stack = np.stack([r["cautious_mean"] for _, r in episodes])  # (n_episodes, len(CAUTIOUS_FIELDS))

        summary_row = {
            "fps": fps, "n_episodes": len(episodes),
            "mean_reward": round(float(rewards.mean()), 3), "std_reward": round(float(rewards.std()), 3),
            "mean_nav_reward": round(float(nav_rewards.mean()), 3), "std_nav_reward": round(float(nav_rewards.std()), 3),
            "reached_goal_rate": round(float(reached.mean()), 4),
            "off_track_rate": round(float(off_track.mean()), 4),
        }
        for i, name in enumerate(CAUTIOUS_FIELDS):
            summary_row[f"mean_{name}"] = round(float(cautious_stack[:, i].mean()), 4)
            summary_row[f"std_{name}"] = round(float(cautious_stack[:, i].std()), 4)
        summary_rows.append(summary_row)

        print(f"FPS {fps:2d} | reward {summary_row['mean_reward']:7.1f} ± {summary_row['std_reward']:5.1f} "
              f"| reached_goal {summary_row['reached_goal_rate']:6.1%} "
              f"| off_track {summary_row['off_track_rate']:6.1%}")

    summary_path = os.path.join(args.out_dir, f"{tag}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print("\n" + "=" * 60)
    print(f"{'FPS':>4} {'reward':>10} {'std':>8} {'reached_goal':>13} {'off_track':>10}")
    print("-" * 60)
    for s in summary_rows:
        print(f"{s['fps']:>4} {s['mean_reward']:>10.1f} {s['std_reward']:>8.1f} "
              f"{s['reached_goal_rate']:>13.1%} {s['off_track_rate']:>10.1%}")

    print(f"\nSaved per-FPS per-episode CSVs: {per_episode_paths}")
    print(f"Saved summary: {summary_path}")

if __name__ == "__main__":
    main()
