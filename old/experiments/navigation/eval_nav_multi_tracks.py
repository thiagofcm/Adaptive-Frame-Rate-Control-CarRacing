from datetime import datetime
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import argparse
import numpy as np
import torch
import gymnasium as gym
from old.experiments.navigation.train_nav import Agent, CarRacingPreprocessing
import csv 

SIM_FPS = 50
FPS_BASELINES = [1, 5, 10, 25, 50]


def make_eval_env(env_id):
    env = gym.make(env_id, continuous=False, render_mode="rgb_array",
                   lap_complete_percent=0.95, max_episode_steps=4000)
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    return env


def run_episode(env, agent, device, seed, fps):
    obs, _ = env.reset(seed=seed)
    obs_interval = int(SIM_FPS / fps)            # int: clean for 1,5,10,25,50
    last_sampled_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    ep_return, ep_len, done = 0.0, 0, False
    frames_consumed = 0
    # Goal Settings:
    track = np.asarray(env.unwrapped.track, dtype=np.float32)
    goal_frac   = 0.95
    goal_xy     = track[int(goal_frac * len(track)), 2:4]
    goal_radius = 20.0
    min_steps   = 20

    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        x, y = env.unwrapped.car.hull.position
        reached_goal = (np.hypot(x - goal_xy[0], y - goal_xy[1]) < goal_radius) \
                       and (env.step_counter > min_steps)

        if env.step_counter % obs_interval == 0:
            last_sampled_obs = obs_t
            frames_consumed += 1

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(last_sampled_obs, deterministic=True)

        obs, r, term, trunc, _ = env.step(action.item())
        ep_return += r
        ep_len += 1
        done = term or trunc or reached_goal

    return ep_return, ep_len, frames_consumed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env-id", default="CarRacing-v3")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="experiments/navigation/eval")
    
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%d-%m-%H-%M-%S")
    tag = f"feval_nav_policy_{date_str}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _Dummy:
        single_action_space = gym.spaces.Discrete(5)
    agent = Agent(_Dummy()).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    sd = ckpt["agent_state_dict"] if isinstance(ckpt, dict) and "agent_state_dict" in ckpt else ckpt
    agent.load_state_dict(sd)
    agent.eval()
    print(f"Loaded {args.ckpt}  (tag={tag})")

    seeds = [args.seed + i for i in range(args.episodes)]
    env = make_eval_env(args.env_id)

    per_episode_rows = []   # raw: one row per (fps, seed)
    summary = {}

    for fps in FPS_BASELINES:
        returns, lengths, frames = [], [], []
        for seed in seeds:
            print("Evaluating | fps: {:2d} | seed: {:3d}".format(fps, seed))
            ret, length, fc = run_episode(env, agent, device, seed, fps)
            returns.append(ret); lengths.append(length); frames.append(fc)
            per_episode_rows.append({
                "tag": tag, "fps": fps, "seed": seed,
                "return": ret, "length": length, "frames": fc,
            })

        returns, lengths, frames = np.array(returns), np.array(lengths), np.array(frames)
        summary[fps] = {
            "tag": tag, "fps": fps, "episodes": args.episodes,
            "ret_mean": returns.mean(), "ret_std": returns.std(),
            "ret_median": np.median(returns),
            "ret_min": returns.min(), "ret_max": returns.max(),
            "len_mean": lengths.mean(),
            "frames_mean": frames.mean(),
        }
        print(f"FPS {fps:2d} | return {returns.mean():7.1f} ± {returns.std():5.1f} "
              f"| median {np.median(returns):7.1f} | frames {frames.mean():6.0f} "
              f"| len {lengths.mean():.0f}")

    env.close()

    # --- write per-episode CSV (raw data) ---
    per_ep_path = os.path.join(args.out_dir, f"{tag}_per_episode.csv")
    with open(per_ep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tag", "fps", "seed", "return", "length", "frames"])
        w.writeheader()
        w.writerows(per_episode_rows)

    # --- write summary CSV (tradeoff table) ---
    summary_path = os.path.join(args.out_dir, f"{tag}_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fields = ["tag", "fps", "episodes", "ret_mean", "ret_std", "ret_median",
                  "ret_min", "ret_max", "len_mean", "frames_mean"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for fps in FPS_BASELINES:
            w.writerows([summary[fps]])

    # console table
    print("\n" + "=" * 60)
    print(f"{'FPS':>4} {'frames':>8} {'return':>10} {'std':>8} {'median':>8}")
    print("-" * 60)
    for fps in FPS_BASELINES:
        s = summary[fps]
        print(f"{fps:>4} {s['frames_mean']:>8.0f} {s['ret_mean']:>10.1f} "
              f"{s['ret_std']:>8.1f} {s['ret_median']:>8.1f}")

    print(f"\nSaved: {per_ep_path}")
    print(f"Saved: {summary_path}")

if __name__ == "__main__":
    main()