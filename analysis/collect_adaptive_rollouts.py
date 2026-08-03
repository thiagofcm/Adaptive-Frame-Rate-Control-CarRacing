"""
collect_adaptive_rollouts.py

Data-collection step for the curvature-attention investigation (Steps 1b/2a of the
FPS-policy investigation plan). Drives a trained FPS-selection checkpoint
(train_adaptive_fps_track_aware_lstm.py's Agent) against the frozen nav-controller,
and at every REAL decision instant (info["frame_consumed"]==True) records:

  - the policy's full action distribution (entropy, top-1 probability) -- exposed
    directly here (not just via the training script's masked-average entropy_loss),
    so Step 1b can check whether the policy is actually confident/state-conditioned
    at the specific states it visits, not just on masked-average over the rollout.
  - ground-truth track curvature at the car's current position, read from a
    FRESH, independently-tracked CautiousVars instance (reset_track_reading() +
    per-tick _nearest_idx() calls every physics tick) -- deliberately NOT reusing
    the wrapper's own `cautious_sensors`, which only refreshes at sampling instants
    and is therefore stale by design between decisions. We want the true curvature
    at the car's actual current position regardless of what the policy currently
    perceives, so the segmentation in Step 2a isn't itself confounded by staleness.

One row per (seed, decision). Unit of analysis is the decision, not the physics
tick -- consecutive ticks under a held FPS choice are not independent draws (see
investigation plan Step 2a), so per-tick rows would pseudo-replicate.

Usage:
  python analysis/collect_adaptive_rollouts.py --ckpt runs/.../final.pt \
      --n-episodes 150 --seed-start 100000 --workers 64 --out-csv analysis_out/adaptive_rollouts.csv
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
import torch
torch.set_num_threads(1)

# Repo root (this file lives in analysis/) -- needed so the top-level eval script
# and utils package are importable regardless of the caller's cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_adaptive_fps_track_aware import make_eval_env, load_adaptive_agent, FPS_CHOICES, NAV_MODEL_PATH
from utils.cautious_variables import CautiousVars


def run_episode(ckpt_path, seed, nav_model_path, env_id, fc, budget, max_episode_steps):
    device = torch.device("cpu")
    agent = load_adaptive_agent(ckpt_path, device)
    env = make_eval_env(env_id, nav_model_path, fc, budget, max_episode_steps)
    obs, _ = env.reset(seed=seed)

    # Fresh, independently-tracked ground-truth curvature reader -- see module
    # docstring for why this is NOT the wrapper's own (stale-by-design) cautious_sensors.
    gt = CautiousVars()
    gt.reset_track_reading(env)

    lstm_state = (
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size),
        torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size),
    )
    done_t = torch.zeros(1)

    rows = []
    tick = 0
    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            hidden, lstm_state = agent.get_states(obs_t, lstm_state, done_t)
            logits = agent.actor(hidden)
            probs = torch.softmax(logits, dim=-1)[0]
            entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())
            top1_prob = float(probs.max().item())
            action = int(torch.argmax(logits, dim=-1).item())

        obs, r, term, trunc, info = env.step(action)
        tick += 1
        done = term or trunc

        # Nearest-node search every physics tick (not just decision instants) so the
        # local search window in CautiousVars._nearest_idx stays valid regardless of
        # how far apart decisions are spaced (e.g. FPS=1 -> 50 ticks between decisions,
        # during which the car keeps moving at the full physics rate).
        x, y = env.unwrapped.car.hull.position
        idx = gt._nearest_idx(np.array([x, y], dtype=np.float64))

        if info["frame_consumed"]:
            rows.append({
                "seed": seed,
                "tick": tick,
                "chosen_fps": int(info["chosen_fps"]),
                "entropy": entropy,
                "top1_prob": top1_prob,
                "curv_signed": float(gt.curv[idx]),
                "curv_abs": abs(float(gt.curv[idx])),
                "turn": float(gt.turn[idx]),
                "in_curve": bool(env.unwrapped.in_curve),
                "off_track": int(sum(len(w.tiles) == 0 for w in env.unwrapped.car.wheels) >= 2),
                "track_hash": info["track_hash"],
            })

    env.close()
    return rows


def _worker(job):
    seed, ckpt_path, nav_model_path, env_id, fc, budget, max_episode_steps = job
    return run_episode(ckpt_path, seed, nav_model_path, env_id, fc, budget, max_episode_steps)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--env-id", default="CarRacing_VarFramerate")
    p.add_argument("--nav-model-path", default=NAV_MODEL_PATH)
    p.add_argument("--fc", type=float, default=1.8, help="frame_cost the checkpoint was trained under")
    p.add_argument("--budget", type=float, default=180)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--n-episodes", type=int, default=150)
    p.add_argument("--seed-start", type=int, default=100000,
                    help="deliberately far from RUN_SEED=42 (eval script) and seeds 0-19 "
                         "(frame_cost_calibration.py) so this analysis's seeds don't overlap "
                         "with any prior eval/calibration CSVs")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    jobs = [(s, args.ckpt, args.nav_model_path, args.env_id, args.fc, args.budget, args.max_episode_steps)
            for s in seeds]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["seed", "tick", "chosen_fps", "entropy", "top1_prob",
                  "curv_signed", "curv_abs", "turn", "in_curve", "off_track", "track_hash"]

    print(f"[collect_adaptive_rollouts] running {len(jobs)} episodes across {args.workers} workers...")
    all_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        done = 0
        for fut in as_completed(futures):
            all_rows.extend(fut.result())
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"[collect_adaptive_rollouts] {done}/{len(jobs)} episodes done, "
                      f"{len(all_rows)} decisions logged so far")

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[collect_adaptive_rollouts] wrote {len(all_rows)} decision rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
