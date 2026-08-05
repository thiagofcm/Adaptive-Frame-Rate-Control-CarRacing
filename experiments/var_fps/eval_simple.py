"""
eval_simple.py

Simplified CarRacing var-FPS evaluator, ported from a LunarLander eval.py
(same overall shape: constants -> model loading -> per-episode rollout ->
CSV summary), adapted to this repo's env/wrapper stack:

  env:      CarRacing_VarFramerate      (envs/car_racing_var_fps.py)
  wrapper:  AdaptiveFPS_TrackAware_Wrapper (wrappers/adaptive_fps_track_aware_wrapper.py)
            -- loads its own frozen driving CNN (NavModel) from --nav-model-path,
            so this script never constructs a nav model itself.
  policy:   Agent (LSTM-PPO FPS-selector, experiments/var_fps/train_adaptive_fps_track_aware_lstm.py)

Unlike the LunarLander version, episodes are evaluated in parallel -- one whole
episode per worker process (concurrent.futures.ProcessPoolExecutor), matching
the pattern already used by eval_adaptive_fps_track_aware.py's --n-episodes
mode. CarRacing episodes are Box2D-physics + CNN-forward-pass heavy, so a
serial 100-episode loop in one process is far slower than this repo's other
CarRacing eval scripts already avoid.
"""
from datetime import datetime
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["OMP_NUM_THREADS"]     = "1"
os.environ["MKL_NUM_THREADS"]     = "1"
os.environ["OPENBLAS_NTHREADS"]   = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

import os
import sys
import math
import argparse
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import envs.car_racing_var_fps  # noqa: F401 -- registers "CarRacing_VarFramerate"
from envs.car_racing_var_fps import CURVE_TILE_TURN_THRESHOLD
from wrappers.pre_processing import CarRacingPreprocessing
from wrappers.adaptive_fps_track_aware_wrapper import AdaptiveFPS_TrackAware_Wrapper
from experiments.var_fps.train_adaptive_fps_track_aware_lstm import Agent as AgentEval
from utils.cautious_variables import OFF_TRACK_WHEEL_THRESHOLD

# =========================
# USER INPUT
# =========================
ENV_ID              = "CarRacing_VarFramerate"
OBS_DIM             = 14
ACTION_SPACE_LENGTH = 5
LSTM_HIDDEN_SIZE    = 64
FPS_TO_ACTION       = {1: 0, 5: 1, 10: 2, 25: 3, 50: 4}

N_EPISODES        = 100
N_RUNS            = 1
RUN_SEED          = 42
MAX_EVAL_WORKERS  = 16
NAV_MODEL_PATH    = "experiments/navigation/CarRacing-v3__train__1__1785768450/agent_step3276800.pt"
MAX_EPISODE_STEPS = 1000

# Consecutive non-matching ticks required before a curve or straight segment is
# considered over -- a couple of stray classification flips shouldn't split one
# physical segment into two. Applies symmetrically to both segment types.
EXIT_PATIENCE = 3


# ─────────────────────────────────────────
# Env / model loading
# ─────────────────────────────────────────

def make_env(nav_model_path, frame_cost, budget, max_episode_steps):
    env = gym.make(ENV_ID, continuous=False, render_mode="rgb_array")
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    # NavModel stays on CPU regardless of the FPS-selector's device: single-sample
    # CNN inference every physics tick is dominated by call overhead, not compute.
    env = AdaptiveFPS_TrackAware_Wrapper(env, nav_model_path, device="cpu",
                                          frame_cost=frame_cost, budget=budget)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def load_agent(ckpt_path, device):
    class _Dummy:
        single_observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        single_action_space = gym.spaces.Discrete(ACTION_SPACE_LENGTH)

    agent = AgentEval(_Dummy(), lstm_hidden_size=LSTM_HIDDEN_SIZE).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    agent.load_state_dict(sd)
    agent.eval()
    return agent


# ─────────────────────────────────────────
# Utils
# ─────────────────────────────────────────

def get_seeds(run, n_episodes):
    return [RUN_SEED + run * n_episodes + i for i in range(n_episodes)]


# ─────────────────────────────────────────
# Evaluate a single episode (one ProcessPoolExecutor job)
# ─────────────────────────────────────────

def run_episode(job):
    (run, seed, model_path, nav_model_path, frame_cost, budget, fixed, max_episode_steps) = job

    env = make_env(nav_model_path, frame_cost, budget, max_episode_steps)
    device = torch.device("cpu")

    agent = None
    lstm_state = None
    done_t = None
    if fixed == 0:
        agent = load_agent(model_path, device)
        lstm_state = (
            torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size, device=device),
            torch.zeros(agent.lstm.num_layers, 1, agent.lstm.hidden_size, device=device),
        )
        done_t = torch.zeros(1, device=device)

    obs, _ = env.reset(seed=seed)
    terminated, truncated = False, False

    total_reward     = 0.0
    total_nav_reward = 0.0
    fps_trace         = []
    off_track_ticks    = 0
    tick_count          = 0
    info = {}

    # Per-curve-segment speed tracking (see module-level EXIT_PATIENCE constant).
    curve_mean_speeds        = []   # one finalized mean speed per distinct curve
    in_curve                 = False
    current_curve_speed_sum  = 0.0
    current_curve_tick_count = 0
    non_curve_streak         = 0

    # Per-straight-segment speed tracking, symmetric to the curve tracking above.
    straight_mean_speeds        = []   # one finalized mean speed per distinct straight
    in_straight                 = False
    current_straight_speed_sum  = 0.0
    current_straight_tick_count = 0
    non_straight_streak         = 0

    while not (terminated or truncated):
        if fixed != 0:
            action = FPS_TO_ACTION[fixed]
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, _, _, _, lstm_state = agent.get_action_and_value(
                    obs_t, lstm_state, done_t, deterministic=True
                )
            action = int(a.item())

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward     += reward
        total_nav_reward += info.get("nav_reward", 0.0)
        fps_trace.append(info["chosen_fps"])

        # Off-track duration: same >=N wheels-off-road threshold the wrapper's
        # own "off_track"/"time_off_track" observation features are trained on.
        off_track_wheel_count = sum(len(w.tiles) == 0 for w in env.unwrapped.car.wheels)
        if off_track_wheel_count >= OFF_TRACK_WHEEL_THRESHOLD:
            off_track_ticks += 1
        tick_count += 1

        # Curve detection: same tile-under-car lookup the wrapper uses for its
        # wrong-direction check, and the same beta-delta curve test as
        # CarRacing_VarFramerate._is_curve_tile() (envs/car_racing_var_fps.py).
        current_tile_idx = None
        for w in env.unwrapped.car.wheels:
            if len(w.tiles) > 0:
                current_tile_idx = next(iter(w.tiles)).idx
                break

        is_curve_tick = False
        if current_tile_idx is not None:
            track = env.unwrapped.track
            n_tiles = len(track)
            beta_here = track[current_tile_idx][1]
            beta_next = track[(current_tile_idx + 1) % n_tiles][1]
            dh = (beta_next - beta_here + math.pi) % (2 * math.pi) - math.pi
            is_curve_tick = abs(dh) > CURVE_TILE_TURN_THRESHOLD
        # A straight tick requires a valid on-track tile too -- ticks where
        # current_tile_idx is None (off track / no tile identified) are neither
        # curve nor straight ticks.
        is_straight_tick = current_tile_idx is not None and not is_curve_tick

        if is_curve_tick or is_straight_tick:
            speed = float(np.linalg.norm(env.unwrapped.car.hull.linearVelocity))

        if is_curve_tick:
            if not in_curve:
                # Entering a new curve segment.
                in_curve = True
                current_curve_speed_sum = 0.0
                current_curve_tick_count = 0
            current_curve_speed_sum  += speed
            current_curve_tick_count += 1
            non_curve_streak = 0
        elif in_curve:
            # Exit-patience: only close the segment after EXIT_PATIENCE
            # consecutive non-curve ticks. These patience ticks are not curve
            # ticks themselves, so their speed is never added to the sum above.
            non_curve_streak += 1
            if non_curve_streak >= EXIT_PATIENCE:
                if current_curve_tick_count > 0:
                    curve_mean_speeds.append(current_curve_speed_sum / current_curve_tick_count)
                in_curve = False
                current_curve_speed_sum = 0.0
                current_curve_tick_count = 0
                non_curve_streak = 0

        if is_straight_tick:
            if not in_straight:
                # Entering a new straight segment.
                in_straight = True
                current_straight_speed_sum = 0.0
                current_straight_tick_count = 0
            current_straight_speed_sum  += speed
            current_straight_tick_count += 1
            non_straight_streak = 0
        elif in_straight:
            # Exit-patience: only close the segment after EXIT_PATIENCE
            # consecutive non-straight ticks. These patience ticks are not
            # straight ticks themselves, so their speed is never added above.
            non_straight_streak += 1
            if non_straight_streak >= EXIT_PATIENCE:
                if current_straight_tick_count > 0:
                    straight_mean_speeds.append(current_straight_speed_sum / current_straight_tick_count)
                in_straight = False
                current_straight_speed_sum = 0.0
                current_straight_tick_count = 0
                non_straight_streak = 0

    # Episode ended while still on a curve/straight -- save those segments too.
    if in_curve and current_curve_tick_count > 0:
        curve_mean_speeds.append(current_curve_speed_sum / current_curve_tick_count)
    if in_straight and current_straight_tick_count > 0:
        straight_mean_speeds.append(current_straight_speed_sum / current_straight_tick_count)

    frame_count = info["episode_frame_count"]
    successful  = bool(info["reached_goal"])
    time_off_track = (off_track_ticks / tick_count) if tick_count > 0 else 0.0

    env.close()

    print(f"  Episode {seed - RUN_SEED + 1}/{N_EPISODES} | reward={total_reward:.2f} | "
          f"success={successful} | frames={frame_count}")

    return {
        "run": run,
        "reward": total_reward,
        "nav_reward": total_nav_reward,
        "frames": frame_count,
        "success": float(successful),
        "time_off_track": time_off_track,
        "curve_mean_speeds": curve_mean_speeds,
        "straight_mean_speeds": straight_mean_speeds,
        "fps_trace": fps_trace,
    }


# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--fc", type=float, required=True)
    parser.add_argument("--budget", type=float, required=True)
    parser.add_argument("--fixed", type=float, default=0.0, required=False)
    parser.add_argument("--nav-model-path", type=str, default=NAV_MODEL_PATH)
    parser.add_argument("--workers", type=int, default=MAX_EVAL_WORKERS)
    parser.add_argument("--max-episode-steps", type=int, default=MAX_EPISODE_STEPS)
    args = parser.parse_args()

    model_path     = Path(args.model)
    frame_cost     = args.fc
    budget         = args.budget
    parent_folder  = model_path.parent
    results_folder = "score_results"
    output_dir     = os.path.join(results_folder, parent_folder)
    fixed          = args.fixed
    os.makedirs(output_dir, exist_ok=True)

    if fixed == 0 and not model_path.exists():
        sys.exit(f"--model checkpoint not found: {model_path}")
    if not Path(args.nav_model_path).exists():
        sys.exit(f"--nav-model-path checkpoint not found: {args.nav_model_path}")

    print(f"\nStarting evaluation: {N_RUNS} runs x {N_EPISODES} episodes each, "
          f"{args.workers} parallel workers")
    print(f"Seeds: {RUN_SEED} – {RUN_SEED + N_EPISODES - 1}")

    jobs = [
        (run, seed, str(model_path), args.nav_model_path, frame_cost, budget,
         fixed, args.max_episode_steps)
        for run in range(N_RUNS)
        for seed in get_seeds(run, N_EPISODES)
    ]

    run_metrics = {
        "rewards":        [[] for _ in range(N_RUNS)],
        "nav_rewards":    [[] for _ in range(N_RUNS)],
        "frames":         [[] for _ in range(N_RUNS)],
        "success":        [[] for _ in range(N_RUNS)],
        "time_off_track": [[] for _ in range(N_RUNS)],
    }
    all_curve_mean_speeds    = []  # one entry per distinct curve,    across every episode/worker
    all_straight_mean_speeds = []  # one entry per distinct straight, across every episode/worker

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_episode, job) for job in jobs]
        for fut in as_completed(futures):
            result = fut.result()
            r = result["run"]
            run_metrics["rewards"][r].append(result["reward"])
            run_metrics["nav_rewards"][r].append(result["nav_reward"])
            run_metrics["frames"][r].append(result["frames"])
            run_metrics["success"][r].append(result["success"])
            run_metrics["time_off_track"][r].append(result["time_off_track"])
            all_curve_mean_speeds.extend(result["curve_mean_speeds"])
            all_straight_mean_speeds.extend(result["straight_mean_speeds"])

    for key in run_metrics:
        run_metrics[key] = np.array(run_metrics[key])

    # Mean/std across all N_RUNS x N_EPISODES episode outcomes, flattened --
    # NOT axis=0 (which, at N_RUNS=1, computes std across a single-value
    # "column" per episode and always reports 0).
    rew_m,  rew_std  = float(run_metrics["rewards"].mean()),        float(run_metrics["rewards"].std())
    nav_m,  nav_std  = float(run_metrics["nav_rewards"].mean()),    float(run_metrics["nav_rewards"].std())
    frm_m,  frm_std  = float(run_metrics["frames"].mean()),         float(run_metrics["frames"].std())
    succ_m, succ_std = float(run_metrics["success"].mean() * 100),  float(run_metrics["success"].std() * 100)
    tot_m,  tot_std  = float(run_metrics["time_off_track"].mean()), float(run_metrics["time_off_track"].std())

    # Mean/std across distinct curve/straight segments (each already averaged
    # over its own ticks in run_episode) -- every segment counts equally
    # regardless of length, unlike averaging over raw ticks.
    n_curves = len(all_curve_mean_speeds)
    if n_curves > 0:
        mean_speed_at_curves = float(np.mean(all_curve_mean_speeds))
        std_speed_at_curves  = float(np.std(all_curve_mean_speeds))
    else:
        mean_speed_at_curves = float("nan")
        std_speed_at_curves  = float("nan")

    n_straights = len(all_straight_mean_speeds)
    if n_straights > 0:
        mean_speed_at_straights = float(np.mean(all_straight_mean_speeds))
        std_speed_at_straights  = float(np.std(all_straight_mean_speeds))
    else:
        mean_speed_at_straights = float("nan")
        std_speed_at_straights  = float("nan")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    output_dir = "score_results/runs"
    os.makedirs(output_dir, exist_ok=True)

    if fixed > 0:
        csv_path = os.path.join(output_dir, f"eval_results_fixed_{fixed}.csv")
    else:
        date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = os.path.join(output_dir, f"eval_results_adaptive_{date_str}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "frame_cost", "budget", "n_runs", "n_episodes",
            "reward_mean",     "reward_std",
            "nav_reward_mean", "nav_reward_std",
            "frames_mean",     "frames_std",
            "success_pct_mean","success_pct_std",
            "time_off_track_mean", "time_off_track_std",
            "mean_speed_at_curves", "std_speed_at_curves", "n_curves",
            "mean_speed_at_straights", "std_speed_at_straights", "n_straights",
        ])
        writer.writerow([
            str(model_path), frame_cost, budget, N_RUNS, N_EPISODES,
            f"{rew_m:.4f}",  f"{rew_std:.4f}",
            f"{nav_m:.4f}",  f"{nav_std:.4f}",
            f"{frm_m:.2f}",  f"{frm_std:.2f}",
            f"{succ_m:.2f}", f"{succ_std:.2f}",
            f"{tot_m:.4f}",  f"{tot_std:.4f}",
            f"{mean_speed_at_curves:.4f}", f"{std_speed_at_curves:.4f}", n_curves,
            f"{mean_speed_at_straights:.4f}", f"{std_speed_at_straights:.4f}", n_straights,
        ])

    print(f"\nResults saved → {csv_path}")

    # ── Console summary ──────────────────────────────────────────────────────
    print("\n===== Evaluation Summary =====")
    print(f"Model              : {model_path}")
    print(f"Runs x Eps         : {N_RUNS} x {N_EPISODES} = {N_RUNS*N_EPISODES} total episodes")
    print(f"Reward             : {rew_m:.4f} ± {rew_std:.4f}")
    print(f"Nav Reward         : {nav_m:.4f} ± {nav_std:.4f}")
    print(f"Frames             : {frm_m:.2f} ± {frm_std:.2f}")
    print(f"Success            : {succ_m:.2f}% ± {succ_std:.2f}%")
    print(f"Time off track     : {tot_m:.4f} ± {tot_std:.4f}")
    print(f"Speed at curves    : {mean_speed_at_curves:.4f} ± {std_speed_at_curves:.4f}  (n_curves={n_curves})")
    print(f"Speed at straights : {mean_speed_at_straights:.4f} ± {std_speed_at_straights:.4f}  (n_straights={n_straights})")
