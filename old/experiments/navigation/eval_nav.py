import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")          # headless cluster: no display, must set before pyplot
import matplotlib.pyplot as plt

# Reuse classes from your training script
from old.experiments.navigation.train_nav import Agent, CarRacingPreprocessing, image_preprocessing

SIM_FPS = 50

def make_eval_env(env_id, seed):
    env = gym.make(env_id, continuous=False, render_mode="rgb_array", lap_complete_percent=0.95, max_episode_steps=4000)
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    env.reset(seed=seed)
    return env

def plot_track_trajectory(env, positions, speeds, save_path,
                          title=None, stall_speed=1.0):
    """
    Top-down map of the full CarRacing track with the agent's path overlaid.

    env        : the (wrapped) env — env.unwrapped.track is read for the map.
                 MUST be called before env.close().
    positions  : list of (x, y) world coords, one per step.
    speeds     : list of |linearVelocity| per step (same length as positions).
    save_path  : output .png path.
    stall_speed: speed below which a point is marked as 'stalled'.
    """
    track = np.asarray(env.unwrapped.track, dtype=np.float32)   # (N,4): alpha,beta,x,y
    tx, ty = track[:, 2], track[:, 3]

    traj = np.asarray(positions, dtype=np.float32)
    spd  = np.asarray(speeds, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(7, 7))

    # track: thick grey band + dashed centerline, loop closed
    txc, tyc = np.append(tx, tx[0]), np.append(ty, ty[0])
    ax.plot(txc, tyc, color="0.75", lw=8, solid_capstyle="round", zorder=1)
    ax.plot(txc, tyc, "k--", lw=1, zorder=2)

    # trajectory colored by speed
    sc = ax.scatter(traj[:, 0], traj[:, 1], c=spd, cmap="viridis", s=10, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.5)
    cbar.set_label("speed |v|")

    # ring the stalls
    stall = spd < stall_speed
    if stall.any():
        ax.scatter(traj[stall, 0], traj[stall, 1], facecolors="none",
                   edgecolors="red", s=70, lw=1.2, zorder=4,
                   label=f"stalled (|v| < {stall_speed})")

    # start / end
    ax.plot(traj[0, 0], traj[0, 1], "o", color="lime", ms=11, zorder=5, label="start")
    ax.plot(traj[-1, 0], traj[-1, 1], "X", color="red", ms=11, zorder=2, label="end")

    print(f"The trajectory ended at: {traj[-1, 0], traj[-1, 1]}")

    goal_frac   = 0.95                       # how far around the loop to finish
    goal_idx    = int(goal_frac * len(track))
    goal_xy     = track[goal_idx, 2:4]

    print(f"The trajectory ended at: ({traj[-1, 0]:.1f}, {traj[-1, 1]:.1f})")
    print(f"Goal ({goal_frac:.0%} around) at:  ({goal_xy[0]:.1f}, {goal_xy[1]:.1f})")

    ax.plot(goal_xy[0], goal_xy[1], "*", color="gold", ms=20,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6,
            label=f"goal ({goal_frac:.0%})")

    # optional: show the acceptance radius
    goal_radius = 20.0
    ax.add_patch(plt.Circle((goal_xy[0], goal_xy[1]), goal_radius,
                            fill=False, color="gold", lw=1.2, ls="--", zorder=6))

    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title)
    fig.legend(loc="lower center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 0.0), frameon=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return save_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to .pt file")
    p.add_argument("--env-id", default="CarRacing-v3")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--video-dir", default="eval_videos")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--fps", type=int, default=50)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.video_dir, exist_ok=True)

    # --- build dummy vector env just to satisfy Agent's __init__ ---
    # Agent needs envs.single_action_space.n during construction
    class _Dummy:
        single_action_space = gym.spaces.Discrete(5)
    agent = Agent(_Dummy()).to(device)

    # --- load checkpoint ---
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
        agent.load_state_dict(ckpt["agent_state_dict"])
        print(f"Loaded checkpoint from iteration {ckpt.get('iteration', '?')}, "
              f"global_step {ckpt.get('global_step', '?')}")
    else:
        agent.load_state_dict(ckpt)  # plain state_dict (e.g. final.pt)
        print(f"Loaded raw state_dict from {args.ckpt}")
    agent.eval()

    returns, lengths = [], []
    for ep in range(args.episodes):
        env = make_eval_env(args.env_id, args.seed + ep)
        obs, _ = env.reset(seed=args.seed)
        frames, ep_return, ep_len, done = [], 0.0, 0, False
        obs_interval = SIM_FPS/args.fps
        last_sampled_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        positions, speeds = [], []

        track = np.asarray(env.unwrapped.track, dtype=np.float32) 
        goal_frac   = 0.95
        goal_idx    = int(goal_frac * len(track))
        goal_xy     = track[goal_idx, 2:4]
        goal_radius = 20.0                    # world units; tune to your scale
        min_steps   = 20
        total_frame_counter = 0

        while not done:
            x, y = env.unwrapped.car.hull.position
            vx, vy = env.unwrapped.car.hull.linearVelocity

            reached = (np.hypot(x - goal_xy[0], y - goal_xy[1]) < goal_radius) and (ep_len > min_steps)

            positions.append((x, y))
            speeds.append(float(np.hypot(vx, vy)))

            if args.save_video:
                frames.append(env.unwrapped.render())   # raw 96x96 RGB

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            if env.step_counter % obs_interval == 0:
                last_sampled_obs = obs_t
                frame_consumed = True
                obs_type = 'Fresh'
                total_frame_counter+=1
            else:
                print(f" Step: {env.step_counter} - Stale Observation - X: {x}, Y: {y}")
                frame_consumed = False
                obs_type = 'Fresh'
            
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(last_sampled_obs, deterministic=True)

            print(f" Step: {env.step_counter} - {obs_type} Observation - Frames Consumed: {total_frame_counter} -  X: {x}, Y: {y}, Action:{action}")
            obs, r, term, trunc, _ = env.step(action.item())
            ep_return += r
            ep_len += 1
            done = term or trunc or reached

        print(f"Episode completed consuming {total_frame_counter} frames")

        plot_track_trajectory(
            env, positions, speeds,
            save_path=os.path.join(args.video_dir, f"traj_seed{args.seed}_fps{args.fps}_ret{int(ep_return)}_goal.png"),
            title=f"fps={args.fps}  return={ep_return:.0f}  len={ep_len}",
        )
        env.close()
        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"ep {ep}: return={ep_return:7.1f}  length={ep_len}")

        if args.save_video and frames:
            run_string  = args.ckpt.split("/")[-2]
            ckpt_string = args.ckpt.split("/")[-1].split(".")[0]
            video_dir = os.path.join("eval_videos", run_string,ckpt_string)
            os.makedirs(video_dir, exist_ok=True)

            print(video_dir)
            video_path = os.path.join(video_dir, f"seed{args.seed}_fps_{args.fps}_ret{int(ep_return)}.mp4")
            imageio.mimsave(video_path, frames, fps=30,
                            codec="libx264", pixelformat="yuv420p",
                            macro_block_size=1)
            print(f"  saved {video_path}")

    print(f"\nMean return: {np.mean(returns):.1f} ± {np.std(returns):.1f}  "
          f"over {args.episodes} episodes")
    print(f"Mean length: {np.mean(lengths):.0f}")


if __name__ == "__main__":
    main()