"""
cautious_variables.py

Extract low-dimensional "cautious variables" from CarRacing-v3 that signal when
the frozen navigation controller is at risk -- so the FPS policy can learn *when*
to spend perception, without looking at the image itself.

Design principle:
  Track geometry (heading, curvature, arclength) is STATIC for an episode.
  -> Precompute it once in reset_track() right after env.reset().
  -> Per step, only car-relative quantities update (nearest node, cross-track
     error, off-track flag, speed, distance/time to next curve). This is cheap.

All reaches go through env.unwrapped (you are wrapped deep). The three attributes
to check on version drift:
    env.unwrapped.car.hull.{position, angle, linearVelocity}
    env.unwrapped.car.wheels[i].tiles          # set of road tiles in contact
    env.unwrapped.track  ->  list of (alpha, beta, x, y)   # centerline nodes

NOTE: all distances are in CarRacing WORLD units (not meters). The thresholds
below are starting guesses -- calibrate them by logging curv / speed over one
frozen-controller rollout before trusting them (see __main__).
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from wrappers.pre_processing import CarRacingPreprocessing, image_preprocessing
import argparse
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")          # headless cluster: no display, must set before pyplot
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from PIL import Image, ImageDraw, ImageFont
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH

SPEED_REF = 100.0     # ~ measured top speed of the controller
TIME_REF  = 10.0      # seconds; time_to_curve horizon you care about

SIM_FPS = 50

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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

def make_eval_env(env_id, seed):
    env = gym.make(env_id, continuous=False, render_mode="rgb_array", lap_complete_percent=0.95, max_episode_steps=4000)
    env = CarRacingPreprocessing(env, skip_frames=4, stack_frames=4)
    env.reset(seed=seed)
    return env

def draw_debug_text(frame, debug_dict, step_counter):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # text_lines = [ 
    #     f"Frame: {step_counter}",
    #     f"curv_here: {debug_dict['curv_here']:.3f}",
    #     f"cross_track: {debug_dict['cross_track']:.3f}",
    #     f"off_track: {debug_dict['off_track']}",
    #     f"dist_curve: {debug_dict['dist_to_curve']:.3f}",
    #     f"speed: {debug_dict['speed']:.3f}",
    #     f"time_to_curve: {debug_dict['time_to_curve']:.3f}",
    # ]

    text_lines = [
        f"Frame: {step_counter}",
        f"x: {debug_dict['x']:.3f}",
        f"y: {debug_dict['y']:.3f}",
        f"vx: {debug_dict['vx']:.3f}",
        f"vy: {debug_dict['vy']:.3f}",
        f"dist_to_curve: {debug_dict['dist_to_curve']:.3f}",
        f"time_to_curve: {debug_dict['time_to_curve']:.3f}",
        f"cross_track: {debug_dict['cross_track']:.3f}",
        f"off_track: {debug_dict['off_track']}",
    ]

    x, y = 10, 10
    line_height = 18

    # black background box
    box_w = 230
    box_h = line_height * len(text_lines) + 10
    draw.rectangle([x - 5, y - 5, x + box_w, y + box_h], fill=(0, 0, 0))

    for i, line in enumerate(text_lines):
        draw.text((x, y + i * line_height), line, fill=(255, 255, 255))

    return np.array(img)

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None, deterministic=False):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = logits.argmax(dim=-1) if deterministic else probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

class CautiousVars:
    def __init__(self, curve_thresh=0.04, lookahead=300.0, search_window=25):
        # curve_thresh : |curvature| (rad/world-unit) above which a node = "a curve"
        # lookahead    : how far ahead to scan for the next curve (caps dist_to_curve)
        # search_window: +/- nodes around last index to search for the nearest node
        self.curve_thresh = curve_thresh
        self.lookahead = lookahead
        self.search_window = search_window
        self.last_idx = 0

    def reset_track_reading(self, env):
        """
        @function: Read the map and initiliaze the cautious variables.
        This function is called every reset function after env.reset()
        Each track has mutiple nodes. Each node has the form: (alpha, beta, x, y).
        where alpha is the polar angle of the node around the map center,
        beta is the normal direction of the node.
        x and y are positions.
        """
        # Get x,y position list from the whole track (env.unwrapped.track).
        xy = np.asarray([(x, y) for (_a, _b, x, y) in env.unwrapped.track], dtype=np.float64)
        # Shifts every row of xy table upward [A,B,C,D] becomes [B,C,D,A]
        nxt = np.roll(xy, -1, axis=0)
        # Shifts every row of xy table downward [A,B,C,D] becomes [D,A,B,C]
        prv = np.roll(xy,  1, axis=0)

        # Get Per-node tangent (tang) vector, heading via central difference of positions
        tang = nxt - prv
        # By computing this arctang between the values in the vector tang, you can obtaint
        # a vector representing the heading of the track, what direction it is pointed to.
        heading = np.arctan2(tang[:, 1], tang[:, 0])

        # Get the vector that represents the segment that goes from current xy to next xy
        seg = nxt - xy
        # Get the length of this sequence by computing Eucliden Distance
        seg_len = np.maximum(np.linalg.norm(seg, axis=1), 1e-6)

        # local turn per segment, wrapped to (-pi, pi] (handles the loop seam)
        # Computes the change in road direction from one node to the next.
        dh = np.diff(heading, append=heading[0])
        # This line wraps it back into the range: -pi, pi
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        # The curvature of the tail 'curv' is the rate o track
        # normal divided by the size of the segment lenght
        curv = dh / seg_len                                     # rad / world-uniti fi
        # Curv could be interpreted as the signature of the curve, with direction (+,-):
        # Straight        :  0
        # Gentle left     : +0.03
        # Sharp left      : +0.12
        # Gentle right    : -0.03
        # Sharp right     : -0.12
        # Turn is just the magnitude of the curve. Absolute value of the curvature
        turn = np.abs(dh)

        self.xy = xy
        self.heading = heading
        self.curv = curv
        self.turn = turn
        self.seg_len = seg_len
        self.last_idx = 0
        return self

    def get_cautious_var(self, env):
        # Car object
        car = env.unwrapped.car
        # Position x and y of the car
        p = np.array([car.hull.position[0], car.hull.position[1]])
        # Velocity x and y of the car
        v = np.array([car.hull.linearVelocity[0], car.hull.linearVelocity[1]])
        # Speed of the car from the velocities
        speed = float(np.linalg.norm(v))

        # Get the index of the closest tail.
        closest_idx = self._nearest_idx(p)

        # ------------------- GETTING CROSS TRACK/DISTANCE -------------------------
        # Heading (direction) of the track at the closest centerline node.
        track_heading = self.heading[closest_idx]
        # Unit normal vector to the track (90° counterclockwise from the heading).
        # This vector points laterally across the track.
        track_normal = np.array([-np.sin(track_heading),np.cos(track_heading)])
        # Signed lateral distance from the car to the track centerline.
        # It is obtained by computing the diff between current position (p)
        # and xy of the node of closest track. Then the dot product between
        # this difference and the normal vector of the track is computed.
        # The Track normal is computed because 
        # Positive and negative values indicate opposite sides of the centerline.
        cross_track = float((p - self.xy[closest_idx]) @ track_normal)

        # DEBUG
        # print("p:", p)
        # print("self.xy[i]:", self.xy[i])
        # print("p - self.xy[i]:", p - self.xy[i])
        # print("n_hat:", track_normal)
        # print("dot:", (p - self.xy[i]) @ track_normal)
        # print("------------------------------------------------------------------")

        # ------------------- GETTING OFF TRACK WHEEL FLAGS -------------------------
        # Value is between 0 and 1. 0.25 means 1 wheel off track.
        # 0.0 means all the wheels on the track
        off_track = sum(len(w.tiles) == 0 for w in car.wheels) / 4.0

        # ------------------- GETTING DISTANCE TO NEXT CURVE -------------------------
        N = len(self.xy)
        dist_to_curve = self.lookahead
        turning_ahead = 0.0
        found = False
        acc = 0.0
        j = closest_idx
        while acc < self.lookahead:
            turning_ahead += self.turn[j]
            acc += self.seg_len[j]
            j = (j + 1) % N
            if not found and abs(self.curv[j]) > self.curve_thresh:
                dist_to_curve = acc
                found = True
            if j == closest_idx:
                break

        # ------------------- GETTING DISTANCE TO NEXT CURVE -------------------------
        time_to_curve = dist_to_curve / max(speed, 1e-3)

        # return {
        #     "speed":         speed,           # world-units / s
        #     "cross_track":   cross_track,     # signed lateral offset
        #     "off_track":     float(off_track),# 1.0 if all four wheels off road
        #     "dist_to_curve": dist_to_curve,   # to next curve (capped at lookahead)
        #     "time_to_curve": time_to_curve,   # dist_to_curve / speed
        #     "turning_ahead": turning_ahead,   # integral of |curvature| over lookahead
        #     "curv_here":     float(self.curv[closest_idx]),
        # }

        return np.array([
            p[0],            # x
            p[1],            # y
            v[0]/SPEED_REF,  # vx
            v[1]/SPEED_REF,  # vy
            np.clip(dist_to_curve / self.lookahead, 0, 1), # dist_to_curve -> [0,1],
            time_to_curve,
            np.clip(cross_track / TRACK_WIDTH, -2, 2) / 2,
            off_track,
        ], dtype=np.float32)

    def _nearest_idx(self, p):
        """This function searches in a local window around the last
        index the closest track."""

        # N is the total number of nodes in the track.
        N = len(self.xy)
        # Window for searching the closest node.
        w = self.search_window
        # Search only within a local window around the previous closest node.
        # '%' wraps indices across the beginning/end of the closed-loop track.
        idxs = (self.last_idx + np.arange(-w, w + 1)) % N

        # Among the candidates, get the dx and dy from the current point
        dx = self.xy[idxs, 0] - p[0]
        dy = self.xy[idxs, 1] - p[1]

        # Squared Euclidean distance to each candidate centerline node.
        # The square root is omitted since it does not change which node is closest.
        d2 = dx**2 + dy**2
        # Get the closest track index among the candidates
        self.last_idx = int(idxs[np.argmin(d2)])
        return self.last_idx

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
    
    # ?
    class _Dummy:
        single_action_space = gym.spaces.Discrete(5)
    agent = Agent(_Dummy()).to(device)
    
    # Load Checkpoint
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "agent_state_dict" in ckpt:
        agent.load_state_dict(ckpt["agent_state_dict"])
        print(f"Loaded checkpoint from iteration {ckpt.get('iteration', '?')}, "
              f"global_step {ckpt.get('global_step', '?')}")
    else:
        agent.load_state_dict(ckpt)  # plain state_dict (e.g. final.pt)
        print(f"Loaded raw state_dict from {args.ckpt}")
    agent.eval()

    env = make_eval_env(args.env_id, args.seed)
    cv = CautiousVars()
    obs, _ = env.reset(seed=0)
    cv.reset_track_reading(env)
    frames, ep_return, ep_len, done = [], 0.0, 0, False
    obs_interval = SIM_FPS/args.fps
    last_sampled_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    positions, speeds, log  = [], [], []
    returns, lengths = [], []
    track = np.asarray(env.unwrapped.track, dtype=np.float32) 
    goal_frac   = 0.95
    goal_idx    = int(goal_frac * len(track))
    goal_xy     = track[goal_idx, 2:4]
    goal_radius = 20.0
    min_steps   = 20
    total_frame_counter = 0

    while not done:
        x, y = env.unwrapped.car.hull.position
        vx, vy = env.unwrapped.car.hull.linearVelocity

        reached = (np.hypot(x - goal_xy[0], y - goal_xy[1]) < goal_radius) and (ep_len > min_steps)

        positions.append((x, y))
        speeds.append(float(np.hypot(vx, vy)))

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

        obs, r, term, trunc, _ = env.step(action.item())
        c = cv.get_cautious_var(env)
        x_n, y_n, vx_n, vy_n, dist_n, time_n, cross_n, off_n = c

        log.append((x_n, y_n, vx_n, vy_n, dist_n, time_n, cross_n, off_n))

        print(f" Step: {env.step_counter} - {obs_type} Observation - "
              f"Frames Consumed: {total_frame_counter} - X: {x}, Y: {y}, Action:{action}")
        print(f"x: {x_n}, y: {y_n}, vx: {vx_n}, vy: {vy_n}, "
              f"dist_to_curve: {dist_n}, time_to_curve: {time_n}, "
              f"cross_track: {cross_n}, off_track: {off_n}")
        
        if args.save_video:
            frames.append(env.unwrapped.render())   # raw 96x96 RGB

        ep_return += r
        ep_len += 1
        done = term or trunc or reached
        
        # print("curvature   range:", log[:, 0].min(), log[:, 0].max())
        # print("off-track   frac :", log[:, 1].mean())
        # print("dist_to_curve mean:", log[:, 2].mean())
        # print("speed       range:", log[:, 3].min(), log[:, 3].max())

    print(f"Episode completed consuming {total_frame_counter} frames")
    log = np.array(log)

    plot_track_trajectory(
        env, positions, speeds,
        save_path=os.path.join(args.video_dir, f"traj_seed{args.seed}_fps{args.fps}_ret{int(ep_return)}_goal.png"),
        title=f"fps={args.fps}  return={ep_return:.0f}  len={ep_len}",
    )

    env.close()
    returns.append(ep_return)
    lengths.append(ep_len)
    print(f"return={ep_return:7.1f}  length={ep_len}")

    if args.save_video and frames:

        annotated_frames = []

        for idx, (frame, vals) in enumerate(zip(frames, log)):
            x_n, y_n, vx_n, vy_n, dist_n, time_n, cross_n, off_n = vals
            annotated_frames.append(
                draw_debug_text(
                    frame,
                    {
                        "x": x_n,
                        "y": y_n,
                        "vx": vx_n,
                        "vy": vy_n,
                        "dist_to_curve": dist_n,
                        "time_to_curve": time_n,
                        "cross_track": cross_n,
                        "off_track": off_n
                    },
                    idx+1
                )
            )

        run_string  = args.ckpt.split("/")[-2]
        ckpt_string = args.ckpt.split("/")[-1].split(".")[0]
        video_dir = os.path.join("eval_videos", run_string, ckpt_string)
        os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(
            video_dir,
            f"cautious_var_seed{args.seed}_fps_{args.fps}_ret{int(ep_return)}.mp4",
        )

        imageio.mimsave(
            video_path,
            annotated_frames,
            fps=30,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=1,
        )

    print(f"\nMean return: {np.mean(returns):.1f} ± {np.std(returns):.1f}  "
            f"over {args.episodes} episodes")
    print(f"Mean length: {np.mean(lengths):.0f}")

if __name__ == "__main__":
    main()