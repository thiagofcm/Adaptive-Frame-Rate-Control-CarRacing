import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import spaces
from gymnasium.envs.box2d.car_racing import FPS
import numpy as np
from utils.cautious_variables import CautiousVars

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class NavAgentCNN(nn.Module):
    def __init__(self, n_actions=5):
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
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

class NavModel:
    """Frozen CarRacing nav controller (CleanRL CNN agent) with a predict() interface."""
    def __init__(self, model_path, device, n_actions=5):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        sd = checkpoint.get("agent_state_dict", checkpoint)   # handles both formats
        self.agent = NavAgentCNN(n_actions).to(self.device)
        self.agent.load_state_dict(sd)
        self.agent.eval()
        print(f"Navigation Model loaded on {device}")

    def predict(self, obs, deterministic=True):
        # obs is the raw (4,96,96) image stack
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.agent.actor(self.agent.network(obs_t / 255.0))
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.distributions.Categorical(logits=logits).sample()
        return action.cpu().numpy()[0], None


class AdaptiveFPS_Cautious_SMDP_Wrapper(gym.Wrapper):
    """
    SMDP-formulated adaptive-FPS wrapper: one wrapper step() = one full decision
    window. The FPS action chosen at the start of a window sets obs_interval (the
    window length in physics ticks); NavModel drives the whole window on the
    observation sampled at the window's start -- held fixed and refreshed only
    once the window ends -- and in-window reward is accumulated with per-tick
    discounting so it can be credited as a single value to the FPS decision that
    produced it. `info["window_duration"]` reports how many physics ticks the
    window actually ran (<=obs_interval, shorter on early termination), for
    gamma**duration cross-window GAE bootstrapping in the training loop.

    Same CautiousVars pipeline (11-dim + 3 augmented) as AdaptiveFPS_TrackAware_
    Wrapper (the per-tick sibling in adaptive_fps_track_aware_wrapper.py), and
    relies on CarRacing_VarFramerate (envs/car_racing_var_fps.py) for the
    off-track penalty / curve-passed bonus / wrong-direction / stalled / off-
    track-timeout termination shaping already baked into nav_reward -- frame_cost
    and the budget-overrun penalty are the only reward terms this wrapper adds
    itself, both charged once per decision/window. Unlike the per-tick wrapper,
    there's no separate wrapper-level goal-distance check here: CarRacing_
    VarFramerate's own lap_finished/tile-progress logic already covers "reached
    the end of the track," so this wrapper doesn't duplicate it.
    """
    def __init__(self, env, nav_model_path, device, frame_cost=0.0, budget=50,
                 gamma=0.99, max_physics_steps=1000):
        super().__init__(env)

        # Variable Framerate Settings:
        self.simulation_fps = FPS
        self.frame_cost = frame_cost
        self.budget = budget
        self.fps_choices = [1, 5, 10, 25, 50]
        self.action_space = spaces.Discrete(len(self.fps_choices))
        self.gamma = gamma
        self.max_physics_steps = max_physics_steps

        # Navigation Controller
        if nav_model_path is not None:
            self.navigation_model = NavModel(nav_model_path, device)
        else:
            self.navigation_model = None
        self.navigation_action_space = spaces.Discrete(4)

        # Cautious Variables Class -- same 11-dim layout as the per-tick wrapper:
        # vx, vy, dist_to_curve, curve_severity, heading_alignment, cross_track,
        # cross_track_rate, off_track, time_off_track, episode_completion, curves_passed.
        # No raw (x, y), no time_to_curve, no frame_counter (that's in the augmented
        # block below, same as the per-tick wrapper).
        self.cautious_sensors = CautiousVars()
        self.n_cautious = 11
        self.n_scalars = 3  # window_duration_ratio, fps_ratio, frame_counter
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.n_cautious + self.n_scalars,), dtype=np.float32
        )

        # Variable Framerate Variables:
        self.world_step_count = 0
        self.current_fps = self.simulation_fps
        self.obs_interval = 1
        self.current_obs = None
        self.last_sampled_obs = None

    def reset(self, *, seed=None, options=None):
        observed_frame, info = self.env.reset(seed=seed, options=options)

        self.world_step_count = 0
        self.episode_frame_count = 1
        self.current_fps = self.fps_choices[-1]          # start at fastest (50)
        self.obs_interval = int(self.simulation_fps / self.current_fps)
        self.budget_pen_check = False

        # NavModel always predicts off last_sampled_obs, held fixed for the entire
        # window -- only refreshed at a window boundary (see step()).
        self.last_sampled_obs = observed_frame
        self.current_obs = self.last_sampled_obs

        # Cautious Variables
        self.cautious_sensors.reset_track_reading(self.env)
        # dt_ticks=0: first reading of the episode, no prior tick to diff/accumulate against
        self.last_sampled_cautious_obs = self.cautious_sensors.get_cautious_var(self.env, dt_ticks=0)

        # ticks_in_window is 0 here -- no window has executed yet at reset
        self.cautious_obs = self._get_augmented_obs(self.last_sampled_cautious_obs, ticks_in_window=0)
        return self.cautious_obs, info

    def _get_augmented_obs(self, cautious_obs, ticks_in_window):
        # window_duration_ratio reflects how long the *just-completed* decision window
        # was -- the FPS policy is only ever queried at window boundaries, so this plays
        # the same role obs_age_ratio plays in the per-tick wrapper.
        window_duration_ratio = ticks_in_window / self.simulation_fps
        fps_ratio = self.current_fps / self.simulation_fps
        # how many decisions spent out of budget -- clean 0->1 ramp that hits 1.0
        # exactly when the budget-overrun penalty (below) is about to fire
        frame_counter = np.clip(self.episode_frame_count / self.budget, 0, 1)
        return np.concatenate([
            cautious_obs,
            np.array([window_duration_ratio, fps_ratio, frame_counter], dtype=np.float32),
        ])

    def step(self, fps_action):
        assert self.navigation_model is not None, \
            "navigation_model is None — did you forget to inject it?"

        # This window's decision: apply the chosen FPS for the whole window about to run
        self.current_fps = self.fps_choices[int(fps_action)]
        self.obs_interval = int(self.simulation_fps / self.current_fps)

        total_reward = 0.0
        discount = 1.0
        ticks_in_window = 0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self.obs_interval):
            self.world_step_count += 1

            # NavModel drives on the frame sampled at the START of this window, held
            # fixed for the whole window -- do NOT refresh last_sampled_obs in here,
            # that would silently defeat the point of choosing a sampling rate.
            navigation_action, _ = self.navigation_model.predict(self.last_sampled_obs, deterministic=True)
            observed_frame, nav_reward, terminated, truncated, info = self.env.step(navigation_action)

            self.current_obs = observed_frame
            total_reward += discount * nav_reward
            discount *= self.gamma
            ticks_in_window += 1

            if self.world_step_count >= self.max_physics_steps:
                truncated = True

            if terminated or truncated:
                break

        assert 1 <= ticks_in_window <= self.obs_interval, \
            f"window_duration {ticks_in_window} inconsistent with obs_interval {self.obs_interval}"

        # Window over: this is the new sampled observation -- the one real decision
        # point. Refreshing here (not mid-loop) is what keeps NavModel's input stale
        # for the duration of the window that just ran.
        self.last_sampled_obs = self.current_obs.copy()
        self.episode_frame_count += 1
        self.last_sampled_cautious_obs = self.cautious_sensors.get_cautious_var(self.env, dt_ticks=ticks_in_window)

        # Sampling cost — charged once per decision/window. Because a higher chosen
        # FPS means shorter windows (more decisions over a fixed episode length), a
        # flat per-window charge already penalizes higher sampling rates more.
        total_reward -= self.frame_cost

        if self.episode_frame_count > self.budget and not self.budget_pen_check:
            total_reward += -100
            self.budget_pen_check = True

        self.cautious_obs = self._get_augmented_obs(self.last_sampled_cautious_obs, ticks_in_window)

        # Consolidated debug field, same convention as AdaptiveFPS_TrackAware_Wrapper
        # (minus reached_goal, which this wrapper doesn't track separately -- see class
        # docstring). "off_playfield" is the implicit case: terminated=True but none of
        # the env's other named flags fired.
        if info.get("lap_finished"):
            termination_reason = "lap_finished"
        elif info.get("wrong_direction"):
            termination_reason = "wrong_direction"
        elif info.get("off_track_timeout"):
            termination_reason = "off_track_timeout"
        elif info.get("stalled"):
            termination_reason = "stalled"
        elif terminated:
            termination_reason = "off_playfield"
        elif truncated:
            termination_reason = "timeout"
        else:
            termination_reason = "none"

        info = dict(info)
        info["reward"] = total_reward
        info["nav_reward"] = total_reward
        info["frame_cost"] = self.frame_cost
        info["budget"] = self.budget
        info["chosen_fps"] = self.current_fps
        info["episode_frame_count"] = self.episode_frame_count
        info["window_duration"] = ticks_in_window
        info["physics_steps"] = self.world_step_count
        info["timeout"] = truncated and not terminated
        info["termination_reason"] = termination_reason

        return self.cautious_obs, total_reward, terminated, truncated, info
