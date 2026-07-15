import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import error, spaces
from gymnasium.envs.box2d.car_racing import FPS
import numpy as np
import hashlib
from utils.cautious_variables import CautiousVars
import time

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
        self.agent.load_state_dict(sd)                          # use sd, not checkpoint
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

class HighestFPS_Cautious_SL_Wrapper(gym.Wrapper):
    def __init__(self, env, nav_model_path, device):
        super().__init__(env)

        # Variable Framerate Settings:
        self.simulation_fps= FPS
        self.fps_choices = [1,5,10,25,50]
        self.action_space = spaces.Discrete(len(self.fps_choices))
        self.gamma = 0.99

        # Navigation Controller
        if nav_model_path is not None:
            self.navigation_model = NavModel(nav_model_path, device)
        else:
            self.navigation_model = None 
        self.navigation_action_space = spaces.Discrete(4)

        # Cautious Variables Class
        self.cautious_sensors = CautiousVars()
        self.n_cautious = 8
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_cautious,), dtype=np.float32) 

        # Highest Framerate Variables:
        self.world_step_count = 0
        self.current_fps = self.simulation_fps
        self.obs_interval = 1
        self.current_obs = None
        self.last_sampled_obs = None
        self.max_physics_steps = 500

    def reset(self, *, seed=None, options=None):
        observed_frame, info = self.env.reset(seed=seed, options=options)
        track = np.asarray(self.env.unwrapped.track, dtype=np.float32)

        #track_hash = hash(track.tobytes())
        track_hash =  hashlib.md5(track.tobytes()).hexdigest()
        #print(f"[reset] seed={seed}  track_hash={track_hash}  track_len={len(track)}")

        self.world_step_count = 0
        self.episode_frame_count = 1
        self.current_fps = self.fps_choices[-1]          # start at fastest (50)
        self.obs_interval = int(self.simulation_fps / self.current_fps)
        self.mask_buffer = []                            # step appends to this
        self.reached_goal = False

        # Goal Settings:
        self.goal_frac   = 0.95
        self.goal_xy     = track[int(self.goal_frac * len(track)), 2:4]
        self.goal_radius = 20.0
        self.min_steps   = 20

        # Frames
        self.last_sampled_obs = observed_frame                      # raw (4,96,96) image for the controller
        self.current_obs = self.last_sampled_obs  # augmented (36866,) for the FPS policy

        # Cautious Variables
        self.cautious_sensors.reset_track_reading(self.env)
        self.last_sampled_cautious_obs = self.cautious_sensors.get_cautious_var(self.env)
        self.cautious_obs = self.last_sampled_cautious_obs

        return self.cautious_obs, info

    def step(self, fps_action):
        assert self.navigation_model is not None, \
            "navigation_model is None — did you forget to inject it?"

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

            # Nav controller always acts on the last *sampled* frame — held fixed for the whole
            # window, this is what creates the staleness effect
            navigation_action, _ = self.navigation_model.predict(self.last_sampled_obs, deterministic=True)
            observed_frame, nav_reward, terminated, truncated, info = self.env.step(navigation_action)

            self.current_obs = observed_frame
            total_reward += discount * nav_reward
            discount *= self.gamma
            ticks_in_window += 1

            # Goal check — every physics tick, not just at window boundaries
            x, y = self.env.unwrapped.car.hull.position
            self.reached_goal = (np.hypot(x - self.goal_xy[0], y - self.goal_xy[1]) < self.goal_radius) \
                    and (self.world_step_count > self.min_steps)
            if self.reached_goal:
                terminated = True
            
            if self.world_step_count >= self.max_physics_steps:
                truncated = True
            
            #print(f"Step: {self.world_step_count}, Nav Action: {navigation_action}, "
            #f"FPS Action: {fps_action}, Current FPS: {self.current_fps}, "
            #f"Obs Interval: {self.obs_interval}, Episode Frame Count: {self.episode_frame_count}, ")


            if terminated or truncated:
                break

        # Window over: this is the new sampled observation — the one real decision point
        self.last_sampled_obs = self.current_obs.copy()
        self.episode_frame_count += 1

        self.last_sampled_cautious_obs = self.cautious_sensors.get_cautious_var(self.env)
        self.cautious_obs = self.last_sampled_cautious_obs
        
        info = dict(info)
        info["reward"] = total_reward
        info["nav_reward"] = total_reward
        info["chosen_fps"] = self.current_fps
        info["episode_frame_count"] = self.episode_frame_count
        info["window_duration"] = ticks_in_window
        info["physics_steps"] = self.world_step_count
        info["timeout"] = truncated and not terminated
        info["reached_goal"] = self.reached_goal

        return self.cautious_obs, total_reward, terminated, truncated, info