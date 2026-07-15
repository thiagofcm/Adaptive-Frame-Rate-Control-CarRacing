import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import error, spaces
from gymnasium.envs.box2d.car_racing import FPS
import numpy as np
import hashlib

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
    def __init__(self, model_path, n_actions=5, device="cpu"):
        self.device = device
        checkpoint = torch.load(model_path, map_location=device)
        sd = checkpoint.get("agent_state_dict", checkpoint)
        #print(list(sd.keys())[:6])
        self.agent = NavAgentCNN(n_actions).to(device)

        if "model_state_dict" in checkpoint:
            self.agent.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.agent.load_state_dict(checkpoint)
        self.agent.eval()
    
    def predict(self, obs, deterministic=True):
        # obs is the raw (4,96,96) image stack
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits = self.agent.actor(self.agent.network(obs_t / 255.0))
            action = logits.argmax(dim=-1)
        return action.cpu().numpy()[0], None

class HighestFPSWrapper(gym.Wrapper):
    def __init__(self, env, nav_model_path, device, frame_cost=0.0, budget=50):
        super().__init__(env)

        # Variable Framerate Settings:
        self.simulation_fps= FPS
        self.frame_cost = frame_cost
        self.budget = budget
        self.fps_choices = [1,5,10,25,50]
        self.action_space = spaces.Discrete(len(self.fps_choices))
        self.device= device

        # Observation space adaptations
        self.image_shape = env.observation_space.shape              # (4, 96, 96)
        self.n_image     = int(np.prod(self.image_shape))           # 36864
        self.n_scalars   = 2
        self.observation_space = env.observation_space 
        # Navigation Controller
        if nav_model_path is not None:
            self.navigation_model = NavModel(nav_model_path)
        else:
            self.navigation_model = None 
        self.navigation_action_space = spaces.Discrete(4)

        # Variable Framerate Variables:
        self.world_step_count = 0
        self.steps_since_last_obs = 0
        self.current_fps = self.simulation_fps
        self.obs_interval = 1
        self.fps_penalty = 0.0
        self.current_obs = None
        self.last_sampled_obs = None

    def reset(self, *, seed=None, options=None):
        observed_frame, info = self.env.reset(seed=seed, options=options)
        track = np.asarray(self.env.unwrapped.track, dtype=np.float32)
        #track_hash = hash(track.tobytes())
        #track_hash =  hashlib.md5(track.tobytes()).hexdigest()
        #print(f"[reset] seed={seed}  track_hash={track_hash}  track_len={len(track)}")

        self.world_step_count = 0
        self.steps_since_last_obs = 0
        self.episode_frame_count = 1
        self.fps_penalty = 0.0
        self.current_fps = self.fps_choices[-1]          # start at fastest (50)
        self.obs_interval = int(self.simulation_fps / self.current_fps)
        self.mask_buffer = []                            # step appends to this
        self.reached_goal = False
        self.budget_pen_check = False

        # # Goal Settings:
        # self.goal_frac   = 0.95
        # self.goal_xy     = track[int(self.goal_frac * len(track)), 2:4]
        # self.goal_radius = 20.0
        # self.min_steps   = 20

        self.last_sampled_obs = observed_frame                      # raw (4,96,96) image for the controller
        self.current_obs = self.last_sampled_obs  # augmented (36866,) for the FPS policy
        return self.current_obs, info
  
    def _get_augmented_obs(self, observed_frame):
        flat_img = observed_frame.astype(np.float32).reshape(-1)  
        obs_age_steps = self.steps_since_last_obs
        # normalizing the num of steps since last obs
        obs_age_ratio = obs_age_steps / self.simulation_fps
        # normalizing the num of steps since last obs
        fps_ratio = self.current_fps/self.simulation_fps

        aug_obs = np.concatenate([
            flat_img.astype(np.float32),
            np.array([obs_age_ratio,fps_ratio], dtype=np.float32)])

        return aug_obs

    def _get_debug_obs(self, frame_consumed):
        obs_age_ratio = self.steps_since_last_obs / self.simulation_fps
        fps_ratio     = self.current_fps / self.simulation_fps
        return np.array([float(frame_consumed), obs_age_ratio, fps_ratio], dtype=np.float32)

    def step(self, fps_action):
        assert self.navigation_model is not None, \
            "navigation_model is None — did you forget to inject it?"
        
        # Increment the world step count
        self.world_step_count += 1
        #print("FRAME COST: ", self.frame_cost)

        # Increment the steps since last observation count
        self.steps_since_last_obs += 1

        # 1. Use the currently available sampled observation to compute navigation action
        navigation_action, _ = self.navigation_model.predict(self.last_sampled_obs, deterministic=True)

        # 2. Perform a physics step in the environment using the navigation action,
        # and get the new observation, navigation reward, termination status, truncation status, and info
        observed_frame, nav_reward, terminated, truncated, info = self.env.step(navigation_action)

        # 3. Testing reaching goal condition
        # x, y = self.env.unwrapped.car.hull.position
        # self.reached_goal = (np.hypot(x - self.goal_xy[0], y - self.goal_xy[1]) < self.goal_radius) \
        #            and (self.world_step_count > self.min_steps)

        # 4. Check if it's time to sample a new observation based on the obs_interval
        # If so, update the last sampled observation, reset the steps since last observation count, and increment the episode frame count
        if self.steps_since_last_obs >= self.obs_interval:
            # self.current_obs is updated every physics step, so here we store the fresh observation of this step
            self.last_sampled_obs = observed_frame.copy()
            self.steps_since_last_obs = 0
            self.episode_frame_count += 1
            frame_consumed = True

            # Update FPS and obs_interval based on the action taken by the agent
            # the action is chosen at a sampling instant and affects future sampling
            self.current_fps = self.fps_choices[int(fps_action)]
            self.obs_interval = int(self.simulation_fps / self.current_fps)
            
            # Debbuging mask, which indicates which values in the observation are valid (1 for valid, 0 for invalid)
            #obs_mask = np.ones_like(self.last_sampled_obs, dtype=np.float32)

        else:
            # If it's not time to sample a new observation, we use the last sampled
            # observation (self.last_sampled_obs is not updated)
            frame_consumed = False
            # Debbuging mask, which indicates which values in the observation are valid (1 for valid, 0 for invalid)
            #obs_mask = np.zeros_like(self.last_sampled_obs, dtype=np.float32)

        # 5. Get current obs
        self.current_obs = self.last_sampled_obs

        # DEBBUGING: store the observation mask in a separate buffer for analysis
        #img_sum = float(self.current_obs[:self.n_image].sum())
        #debug_obs = self._get_debug_obs(frame_consumed)
        #debug_obs = self._get_debug_obs(img_sum)
        #print(f"[Adaptive FPS Wrapper] Step: {self.world_step_count}, Obs: {debug_obs}, Action: {self.current_fps}")

        reward = nav_reward #+ 0.05 * (self.current_fps/self.simulation_fps)
        # print("[Adaptive FPS Wrapper] Step: {}, Reward: {}, Nav Reward: {}, Current FPS: {}, Frame Cost: {}, Budget: {}".format(
        #     self.world_step_count, reward, nav_reward, self.current_fps, self.frame_cost, self.budget))
        #print(f"Nav Rew on Highest FPS Wrapper: {reward}")
        
        # if self.reached_goal:
        #     terminated = True
        #     #reward = 150

        info = dict(info)
        info["reward"] = reward
        info["nav_reward"] = nav_reward
        info["frame_cost"] = self.frame_cost
        info["budget"] = self.budget
        info["chosen_fps"] = self.current_fps
        info["episode_frame_count"] = self.episode_frame_count
        info["timeout"] = truncated and not terminated
        info["reached_goal"] = self.reached_goal

        return self.current_obs, reward, terminated, truncated, info