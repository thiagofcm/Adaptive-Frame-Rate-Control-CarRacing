import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.envs.box2d.car_racing import FPS

def image_preprocessing(img):
    img = cv2.resize(img, dsize=(96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

class CarRacing_Preprocessing_Scalars(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, no_operation=50, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames
        
        # Modifications to add scalars into the observation space 
        self.n_image   = self._stack_frames * 96 * 96      # = 4*96*96 = 36864
        self.n_scalars = 2
        
        # New observation space shape: (image + scalars)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=255.0,
            shape=(self.n_image + self.n_scalars,),
            dtype=np.float32,
        )

        self.step_counter = 0

    def _get_augmented_obs(self, observed_frame):
        flat_img = observed_frame.astype(np.float32).reshape(-1)  
        obs_age_steps = self.steps_since_last_obs
        # normalizing the num of steps since last obs
        obs_age_ratio = obs_age_steps / self.simulation_fps
        # normalizing the num of steps since last obs
        fps_ratio = self.current_fps/self.simulation_fps
        aug_obs = np.concatenate([
            flat_img,
            np.array([obs_age_ratio, fps_ratio], dtype=np.float32)
        ])     
        return aug_obs 

    def reset(self, *, seed=None, options=None):
        # For using always the same track
        observation, info = self.env.reset(seed=seed, options=options)
        self.current_fps = FPS
        self.simulation_fps = FPS
        self.steps_since_last_obs = 0
        self.step_counter+=1

        for _ in range(self._no_operation):
            observation, reward, terminated, truncated, info = self.env.step(0)
            
        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        self.current_obs = self._get_augmented_obs(self.stack_state)

        return self.current_obs, info

    def step(self, action):
        total_reward = 0
        self.step_counter+=1

        for i in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        observation = image_preprocessing(observation)
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        self.current_obs = self._get_augmented_obs(self.stack_state)

        # DEBUG:
        img = self.current_obs[:-2]
        # print(f"Step: {self.step_counter}, age: {self.steps_since_last_obs}, FPS: {self.current_fps}")
        # print(f"Img sum: {img.sum():.1f}, mean: {img.mean():.3f}, Scalars: {self.current_obs[-2:]}, Shape: {self.current_obs.shape}")
        # print("----------------------------------------------------------------")
        return self.current_obs, total_reward, terminated, truncated, info