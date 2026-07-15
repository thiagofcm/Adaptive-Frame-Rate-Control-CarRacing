import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

def image_preprocessing(img):
    img = cv2.resize(img, dsize=(96, 96))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

class CarRacingPreprocessing(gym.Wrapper):
    def __init__(self, env, skip_frames=4, stack_frames=4, no_operation=50, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames
        self.observation_space = gym.spaces.Box(low=0.0, high=255.0, shape=(stack_frames, 96, 96), dtype=np.float32)
        self.step_counter = 0

    def reset(self, *, seed=None, options=None):
        # For using always the same track
        seed = 1
        observation, info = self.env.reset(seed=seed, options=options)
        self.step_counter+=1

        for _ in range(self._no_operation):
            observation, reward, terminated, truncated, info = self.env.step(0)
            
        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info

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
        #print(f"Nav Rew on Pre Processing: {total_reward}")
        return self.stack_state, total_reward, terminated, truncated, info