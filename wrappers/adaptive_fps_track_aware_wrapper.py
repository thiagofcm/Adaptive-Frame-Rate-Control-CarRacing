import gymnasium as gym
import torch
import torch.nn as nn
from gymnasium import error, spaces
from gymnasium.envs.box2d.car_racing import FPS
import numpy as np
import hashlib
import math
from utils.cautious_variables import CautiousVars


def _arc_length_point(xy, seg_len, target_arc):
    """Walk the closed-loop centerline `xy` (N,2) and return the point at cumulative
    Euclidean arc length `target_arc` from xy[0], interpolated within whichever segment
    it falls in. `seg_len[i]` must be the distance from xy[i] to xy[(i+1) % N].

    Every node in `xy` is already a valid drivable centerline tile -- this fork's track
    generator (envs/car_racing_var_fps.py:_create_track()) only ever appends road-tile
    centerline points to self.track; there is no separate grass/connector/border tile
    stored there to filter out (the red/white curb "border" tiles are a rendering-only
    overlay on the SAME road tiles, never persisted to self.track). So no filtering step
    is needed before walking it.

    Assumes target_arc is already reduced into [0, seg_len.sum()) by the caller -- see
    the multi-lap wraparound handling in AdaptiveFPS_TrackAware_Wrapper.reset().
    """
    acc = 0.0
    n = len(xy)
    for i in range(n):
        if acc + seg_len[i] >= target_arc:
            frac = (target_arc - acc) / seg_len[i] if seg_len[i] > 1e-9 else 0.0
            nxt = xy[(i + 1) % n]
            return xy[i] + frac * (nxt - xy[i])
        acc += seg_len[i]
    return xy[-1]  # floating-point fallback, shouldn't be reached


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
    
class AdaptiveFPS_TrackAware_Wrapper(gym.Wrapper):
    def __init__(self, env, nav_model_path, device, frame_cost=0.0, budget=50, goal_distance=900.0):
        super().__init__(env)

        # Variable Framerate Settings:
        self.simulation_fps= FPS
        self.frame_cost = frame_cost
        self.budget = budget
        # Fixed absolute arc-length distance (world units, same scale as track x/y --
        # NOT a fraction of track point count) from the start line to the goal point.
        # Calibrated against 20 real generated tracks: total centerline arc length
        # mean=1040, min=925, max=1243 world units -- 900 keeps most tracks at a single
        # lap (matching the old goal_frac=0.95 behavior in typical cases) while still
        # regularly exercising the multi-lap wraparound below on shorter tracks.
        self.goal_distance = goal_distance
        self.fps_choices = [1,5,10,25,50]
        self.action_space = spaces.Discrete(len(self.fps_choices))

        # Navigation Controller
        if nav_model_path is not None:
            self.navigation_model = NavModel(nav_model_path, device)
        else:
            self.navigation_model = None 
        self.navigation_action_space = spaces.Discrete(4)

        # Cautious Variables Class
        self.cautious_sensors = CautiousVars()
        # vx, vy, dist_to_curve, curve_severity, heading_alignment, cross_track,
        # cross_track_rate, off_track, time_off_track, episode_completion, curves_passed
        # -- see utils/cautious_variables.py:get_cautious_var(). No raw (x, y) position,
        # no time_to_curve, no frame_counter (that's in the augmented block below now).
        self.n_cautious = 11
        self.n_scalars = 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_cautious + self.n_scalars,), dtype=np.float32)

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
        self.track_hash = hashlib.md5(track.tobytes()).hexdigest()

        self.world_step_count = 0
        self.steps_since_last_obs = 0
        self.episode_frame_count = 1
        self.fps_penalty = 0.0
        self.current_fps = self.fps_choices[-1]          # start at fastest (50)
        self.obs_interval = int(self.simulation_fps / self.current_fps)
        self.mask_buffer = []                            # step appends to this
        self.reached_goal = False
        self.budget_pen_check = False

        # Goal Settings: fixed absolute arc-length distance from the start line, not a
        # fraction of track point count (point spacing/track length both vary per
        # procedurally generated track -- see AdaptiveFPS_TrackAware_Wrapper.__init__).
        track_xy = track[:, 2:4].astype(np.float64)
        seg_len = np.linalg.norm(np.roll(track_xy, -1, axis=0) - track_xy, axis=1)
        total_track_len = float(seg_len.sum())

        # If this procedurally-generated track is shorter than goal_distance, require
        # extra full laps first -- a single fixed point can't otherwise represent "drive
        # goal_distance total", since the track is a closed loop and any point on it is
        # reachable well before goal_distance ticks have elapsed if we don't also gate on
        # lap count. laps_required=1 (the common case, track long enough) reduces exactly
        # to the old single-lap behavior.
        self.laps_required = max(1, math.ceil(self.goal_distance / total_track_len))
        final_lap_arc = self.goal_distance - (self.laps_required - 1) * total_track_len
        self.goal_xy = _arc_length_point(track_xy, seg_len, final_lap_arc)
        self.goal_radius = 20.0
        self.min_steps = 20

        # Lap-crossing tracker for the multi-lap case above: edge-triggered proximity to
        # the start point (track[0], also where the car spawns -- see CarRacing_VarFramerate
        # .reset()), checked every physics tick (not just sampling instants) so it can't
        # miss a crossing between two low-FPS samples. Not derived from the env's own
        # tile_visited_count/new_lap, which only ever fire once per episode by design (each
        # tile's road_visited flag is a first-contact-only latch -- see FrictionDetector in
        # envs/car_racing_var_fps.py) and so can't count multiple laps.
        self.laps_completed = 0
        self._start_xy = track_xy[0]
        self._near_start = False

        # Frames
        self.last_sampled_obs = observed_frame                      # raw (4,96,96) image for the controller

        # Cautious Variables
        self.cautious_sensors.reset_track_reading(self.env)
        # dt_ticks=0: first reading of the episode, no prior tick to diff/accumulate against
        self.last_sampled_cautious_obs = self.cautious_sensors.get_cautious_var(self.env, dt_ticks=0)

        # Current obs here is the cautious var (11-dim) + 3 scalars
        self.current_obs = self._get_augmented_obs(self.last_sampled_cautious_obs)  # augmented (36866,) for the FPS policy

        info = dict(info)
        # Which procedurally generated track this episode is running on -- audit this
        # against training/eval logs to confirm tracks are actually varying, not being
        # silently pinned to one seed (see wrappers/pre_processing.py).
        info["track_hash"] = self.track_hash
        return self.current_obs, info
  
    def _get_augmented_obs(self, obs):
        cautious_obs = obs
        obs_age_steps = self.steps_since_last_obs
        # normalizing the num of steps since last obs
        obs_age_ratio = obs_age_steps / self.simulation_fps
        # normalizing the num of steps since last obs
        fps_ratio = self.current_fps/self.simulation_fps
        # how many decisions spent out of budget -- clean 0->1 ramp that hits 1.0
        # exactly when the budget-overrun penalty (below) is about to fire
        frame_counter = np.clip(self.episode_frame_count / self.budget, 0, 1)

        aug_obs = np.concatenate([
            cautious_obs,
            np.array([obs_age_ratio, fps_ratio, frame_counter], dtype=np.float32)])
        return aug_obs

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
        x, y = self.env.unwrapped.car.hull.position

        # Edge-triggered lap counter: every tick (not just sampling instants, so a slow
        # chosen FPS can't skip a crossing), check proximity to the start line. Count a
        # lap only on the rising edge (entering the radius, not staying inside it) so
        # lingering near the start doesn't count more than once per pass.
        near_start_now = np.hypot(x - self._start_xy[0], y - self._start_xy[1]) < self.goal_radius
        if near_start_now and not self._near_start and self.world_step_count > self.min_steps:
            self.laps_completed += 1
        self._near_start = near_start_now

        self.reached_goal = (
            self.laps_completed >= self.laps_required - 1
            and np.hypot(x - self.goal_xy[0], y - self.goal_xy[1]) < self.goal_radius
            and self.world_step_count > self.min_steps
        )

        # The base env ends the episode as soon as it detects one lap complete
        # (tile_visited_count/new_lap -- see CarRacing_VarFramerate.step()), which would
        # cut a multi-lap goal (self.laps_required > 1) short before the car ever reaches
        # goal_xy on a later lap. That flag latches true for the rest of the episode once
        # tripped (each tile's road_visited is a first-contact-only flag, so
        # tile_visited_count never drops back down), so suppress exactly this cause of
        # termination for as long as the goal hasn't actually been reached yet -- any
        # OTHER cause (off-track timeout, wrong direction, stalled, off-playfield) still
        # ends the episode normally.
        if (
            info.get("lap_finished")
            and not info.get("off_track_timeout")
            and not info.get("wrong_direction")
            and not info.get("stalled")
            and not self.reached_goal
        ):
            terminated = False

        # 4. Check if it's time to sample a new observation based on the obs_interval
        # If so, update the last sampled observation, reset the steps since last observation count, and increment the episode frame count
        if self.steps_since_last_obs >= self.obs_interval:
            # self.current_obs is updated every physics step, so here we store the fresh observation of this step
            self.last_sampled_obs = observed_frame.copy()
            # dt_ticks = ticks elapsed since the last sample, captured before it's reset below --
            # scales the cross-track-rate and time-off-track features correctly regardless of
            # which FPS was chosen for the window that just elapsed.
            dt_ticks = self.steps_since_last_obs
            self.steps_since_last_obs = 0
            self.episode_frame_count += 1
            self.last_sampled_cautious_obs = self.cautious_sensors.get_cautious_var(self.env, dt_ticks=dt_ticks)
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

        # 5. Concatenate the cautious-var vector with the additional scalars (age ratio, fps ratio, episode frame count)
        # Current obs here is the cautious var (12-dim) + 3 scalars
        self.current_obs = self._get_augmented_obs(self.last_sampled_cautious_obs)
        
        # 6. Compute reward based on the navigation reward obtained from the physics step, and apply a penalty if a new frame was consumed
        frame_penalty = self.frame_cost if frame_consumed else 0.0
        reward = nav_reward - frame_penalty
        
        # DEBBUGING: Cumulate the fps penalty for the episode, which can be used for analysis and debugging
        self.fps_penalty += frame_penalty

        # DEBBUGING:
        np.set_printoptions(suppress=True, precision=4)
        #print(f"[AdaptiveC FPS Wrapper] Step: {self.world_step_count}, Obs: {self.current_obs}, Action: {self.current_fps}")
        #print(f"[AdaptiveC FPS Wrapper] Step: {self.world_step_count}, Obs: {self.current_obs}, Action: {self.current_fps}")

        if not self.reached_goal and self.episode_frame_count > self.budget and not self.budget_pen_check:
            reward = -100
            self.budget_pen_check = True

        if self.reached_goal:
            terminated = True
            reward = 150

        # Consolidated debug field: which single condition actually ended the episode.
        # reached_goal takes priority since it's the wrapper's own success condition and
        # can coincide with an env-reported flag on the same tick; among the env's own
        # terminal flags, "off_playfield" is the implicit case (terminated=True but none
        # of the other named flags fired -- see CarRacing_VarFramerate.step()). Note the
        # budget-overrun penalty does NOT terminate the episode (it's a one-shot penalty,
        # not a hard stop), so it never appears here even though it fires via reward.
        # wrong_direction/off_track_timeout/stalled are checked before lap_finished here
        # (unlike the priority order you might expect) because lap_finished is no longer
        # a reliable one-shot signal in the multi-lap case: it latches true in info for
        # the rest of the episode once one lap is done (see the suppression above), so it
        # would otherwise shadow whatever actually ended the episode on a later tick.
        if self.reached_goal:
            termination_reason = "reached_goal"
        elif info.get("wrong_direction"):
            termination_reason = "wrong_direction"
        elif info.get("off_track_timeout"):
            termination_reason = "off_track_timeout"
        elif info.get("stalled"):
            termination_reason = "stalled"
        elif info.get("lap_finished") and terminated:
            termination_reason = "lap_finished"
        elif terminated:
            termination_reason = "off_playfield"
        elif truncated:
            termination_reason = "timeout"
        else:
            termination_reason = "none"

        info = dict(info)
        info["reward"] = reward
        info["nav_reward"] = nav_reward
        info["frame_cost"] = self.frame_cost
        info["budget"] = self.budget
        info["chosen_fps"] = self.current_fps
        info["episode_frame_count"] = self.episode_frame_count
        info["timeout"] = truncated and not terminated
        info["reached_goal"] = self.reached_goal
        info["termination_reason"] = termination_reason
        info["track_hash"] = self.track_hash
        # Whether fps_action this tick was actually applied (a real sampling instant)
        # or silently discarded (obs_interval not yet elapsed) -- see training loop's
        # policy-loss masking, which must not train on discarded-action ticks.
        info["frame_consumed"] = frame_consumed

        return self.current_obs, reward, terminated, truncated, info