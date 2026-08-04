"""
car_racing_random_spawn.py

A self-contained CarRacing-v3 subclass for training a vector-observation
navigation policy (see experiments/navigation_vector/). Two changes on top of
gymnasium's stock CarRacing:

  1. The car spawns at a random point along the (freshly regenerated) track
     each episode, instead of always at track index 0.
  2. An optional `constant_speed` kwarg pins the car's actual velocity
     magnitude every physics tick (direction still follows that tick's
     physics/steering), reducing the control problem to steering-only.

This does NOT import envs/car_racing_var_fps.py -- the reward-shaping pieces
needed by utils/cautious_variables.py (curves_passed_count/total_curves_on_track,
off-track/wrong-direction/stalled terminations, curve-passed bonus) are copied
here rather than reused, to keep this env decoupled from the adaptive-FPS/goal
-distance logic that lives in CarRacing_VarFramerate.
"""

import math

import numpy as np

from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.envs.box2d.car_racing import CarRacing, FPS, PLAYFIELD, FrictionDetector
from gymnasium.envs.registration import register

# cos(car heading - track direction at the current tile): +1 aligned, -1 exactly
# backward. Below this, the car counts as driving the wrong way round the track
# rather than just cornering hard, and the episode is terminated instead of
# letting it keep racking up reward running backward.
WRONG_DIRECTION_ALIGNMENT_THRESHOLD = -0.8

# Terminate if the car sits below STATIC_SPEED_THRESHOLD (world-units/s) for
# more than STATIC_TIMEOUT_TICKS consecutive physics ticks, so a truly stuck
# car doesn't just sit there accumulating negative reward for the rest of the
# episode.
STATIC_SPEED_THRESHOLD = 1.0
STATIC_TIMEOUT_TICKS = 400

# Per-segment |beta[i+1]-beta[i]| angle threshold for "this tile is part of a
# curve" -- an independent calibration from CautiousVars' curve_thresh (which
# is a curvature RATE in rad/world-unit, not a raw per-segment angle).
CURVE_TILE_TURN_THRESHOLD = 0.05
# Reward bonus paid once, on cleanly exiting a curve region (no off-track/
# wrong-direction event while inside it).
CURVE_PASSED_BONUS = 20.0


class CarRacing_RandomSpawn(CarRacing):
    """CarRacing-v3 with randomized car spawn position and an optional
    constant-speed physics override, for training a vector-observation
    (cautious-variables) navigation policy."""

    def __init__(self, *args, constant_speed: float | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.constant_speed = constant_speed

    def _is_curve_tile(self, idx):
        """Whether track node `idx` sits inside a curve, from the raw
        per-segment beta change (self.track[i][1] is already the
        steering-integrator angle used to build the track)."""
        N = len(self.track)
        beta_here = self.track[idx][1]
        beta_next = self.track[(idx + 1) % N][1]
        dh = beta_next - beta_here
        dh = (dh + math.pi) % (2 * math.pi) - math.pi
        return abs(dh) > CURVE_TILE_TURN_THRESHOLD

    def _count_curve_regions(self):
        """One-time scan (called from reset(), after the track exists)
        counting how many separate curve regions the full lap has -- the
        denominator for the curves_passed observation feature in
        utils/cautious_variables.py."""
        N = len(self.track)
        count = 0
        prev_is_curve = self._is_curve_tile(N - 1)
        for i in range(N):
            cur_is_curve = self._is_curve_tile(i)
            if cur_is_curve and not prev_is_curve:
                count += 1
            prev_is_curve = cur_is_curve
        return count

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # Call gym.Env.reset directly (not CarRacing.reset) -- CarRacing.reset's
        # own body already does a full destroy/build/track-create/car-spawn cycle
        # that we're about to redo anyway with the randomized spawn below, so
        # calling it here would just build the Box2D world twice per reset.
        super(CarRacing, self).reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.static_ticks = 0  # consecutive ticks with speed < STATIC_SPEED_THRESHOLD
        self.off_track_ticks = 0  # consecutive ticks with >=3 wheels off road
        self.in_curve = False
        self.curve_clean = True
        self.curves_passed_count = 0

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]
            self._reinit_colors(randomize)

        while True:
            success = self._create_track()
            if success:
                break
            if self.verbose:
                print(
                    "retry to generate track (normal if there are not many"
                    "instances of this message)"
                )

        # Randomized spawn point: pick a random track node instead of always
        # index 0. track[idx] = (alpha, beta, x, y); beta is the node's
        # orientation, so Car(...) below spawns the car correctly oriented
        # for wherever it starts, exactly as the original track[0] call did.
        spawn_idx = self.np_random.integers(0, len(self.track))
        self.car = Car(self.world, *self.track[spawn_idx][1:4])
        self.total_curves_on_track = max(self._count_curve_regions(), 1)  # avoid /0

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def step(self, action):
        assert self.car is not None
        if action is not None:
            if self.continuous:
                action = action.astype(np.float64)
                self.car.steer(-action[0])
                self.car.gas(action[1])
                self.car.brake(action[2])
            else:
                self.car.steer(-0.6 * (action == 1) + 0.6 * (action == 2))
                self.car.gas(0.2 * (action == 3))
                self.car.brake(0.8 * (action == 4))

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        if self.constant_speed is not None:
            vx, vy = self.car.hull.linearVelocity
            speed = math.hypot(vx, vy)
            if speed > 1e-3:
                scale = self.constant_speed / speed
                self.car.hull.linearVelocity = (vx * scale, vy * scale)
            else:
                # At rest (e.g. the very first tick after spawn): fall back to
                # the hull's facing direction rather than dividing by ~0.
                angle = self.car.hull.angle + math.pi / 2
                self.car.hull.linearVelocity = (
                    self.constant_speed * math.cos(angle),
                    self.constant_speed * math.sin(angle),
                )

        self.state = self._render("state_pixels")

        step_reward = 0
        terminated = False
        truncated = False
        info = {}
        if action is not None:  # First step without action, called from reset()
            self.reward -= 0.1
            self.car.fuel_spent = 0.0

            # Off-track penalty: fires once >=3 of 4 wheels have lost contact
            # with the road.
            off_track_wheels = sum(len(w.tiles) == 0 for w in self.car.wheels)
            if off_track_wheels >= 3:
                self.reward -= 0.1

            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward

            if off_track_wheels >= 3:
                self.off_track_ticks += 1
            else:
                self.off_track_ticks = 0
            if self.off_track_ticks > STATIC_TIMEOUT_TICKS:
                terminated = True
                info["off_track_timeout"] = True
                step_reward = -100

            if self.tile_visited_count == len(self.track) or self.new_lap:
                terminated = True
                info["lap_finished"] = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                info["lap_finished"] = False
                step_reward = -100

            # Wrong-direction termination: find the tile currently under any wheel.
            current_tile_idx = None
            for w in self.car.wheels:
                if len(w.tiles) > 0:
                    current_tile_idx = next(iter(w.tiles)).idx
                    break
            wrong_direction_now = False
            if current_tile_idx is not None:
                track_heading = self.track[current_tile_idx][1]  # beta
                heading_alignment = math.cos(self.car.hull.angle - track_heading)
                if heading_alignment < WRONG_DIRECTION_ALIGNMENT_THRESHOLD:
                    terminated = True
                    info["wrong_direction"] = True
                    step_reward = -100
                    wrong_direction_now = True

            # Curves-passed tracking: entering/exiting a curve region. "Clean"
            # means no off-track/wrong-direction event fired while inside the
            # region -- only a clean exit earns the bonus and counts toward
            # curves_passed_count (read by CautiousVars for the observation
            # feature).
            if current_tile_idx is not None:
                is_curve_now = self._is_curve_tile(current_tile_idx)
                if is_curve_now and not self.in_curve:
                    self.in_curve = True
                    self.curve_clean = True
                elif not is_curve_now and self.in_curve:
                    self.in_curve = False
                    if self.curve_clean:
                        self.curves_passed_count += 1
                        step_reward += CURVE_PASSED_BONUS
                        info["curve_passed"] = True
                if self.in_curve and (off_track_wheels >= 3 or wrong_direction_now):
                    self.curve_clean = False

            # Stalled-car termination.
            speed = float(np.linalg.norm(self.car.hull.linearVelocity))
            if speed < STATIC_SPEED_THRESHOLD:
                self.static_ticks += 1
            else:
                self.static_ticks = 0
            if self.static_ticks > STATIC_TIMEOUT_TICKS:
                terminated = True
                info["stalled"] = True
                step_reward = -100

        if self.render_mode == "human":
            self.render()
        return self.state, step_reward, terminated, truncated, info


register(
    id="CarRacing_RandomSpawn",
    entry_point="envs.car_racing_random_spawn:CarRacing_RandomSpawn",
)
