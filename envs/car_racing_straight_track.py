"""
car_racing_straight_track.py

A synthetic "stadium" track (two straight segments joined by two semicircular
turns) for isolating straight-line driving performance from curve-navigation --
built to answer a specific question: is a high sensing rate (FPS) actually
needed on straight sections, or does a lower FPS do just as well there once
curve-navigation is removed as a confound entirely?

A perfectly straight CLOSED LOOP is geometrically impossible (total curvature
around any simple closed curve must be 2*pi), and this env's reward/termination/
lap logic (tile_visited_count, FrictionDetector, _is_curve_tile) all assume a
closed loop. So instead: build a closed stadium loop, and place a controllable
goal point (via AdaptiveFPS_TrackAware_Wrapper's goal_distance) strictly within
the length of one straight segment, so an evaluated episode never has to
navigate either turn at all.

Geometry is sized to stay well inside PLAYFIELD = 2000/SCALE = 333.3 world
units (envs/car_racing_var_fps.py) -- the base env hard-terminates with -100
reward if the car ever exceeds +-PLAYFIELD on either axis. With
straight_length=400, turn_radius=80: the centerline's farthest point from
origin is straight_length/2 + turn_radius = 280 world units; adding the road's
half-width + border (~8 units) keeps every drivable point under ~288, comfortably
inside +-333.3.
"""
import math

import numpy as np
from gymnasium.envs.registration import register

from envs.car_racing_var_fps import (
    CarRacing_VarFramerate,
    TRACK_DETAIL_STEP,
    TRACK_WIDTH,
    TRACK_TURN_RATE,
    BORDER,
    BORDER_MIN_COUNT,
)

STRAIGHT_LENGTH = 400.0   # world units per straight segment
TURN_RADIUS = 80.0        # world units; see module docstring for the PLAYFIELD margin math


class CarRacing_StraightTrack(CarRacing_VarFramerate):
    """Same env/reward/physics as CarRacing_VarFramerate, but on a deterministic
    stadium track (two straights + two semicircular turns) instead of a
    procedurally-generated one. Only _create_track() is overridden -- reset(),
    step(), _is_curve_tile(), FrictionDetector etc. all operate generically over
    self.track/self.road/self.road_poly regardless of how they were built."""

    def _create_track(self):
        # Must be reset here, not left to the caller: gymnasium's base CarRacing.reset()
        # calls _create_track() once itself (before CarRacing_VarFramerate.reset() repeats
        # the whole destroy/rebuild sequence and calls it again) -- on the very first
        # reset self.road is still None (from __init__), and _destroy() no-ops on a
        # falsy self.road, so nothing else clears it before this runs. Matches
        # CarRacing_VarFramerate._create_track()'s own `self.road = []` at its top.
        self.road = []

        L = STRAIGHT_LENGTH
        R = TURN_RADIUS
        step = TRACK_DETAIL_STEP

        points = []

        # 1. bottom straight: (-L/2,-R) -> (+L/2,-R), moving in +x
        n1 = max(2, int(L / step))
        for i in range(n1):
            points.append((-L / 2 + L * i / n1, -R))

        # 2. right semicircle: center (+L/2, 0), sweep -90deg -> +90deg (through 0deg)
        n2 = max(2, int(math.pi * R / step))
        for i in range(n2):
            theta = -math.pi / 2 + math.pi * i / n2
            points.append((L / 2 + R * math.cos(theta), R * math.sin(theta)))

        # 3. top straight: (+L/2,+R) -> (-L/2,+R), moving in -x
        n3 = max(2, int(L / step))
        for i in range(n3):
            points.append((L / 2 - L * i / n3, R))

        # 4. left semicircle: center (-L/2, 0), sweep +90deg -> +270deg (closes the loop
        #    back to point 1's start, (-L/2,-R), at theta=270deg=-90deg)
        n4 = max(2, int(math.pi * R / step))
        for i in range(n4):
            theta = math.pi / 2 + math.pi * i / n4
            points.append((-L / 2 + R * math.cos(theta), R * math.sin(theta)))

        xy = np.array(points, dtype=np.float64)
        nxt = np.roll(xy, -1, axis=0)
        d = nxt - xy
        true_heading = np.arctan2(d[:, 1], d[:, 0])
        # this codebase's established beta convention: true heading = beta + 90deg
        # (see utils/cautious_variables.py's heading_alignment derivation)
        beta = true_heading - math.pi / 2
        # alpha (polar angle around map center) isn't read anywhere at runtime --
        # only CarRacing_VarFramerate._create_track()'s own checkpoint-chasing
        # algorithm (bypassed here) uses it. Filled in for structural consistency only.
        alpha = np.arctan2(xy[:, 1], xy[:, 0])

        track = list(zip(alpha.tolist(), beta.tolist(), xy[:, 0].tolist(), xy[:, 1].tolist()))

        # --- Everything below is the "Create tiles" logic from
        # CarRacing_VarFramerate._create_track() (car_racing_var_fps.py:447-519),
        # reused verbatim -- it only consumes `track`, agnostic to how it was built. ---

        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = False
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                        y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                        y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append((
                    [b1_l, b1_r, b2_r, b2_l],
                    (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                ))

        self.track = track
        return True


register(
    id="CarRacing_StraightTrack",
    entry_point="envs.car_racing_straight_track:CarRacing_StraightTrack",
)
