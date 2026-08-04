from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.registration import register

from envs.car_racing_var_fps import CarRacing_VarFramerate, FrictionDetector


class CarRacing_VarFramerate_RandomStart(CarRacing_VarFramerate):
    """CarRacing_VarFramerate, but reset() spawns the car at a random track node each
    episode instead of always index 0. Everything else (_create_track, step,
    FrictionDetector, curve counting, etc.) is inherited unchanged."""

    def reset(self, *, seed=None, options=None):
        # Full copy of CarRacing_VarFramerate.reset() (envs/car_racing_var_fps.py),
        # with only the car-spawn line changed. IMPORTANT: calls CarRacing.reset()
        # directly (the grandparent class), NOT super().reset() -- a zero-arg
        # super() call here would resolve to CarRacing_VarFramerate.reset(), i.e. the
        # exact method this override replaces, and would recursively re-run the whole
        # reset sequence (double track generation, double physics step via the
        # trailing self.step(None)) before this method's own code even runs.
        CarRacing.reset(self, seed=seed)
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

        # The one line that differs from the parent: spawn at a random track node
        # instead of always index 0. self.np_random is already (re)seeded by
        # CarRacing.reset() above -- the same generator _create_track()/
        # _reinit_colors() already draw from -- so this is reproducible from `seed=`
        # and advances correctly across autoreset episodes exactly like track
        # generation itself does. Stored on self so the wrapper can read it back.
        self.spawn_idx = int(self.np_random.integers(0, len(self.track)))
        self.car = Car(self.world, *self.track[self.spawn_idx][1:4])

        self.total_curves_on_track = max(self._count_curve_regions(), 1)  # avoid /0

        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}


register(
    id="CarRacing_VarFramerate_RandomStart",
    entry_point="envs.car_racing_var_fps_random_start:CarRacing_VarFramerate_RandomStart",
)
