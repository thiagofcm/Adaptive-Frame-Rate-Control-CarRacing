import pandas as pd
import numpy as np

# assuming your straight-track eval script logs per-tick cautious_obs
df25 = pd.read_csv("eval_straight_track/pertick_fc0_0_bud500_fps25.csv")
df50 = pd.read_csv("eval_straight_track/pertick_fc0_0_bud500_fps50.csv")

print("FPS=25 |cross_track_rate_norm| mean:", df25["cross_track_rate_norm"].abs().mean())
print("FPS=50 |cross_track_rate_norm| mean:", df50["cross_track_rate_norm"].abs().mean())
print("FPS=25 |cross_track_rate_norm| p95:", df25["cross_track_rate_norm"].abs().quantile(0.95))
print("FPS=50 |cross_track_rate_norm| p95:", df50["cross_track_rate_norm"].abs().quantile(0.95))