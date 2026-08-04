"""
reconstruct_shaping_reward.py

Standalone, read-only, offline analysis script. Takes a single per-tick CSV already
produced by analysis/eval_straight_track_fps.py (one file per FPS, e.g.
pertick_fc1_8_bud500_0_fps25.csv) and reconstructs, tick by tick, what the reward
would have been if potential-based reward shaping (PBRS) had been added on top of the
already-logged nav_reward/frame_cost -- purely from the CSV, no env/wrapper/model
involved. Does not import or modify the wrapper, CautiousVars, or eval_straight_track_fps.py.

Potential function: Phi(cross_track) = -abs(cross_track) -- a state-only potential
built from the CSV's cross_track column, NOT cross_track_rate_norm. cross_track_rate_norm
is already a per-tick FINITE DIFFERENCE of cross_track (see utils/cautious_variables.py's
get_cautious_var()); PBRS's own shaping term is gamma*Phi(s')-Phi(s), itself a difference
of Phi across consecutive states. Building Phi from cross_track_rate_norm would therefore
double-differentiate the same underlying quantity (rate-of-a-rate), which is not what a
potential function is supposed to be. cross_track is a raw position, so PBRS's single
differencing step over it correctly recovers rate-like (rewarding-improvement) behavior
without over-differentiating.

frame_consumed inference: the per-tick CSV has no explicit frame_consumed column, but it
does have obs_age_ratio, and AdaptiveFPS_TrackAware_Wrapper.step() (see
wrappers/adaptive_fps_track_aware_wrapper.py) resets self.steps_since_last_obs to 0
*before* computing obs_age_ratio on exactly (and only) the ticks where
frame_consumed=True; obs_age_ratio is otherwise > 0. So frame_consumed is recovered
exactly (not a fuzzy proxy) as `obs_age_ratio == 0.0`. This is verified against the
wrapper's source, not guessed. If a per-tick CSV is ever produced without an
obs_age_ratio column, this script refuses to run rather than silently assume a value.

Usage:
  python analysis/reconstruct_shaping_reward.py \\
      --pertick-csv eval_straight_track/pertick_fc1_8_bud500_0_fps25.csv \\
      --potential-lambda 0.05,0.1,0.5,1.0
"""
import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Matches analysis/eval_straight_track_fps.py's naming convention exactly:
#   f"pertick_fc{fmt_tag(frame_cost)}_bud{fmt_tag(budget)}_fps{fps}.csv"
# where fmt_tag(x) = str(x).replace(".", "_"). fc/bud groups are restricted to
# digits+underscore only, so this can't misparse on the literal "_bud"/"_fps" markers.
FILENAME_RE = re.compile(r"^pertick_fc(?P<fc>[\d_]+)_bud(?P<bud>[\d_]+)_fps(?P<fps>\d+)\.csv$")

OBS_AGE_EPS = 1e-6  # float-equality tolerance for the obs_age_ratio == 0.0 frame_consumed check


def fmt_tag(x):
    """Filename-safe formatting matching eval_straight_track_fps.py's fc_1_8-style tags."""
    return str(x).replace(".", "_")


def parse_source_tag(csv_path):
    """Best-effort (fps, frame_cost, budget, tag) extraction from the source filename's
    naming convention. Returns None for any field it can't parse -- callers must not
    silently assume a value for anything that comes back None (see --frame-cost handling
    in main())."""
    m = FILENAME_RE.match(os.path.basename(csv_path))
    if not m:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        return {"fps": None, "frame_cost": None, "budget": None, "tag": base}
    fps = int(m.group("fps"))
    frame_cost = float(m.group("fc").replace("_", "."))
    budget = float(m.group("bud").replace("_", "."))
    tag = f"fps{fps}_fc_{m.group('fc')}_bud_{m.group('bud')}"
    return {"fps": fps, "frame_cost": frame_cost, "budget": budget, "tag": tag}


def read_pertick_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows")

    header = set(rows[0].keys())
    required = {"tick", "cross_track", "nav_reward", "reward"}
    missing = required - header
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    if "obs_age_ratio" not in header:
        # See module docstring: this is the field frame_consumed is inferred from.
        # Flagging rather than guessing, per the task requirement.
        raise ValueError(
            f"{path} has no obs_age_ratio column -- can't infer frame_consumed without "
            "a field that distinguishes real decision ticks from stale/held ticks. "
            "Refusing to guess; regenerate the per-tick CSV with the current "
            "eval_straight_track_fps.py (which logs the full augmented-obs vector)."
        )

    tick = np.array([int(r["tick"]) for r in rows], dtype=np.int64)
    cross_track = np.array([float(r["cross_track"]) for r in rows], dtype=np.float64)
    nav_reward = np.array([float(r["nav_reward"]) for r in rows], dtype=np.float64)
    original_reward = np.array([float(r["reward"]) for r in rows], dtype=np.float64)
    obs_age_ratio = np.array([float(r["obs_age_ratio"]) for r in rows], dtype=np.float64)
    frame_consumed = np.abs(obs_age_ratio) < OBS_AGE_EPS

    return {
        "tick": tick,
        "cross_track": cross_track,
        "nav_reward": nav_reward,
        "original_reward": original_reward,
        "frame_consumed": frame_consumed,
    }


def resolve_terminal_index(tick, terminal_tick_override):
    """Index (not tick number) of the row to treat as the episode's terminal tick.

    Default: the CSV's LAST row. eval_straight_track_fps.py's run_episode() logs a row
    every loop iteration and only exits the `while not done:` loop right after appending
    that iteration's row -- so the last row in the file IS the true terminal tick by
    construction, with no separate "terminated"/"truncated" column needed. in_curve
    can't be used as a termination signal here: on this straight-track experiment it's
    always False by design (see eval_straight_track_fps.py's own verification block),
    not an indicator of episode end. --terminal-tick is the escape hatch for a CSV that
    doesn't follow this convention (e.g. a hand-trimmed file).
    """
    if terminal_tick_override is None:
        return len(tick) - 1
    matches = np.nonzero(tick == terminal_tick_override)[0]
    if len(matches) == 0:
        raise ValueError(
            f"--terminal-tick {terminal_tick_override} not found in the CSV's tick column "
            f"(range [{tick.min()}, {tick.max()}])"
        )
    return int(matches[0])


def reconstruct(data, gamma, frame_cost, terminal_idx):
    """Lambda-independent reconstruction. Truncates everything to rows [0, terminal_idx]
    so a --terminal-tick override before the file's true end behaves as "episode ends
    here", discarding any trailing rows."""
    tick = data["tick"][: terminal_idx + 1]
    cross_track = data["cross_track"][: terminal_idx + 1]
    nav_reward = data["nav_reward"][: terminal_idx + 1]
    original_reward = data["original_reward"][: terminal_idx + 1]
    frame_consumed = data["frame_consumed"][: terminal_idx + 1]

    # Phi(s_i) = -|cross_track_i| -- state-only potential (see module docstring for why
    # cross_track, not cross_track_rate_norm).
    phi = -np.abs(cross_track)

    # next_phi[i] = Phi(s_{i+1}), i.e. the potential of the state THIS row's tick
    # transitions into. Forced to 0.0 at the terminal row -- PBRS's policy-invariance
    # guarantee requires Phi(terminal/absorbing state) = 0, and the state "after" the
    # terminal tick has no cross_track of its own to derive a potential from anyway.
    next_phi = np.empty_like(phi)
    next_phi[:-1] = phi[1:]
    next_phi[-1] = 0.0

    frame_cost_applied = np.where(frame_consumed, frame_cost, 0.0)
    base_reward = nav_reward - frame_cost_applied
    # shaping_reward_raw is lambda=1: F(s,a,s') = gamma*Phi(s') - Phi(s). Scaling by an
    # arbitrary lambda for a sweep is then just a multiply -- see apply_lambda().
    shaping_reward_raw = gamma * next_phi - phi

    return {
        "tick": tick,
        "cross_track": cross_track,
        "phi": phi,
        "frame_consumed": frame_consumed,
        "frame_cost_applied": frame_cost_applied,
        "nav_reward": nav_reward,
        "base_reward": base_reward,
        "shaping_reward_raw": shaping_reward_raw,
        "original_reward": original_reward,
    }


def apply_lambda(recon, lam):
    shaping_reward_scaled = lam * recon["shaping_reward_raw"]
    total_reward_with_shaping = recon["base_reward"] + shaping_reward_scaled
    return shaping_reward_scaled, total_reward_with_shaping


def write_pertick_csv(recon, lam_primary, out_path):
    shaping_reward_scaled, total_reward_with_shaping = apply_lambda(recon, lam_primary)
    fields = ["tick", "cross_track", "phi", "frame_consumed", "frame_cost_applied",
              "nav_reward", "shaping_reward_raw", "shaping_reward_scaled",
              "base_reward", "total_reward_with_shaping", "original_reward"]
    n = len(recon["tick"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i in range(n):
            w.writerow([
                int(recon["tick"][i]),
                round(float(recon["cross_track"][i]), 5),
                round(float(recon["phi"][i]), 5),
                bool(recon["frame_consumed"][i]),
                round(float(recon["frame_cost_applied"][i]), 5),
                round(float(recon["nav_reward"][i]), 5),
                round(float(recon["shaping_reward_raw"][i]), 5),
                round(float(shaping_reward_scaled[i]), 5),
                round(float(recon["base_reward"][i]), 5),
                round(float(total_reward_with_shaping[i]), 5),
                round(float(recon["original_reward"][i]), 5),
            ])
    return shaping_reward_scaled, total_reward_with_shaping


def summarize(recon, lam, fps, gamma):
    shaping_reward_scaled, total_reward_with_shaping = apply_lambda(recon, lam)
    total_nav_reward = float(recon["nav_reward"].sum())
    total_frame_cost_paid = float(recon["frame_cost_applied"].sum())
    total_base_reward = float(recon["base_reward"].sum())
    total_shaping_reward = float(shaping_reward_scaled.sum())
    total_reward_with_shaping_sum = float(total_reward_with_shaping.sum())
    ratio = (100.0 * total_shaping_reward / total_nav_reward
             if total_nav_reward != 0 else float("nan"))
    abs_cross = np.abs(recon["cross_track"])
    return {
        "fps": fps if fps is not None else "unknown",
        "gamma": gamma,
        "potential_lambda": lam,
        "total_nav_reward": round(total_nav_reward, 4),
        "total_frame_cost_paid": round(total_frame_cost_paid, 4),
        "total_base_reward": round(total_base_reward, 4),
        "total_shaping_reward": round(total_shaping_reward, 4),
        "total_reward_with_shaping": round(total_reward_with_shaping_sum, 4),
        "shaping_to_nav_reward_ratio": round(ratio, 3) if ratio == ratio else "nan",  # nan != nan
        "mean_abs_cross_track": round(float(abs_cross.mean()), 5),
        "max_abs_cross_track": round(float(abs_cross.max()), 5),
    }


def plot_shaping(recon, lam_primary, save_path, title):
    shaping_reward_scaled, _ = apply_lambda(recon, lam_primary)
    tick = recon["tick"]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_top.plot(tick, recon["cross_track"], color="steelblue")
    ax_top.axhline(0.0, color="0.6", lw=1, ls="--")
    ax_top.set_ylabel("cross_track")
    ax_top.set_title(title)
    ax_top.grid(alpha=0.3)

    ax_bot.plot(tick, shaping_reward_scaled, color="crimson")
    ax_bot.axhline(0.0, color="0.6", lw=1, ls="--")
    ax_bot.set_xlabel("tick")
    ax_bot.set_ylabel(f"shaping_reward_scaled\n(lambda={lam_primary})")
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pertick-csv", required=True,
                    help="one FPS's per-tick CSV, as produced by eval_straight_track_fps.py")
    p.add_argument("--gamma", type=float, default=0.99, help="must match training gamma")
    p.add_argument("--potential-lambda", default="1.0",
                    help="single float, or comma-separated list to sweep, e.g. '0.05,0.1,0.5,1.0'. "
                         "The per-tick CSV/plot use the first value; the summary CSV gets one row per value.")
    p.add_argument("--frame-cost", type=float, default=None,
                    help="overrides frame_cost parsed from the filename "
                         "(pertick_fc{X}_bud{Y}_fps{Z}.csv); required if the filename doesn't "
                         "follow that convention")
    p.add_argument("--terminal-tick", type=int, default=None,
                    help="tick number to treat as the episode's terminal tick (Phi forced to 0 "
                         "for the state after it). Default: the CSV's last row -- see "
                         "resolve_terminal_index() for why that's already exact.")
    p.add_argument("--out-dir", default="eval_straight_track/shaping_reconstruction")
    args = p.parse_args()

    lambdas = [float(x) for x in args.potential_lambda.split(",")]
    lam_primary = lambdas[0]

    src = parse_source_tag(args.pertick_csv)
    frame_cost = args.frame_cost if args.frame_cost is not None else src["frame_cost"]
    if frame_cost is None:
        raise SystemExit(
            f"Could not parse frame_cost from filename {os.path.basename(args.pertick_csv)!r} "
            "(expected the pertick_fc{X}_bud{Y}_fps{Z}.csv convention) -- pass --frame-cost explicitly."
        )
    if args.frame_cost is None:
        print(f"[reconstruct_shaping_reward] frame_cost={frame_cost} parsed from filename")
    else:
        print(f"[reconstruct_shaping_reward] frame_cost={frame_cost} (explicit --frame-cost)")

    data = read_pertick_csv(args.pertick_csv)
    terminal_idx = resolve_terminal_index(data["tick"], args.terminal_tick)
    recon = reconstruct(data, args.gamma, frame_cost, terminal_idx)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = src["tag"]
    lam_tag = fmt_tag(lam_primary)

    pertick_path = os.path.join(args.out_dir, f"shaping_recon_{tag}_lam{lam_tag}.csv")
    write_pertick_csv(recon, lam_primary, pertick_path)
    print(f"wrote {pertick_path}")

    summary_rows = [summarize(recon, lam, src["fps"], args.gamma) for lam in lambdas]
    summary_path = os.path.join(args.out_dir, f"shaping_summary_{tag}.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"wrote {summary_path}")

    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"shaping_plot_{tag}_lam{lam_tag}.png")
    plot_shaping(recon, lam_primary, plot_path,
                 title=f"{tag}  (gamma={args.gamma}, primary lambda={lam_primary})")
    print(f"wrote {plot_path}")

    print(f"\n=== shaping reconstruction summary: {tag} ===")
    header = list(summary_rows[0].keys())
    print(" ".join(f"{h:>22}" for h in header))
    for row in summary_rows:
        print(" ".join(f"{str(row[h]):>22}" for h in header))


if __name__ == "__main__":
    main()
