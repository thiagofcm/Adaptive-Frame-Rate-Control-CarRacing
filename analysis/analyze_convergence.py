"""
analyze_convergence.py

Step 1b of the FPS-policy investigation plan: a direct, per-state check of whether
the trained checkpoint has actually converged to a confident, state-conditioned FPS
strategy -- as opposed to training/losses/entropy (a masked AVERAGE over the whole
rollout) merely looking converged while remaining near-uniform conditional on any
specific state.

Consumes the per-decision CSV written by collect_adaptive_rollouts.py (columns:
seed, tick, chosen_fps, entropy, top1_prob, curv_abs, curv_signed, turn, in_curve,
off_track, track_hash).

Usage:
  python analysis/analyze_convergence.py --csv analysis_out/adaptive_rollouts.csv \
      --out-dir analysis_out
"""
import argparse
import csv
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LN5 = math.log(5)


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "seed": int(row["seed"]),
                "entropy": float(row["entropy"]),
                "top1_prob": float(row["top1_prob"]),
                "curv_abs": float(row["curv_abs"]),
                "in_curve": row["in_curve"] == "True",
                "chosen_fps": int(row["chosen_fps"]),
            })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", default="analysis_out")
    p.add_argument("--n-bins", type=int, default=5)
    args = p.parse_args()

    rows = load_rows(args.csv)
    n = len(rows)
    n_episodes = len(set(r["seed"] for r in rows))
    entropy = np.array([r["entropy"] for r in rows])
    top1 = np.array([r["top1_prob"] for r in rows])
    curv_abs = np.array([r["curv_abs"] for r in rows])
    in_curve = np.array([r["in_curve"] for r in rows])

    print(f"=== Step 1b: per-state convergence check ({n} decisions, {n_episodes} episodes) ===")
    print(f"ln(5) = {LN5:.4f}  (max entropy of a uniform 5-way categorical)")
    print(f"entropy: mean={entropy.mean():.4f}  median={np.median(entropy):.4f}  "
          f"std={entropy.std():.4f}  min={entropy.min():.4f}  max={entropy.max():.4f}")
    print(f"fraction of decisions with entropy > 0.95*ln(5) (near-uniform): "
          f"{(entropy > 0.95 * LN5).mean():.1%}")
    print(f"fraction of decisions with entropy < 1.0 nat (confident-ish): {(entropy < 1.0).mean():.1%}")
    print(f"top1_prob: mean={top1.mean():.4f}  median={np.median(top1):.4f}  "
          f"fraction with top1_prob > 0.6: {(top1 > 0.6).mean():.1%}")

    print(f"\nin_curve fraction of decisions: {in_curve.mean():.1%}")
    print(f"entropy | in_curve=True:  mean={entropy[in_curve].mean():.4f}  n={in_curve.sum()}")
    print(f"entropy | in_curve=False: mean={entropy[~in_curve].mean():.4f}  n={(~in_curve).sum()}")
    print(f"top1_prob | in_curve=True:  mean={top1[in_curve].mean():.4f}")
    print(f"top1_prob | in_curve=False: mean={top1[~in_curve].mean():.4f}")

    # Quantile-binned view: entropy/top1_prob/mean_fps as a function of ground-truth
    # curvature magnitude -- a first, cheap look at state-conditioning (not yet the
    # direction test, that's analyze_fps_vs_curvature.py).
    raw_edges = np.quantile(curv_abs, np.linspace(0, 1, args.n_bins + 1))
    edges = np.unique(raw_edges)
    edges[-1] += 1e-9
    if len(edges) - 1 < args.n_bins:
        print(f"[warn] requested {args.n_bins} bins but curvature has only {len(edges)-1} "
              f"distinct quantile edges (many exactly-zero-curvature straights) -- using {len(edges)-1} bins")
    args.n_bins = len(edges) - 1
    bin_idx = np.digitize(curv_abs, edges[1:-1], right=True)

    print(f"\n{'bin':>4} {'curv range':>22} {'n':>7} {'mean_entropy':>13} {'mean_top1':>10} {'mean_fps':>9}")
    for b in range(args.n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        fps_b = np.array([r["chosen_fps"] for r, m in zip(rows, mask) if m])
        print(f"{b:>4} [{edges[b]:.4f},{edges[b+1]:.4f}) {mask.sum():>7} "
              f"{entropy[mask].mean():>13.4f} {top1[mask].mean():>10.4f} {fps_b.mean():>9.2f}")

    os.makedirs(args.out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(entropy, bins=40, color="steelblue")
    axes[0].axvline(LN5, color="red", ls="--", label=f"ln(5)={LN5:.3f}")
    axes[0].set_xlabel("per-decision entropy (nats)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Distribution of FPS-policy decision entropy")
    axes[0].legend()

    bin_mid = 0.5 * (edges[:-1] + edges[1:])
    mean_ent_per_bin = [entropy[bin_idx == b].mean() if (bin_idx == b).any() else np.nan
                        for b in range(args.n_bins)]
    axes[1].plot(bin_mid, mean_ent_per_bin, "o-", color="darkorange")
    axes[1].axhline(LN5, color="red", ls="--", label=f"ln(5)={LN5:.3f}")
    axes[1].set_xlabel("ground-truth |curvature| (rad/world-unit), quantile bin midpoint")
    axes[1].set_ylabel("mean decision entropy (nats)")
    axes[1].set_title("Entropy vs. curvature (state-conditioning preview)")
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "convergence_check.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved plot -> {out_path}")


if __name__ == "__main__":
    main()
