"""
analyze_fps_vs_curvature.py

Step 2a of the FPS-policy investigation plan: turns the "eyeballed" observation
(high FPS on straights, 25 FPS on curves) into a rigorous, continuous-curvature
plot with proper uncertainty, plus a direction/significance test.

Consumes the per-decision CSV written by collect_adaptive_rollouts.py.

Statistical design:
  - Unit of analysis = one decision (not one physics tick): consecutive ticks
    under a held FPS choice are correlated, not independent draws.
  - Curvature is binned into quantile bins (continuous ground truth, not a binary
    curve/straight split) via |curv_signed| from CautiousVars.curv.
  - Uncertainty (CI ribbon) is a CLUSTER bootstrap at the episode (seed) level:
    resample whole episodes with replacement, recompute each bin's across-episode
    mean-of-per-episode-bin-means every resample. This respects the fact that
    decisions within one episode are not independent of each other.
  - Direction/significance: Spearman rank correlation between curv_abs and
    chosen_fps across all decisions, tested against a null built by shuffling
    each episode's sequence of chosen_fps values independently per episode
    (preserves each episode's own FPS-choice distribution and curvature sequence,
    only breaks the pairing between the two) -- a block/cluster permutation test.

Usage:
  python analysis/analyze_fps_vs_curvature.py --csv analysis_out/adaptive_rollouts.csv \
      --out-dir analysis_out --n-bins 14 --n-bootstrap 2000 --n-permutations 2000
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path):
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "seed": int(row["seed"]),
                "curv_abs": float(row["curv_abs"]),
                "chosen_fps": int(row["chosen_fps"]),
            })
    return rows


def spearman(x, y):
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a))
        # average ties
        sorted_a = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
                j += 1
            if j > i:
                avg = ranks[order[i:j + 1]].mean()
                ranks[order[i:j + 1]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(np.asarray(x, dtype=np.float64)), rank(np.asarray(y, dtype=np.float64))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--out-dir", default="analysis_out")
    p.add_argument("--n-bins", type=int, default=14)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--n-permutations", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap/permutation")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load_rows(args.csv)
    seeds = sorted(set(r["seed"] for r in rows))
    n_episodes = len(seeds)
    print(f"=== Step 2a: FPS vs. ground-truth curvature ({len(rows)} decisions, {n_episodes} episodes) ===")

    curv_all = np.array([r["curv_abs"] for r in rows])
    fps_all = np.array([r["chosen_fps"] for r in rows], dtype=np.float64)

    # Quantile bin edges computed on the pooled data (fixed once, reused for every
    # bootstrap/permutation resample so bin boundaries don't move with the resample).
    # Many track nodes have exactly curv_abs==0 (straights), which can collapse
    # several requested quantiles onto the same edge -- dedupe so every bin is
    # non-degenerate, even if that means fewer than --n-bins bins.
    raw_edges = np.quantile(curv_all, np.linspace(0, 1, args.n_bins + 1))
    edges = np.unique(raw_edges)
    edges[-1] += 1e-9
    n_bins = len(edges) - 1
    if n_bins < args.n_bins:
        print(f"[warn] requested {args.n_bins} bins but curvature has only {n_bins} "
              f"distinct quantile edges (many exactly-zero-curvature straights) -- using {n_bins} bins")
    args.n_bins = n_bins

    # Per-episode arrays, for cluster bootstrap.
    by_seed = defaultdict(lambda: {"curv": [], "fps": []})
    for r in rows:
        by_seed[r["seed"]]["curv"].append(r["curv_abs"])
        by_seed[r["seed"]]["fps"].append(r["chosen_fps"])
    for s in by_seed:
        by_seed[s]["curv"] = np.array(by_seed[s]["curv"])
        by_seed[s]["fps"] = np.array(by_seed[s]["fps"], dtype=np.float64)

    def per_episode_bin_means(seed_list):
        """For each bin, average the per-episode mean chosen_fps over the given
        (possibly resampled-with-repeats) list of episodes. Returns array (n_bins,)."""
        sums = np.zeros(args.n_bins)
        counts = np.zeros(args.n_bins)
        for s in seed_list:
            curv = by_seed[s]["curv"]
            fps = by_seed[s]["fps"]
            bin_idx = np.digitize(curv, edges[1:-1], right=True)
            for b in range(args.n_bins):
                m = bin_idx == b
                if m.any():
                    sums[b] += fps[m].mean()
                    counts[b] += 1
        with np.errstate(invalid="ignore"):
            return np.where(counts > 0, sums / counts, np.nan)

    point_estimate = per_episode_bin_means(seeds)

    # Decision counts per bin (sample-size sanity panel).
    bin_idx_all = np.digitize(curv_all, edges[1:-1], right=True)
    decision_counts = np.array([(bin_idx_all == b).sum() for b in range(args.n_bins)])
    episodes_per_bin = np.array([
        sum(1 for s in seeds if (np.digitize(by_seed[s]["curv"], edges[1:-1], right=True) == b).any())
        for b in range(args.n_bins)
    ])

    print(f"\n{'bin':>4} {'curv range':>22} {'n_decisions':>12} {'n_episodes':>11} {'mean_chosen_fps':>16}")
    for b in range(args.n_bins):
        print(f"{b:>4} [{edges[b]:.4f},{edges[b+1]:.4f}) {decision_counts[b]:>12} "
              f"{episodes_per_bin[b]:>11} {point_estimate[b]:>16.2f}")

    # --- Cluster (episode-level) bootstrap CI per bin ---
    print(f"\nRunning {args.n_bootstrap} episode-level bootstrap resamples for CIs...")
    boot = np.zeros((args.n_bootstrap, args.n_bins))
    for i in range(args.n_bootstrap):
        resample = rng.choice(seeds, size=n_episodes, replace=True)
        boot[i] = per_episode_bin_means(resample)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    # --- Direction/significance test: Spearman corr(curv_abs, chosen_fps), against
    # a null built by permuting each episode's OWN sequence of chosen_fps values
    # independently (preserves within-episode marginal FPS distribution and the
    # curvature sequence, only breaks the pairing -- a block permutation test). ---
    observed_rho = spearman(curv_all, fps_all)
    print(f"\nObserved Spearman rho(curvature, chosen_fps) = {observed_rho:.4f} "
          f"({'negative' if observed_rho < 0 else 'positive'} -> "
          f"{'higher curvature -> LOWER fps (opposite of prior)' if observed_rho < 0 else 'higher curvature -> HIGHER fps (matches prior)'})")

    print(f"Running {args.n_permutations} within-episode block permutations for a null distribution...")
    null_rhos = np.zeros(args.n_permutations)
    seed_positions = defaultdict(list)
    for i, r in enumerate(rows):
        seed_positions[r["seed"]].append(i)
    for k in range(args.n_permutations):
        fps_perm = fps_all.copy()
        for s, idxs in seed_positions.items():
            idxs = np.array(idxs)
            perm = rng.permutation(len(idxs))
            fps_perm[idxs] = fps_all[idxs][perm]
        null_rhos[k] = spearman(curv_all, fps_perm)

    p_value = float(np.mean(np.abs(null_rhos) >= abs(observed_rho)))
    print(f"Permutation-test p-value (two-sided, |null rho| >= |observed rho|): {p_value:.4f}")
    print(f"null rho: mean={null_rhos.mean():.4f}  std={null_rhos.std():.4f}")

    # --- Plot ---
    os.makedirs(args.out_dir, exist_ok=True)
    bin_mid = 0.5 * (edges[:-1] + edges[1:])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(bin_mid, point_estimate, "o-", color="darkorange", label="mean chosen FPS")
    ax1.fill_between(bin_mid, lo, hi, color="darkorange", alpha=0.25, label="95% CI (episode bootstrap)")
    ax1.set_ylabel("mean chosen FPS")
    ax1.set_title(f"FPS vs. ground-truth curvature  "
                   f"(Spearman rho={observed_rho:.3f}, permutation p={p_value:.4f})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.bar(bin_mid, decision_counts, width=(edges[1] - edges[0]) * 0.8, color="steelblue", alpha=0.7)
    ax2.set_xlabel("ground-truth |curvature| (rad/world-unit)")
    ax2.set_ylabel("# decisions")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(args.out_dir, "fps_vs_curvature.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved plot -> {out_path}")


if __name__ == "__main__":
    main()
