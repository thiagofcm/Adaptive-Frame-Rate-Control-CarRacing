"""
plot_tb_csv.py
──────────────
Plots a tensorboard CSV export with smoothing.

Usage:
    python plot_tb_csv.py --csv path/to/file.csv
    python plot_tb_csv.py --csv path/to/file.csv --smooth 0.9 --title "Episodic Return"
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots


plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "figure.titlesize": 60,  # Figure title size.
    "font.size":        60,   # Default font size for all text elements unless overridden later.
    # "axes.titlesize":  60,   # Title size for axes titles (Ex: Timesteps, Reward).
    "axes.labelsize":  56,   # Controls axis label sizes: x-axis and y-axis labels (Ex: Timesteps, Reward).
    "xtick.labelsize": 50,   # Controls the size of tick numbers on the x-axis.
    "ytick.labelsize": 50,   # Controls the size of tick numbers on the y-axis.
    "legend.fontsize": 27,  
    "figure.dpi":      300, # Controls figure resolution (Dots Per Inch).
})

def smooth(values, weight=0.9):
    """Exponential moving average smoothing (like tensorboard)."""
    smoothed = []
    last = values[0]
    for v in values:
        last = last * weight + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",    type=str,   required=True,  help="Path to tensorboard CSV file")
    parser.add_argument("--smooth", type=float, default=0.6,    help="Smoothing weight (0=no smooth, 0.99=heavy)")
    parser.add_argument("--title",  type=str,   default=None,   help="Plot title (default: csv filename)")
    parser.add_argument("--output", type=str,   default=None,   help="Output path for saved plot (default: same dir as csv)")
    parser.add_argument("--xlabel", type=str,   default="TimeStep", help="X axis label")
    parser.add_argument("--ylabel", type=str,   default="Ep Reward", help="Y axis label")
    args = parser.parse_args()

    # ── Load CSV ───────────────────────────────
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows from {args.csv}")
    print(f"Columns: {list(df.columns)}")

    # handle both 'Step' and 'step' column names
    step_col  = next((c for c in df.columns if c.lower() == "step"),  None)
    value_col = next((c for c in df.columns if c.lower() == "value"), None)

    if step_col is None or value_col is None:
        raise ValueError(f"Expected 'Step' and 'Value' columns, got: {list(df.columns)}")

    steps  = df[step_col].values
    values = df[value_col].values

    # ── Smooth ────────────────────────────────
    smoothed = smooth(values, weight=args.smooth)

    # ── Plot ──────────────────────────────────
    title = args.title or os.path.splitext(os.path.basename(args.csv))[0]

    plt.figure(figsize=(18, 15))
    plt.plot(steps, values,   alpha=0.3, color="steelblue", linewidth=1.0, label="Raw")
    plt.plot(steps, smoothed, alpha=1.0, color="steelblue", linewidth=2.0, label=f"Smoothed (w={args.smooth})")

    plt.axhline(y=0,    color="gray", linestyle="--", alpha=0.4)
    #plt.axhline(y=200,  color="green", linestyle="--", alpha=0.4, label="Target (200)")
    #plt.axhline(y=-100, color="red",   linestyle="--", alpha=0.4, label="Crash (-100)")

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(title)
    #plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )
    plt.tight_layout()

    # ── Save ──────────────────────────────────
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(args.csv)[0]
        out_path = f"{base}_plot.png"

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.close()


if __name__ == "__main__":
    main()