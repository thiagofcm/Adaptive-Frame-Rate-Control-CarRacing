from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def export_tensorboard(log_dir, output_dir):
    csv_output = os.path.join(output_dir, "csv")
    os.makedirs(csv_output, exist_ok=True)

    ea = EventAccumulator(log_dir)
    ea.Reload()

    # ── Export each scalar tag to its own CSV ──────────────────────────────
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        df = pd.DataFrame({
            "step":  [e.step  for e in events],
            "value": [e.value for e in events],
        })
        # e.g. "charts/episodic_return" → "charts_episodic_return.csv"
        fname = tag.replace("/", "_") + ".csv"
        df.to_csv(os.path.join(csv_output, fname), index=False)
        print(f"Saved → {fname}")

    # ── Plot all scalars ───────────────────────────────────────────────────
    tags = ea.Tags()["scalars"]
    n    = len(tags)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for i, tag in enumerate(tags):
        events = ea.Scalars(tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]
        axes[i].plot(steps, values)
        axes[i].set_title(tag)
        axes[i].set_xlabel("step")

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "all_scalars.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved → {plot_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir",        type=str,   required=True)
    args = parser.parse_args()

    log_dir    = args.logdir
    output_dir = os.path.join(log_dir,"training_plots")
    os.makedirs(output_dir, exist_ok=True)
    export_tensorboard(log_dir, output_dir)