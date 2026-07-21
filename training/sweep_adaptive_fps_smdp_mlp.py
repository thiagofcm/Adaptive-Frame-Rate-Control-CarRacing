#!/usr/bin/env python3
"""Sweep runner for the adaptive-FPS SMDP (feedforward) experiment.

Reads a sweep config (default: config/sweep_adaptive_fps_smdp_mlp.yaml) containing
every base hyperparameter plus a `sweep:` section mapping parameter names to lists
of values. Builds the Cartesian product of the swept parameters, materializes one
full config file per combination, and launches train_adaptive_fps_smdp_mlp.py
(same directory as this script) once per combination -- each pinned to its own
isolated CPU core (via taskset) and round-robined across available GPUs.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from datetime import datetime

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
TRAIN_SCRIPT = os.path.join(SCRIPT_DIR, "train_adaptive_fps_smdp_mlp.py")
DEFAULT_SWEEP_CONFIG = os.path.join(REPO_ROOT, "config", "sweep_adaptive_fps_smdp_mlp.yaml")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=DEFAULT_SWEEP_CONFIG,
                    help="yaml file with base hyperparameters + a 'sweep:' section of param -> list of values")
    p.add_argument("--total-timesteps", type=int, default=None,
                    help="override total_timesteps for every run in the sweep (default: whatever the sweep config has)")
    p.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="GPU ids to round-robin runs across; omit for CPU-only (the default -- see detect_gpus())")
    p.add_argument("--cpus", type=int, nargs="+", default=None,
                    help="CPU core ids to pin runs to (one dedicated core per run, via taskset); "
                         "omit to auto-assign distinct cores starting at 0")
    p.add_argument("--sequential", action="store_true",
                    help="run one combination at a time instead of launching them all in parallel")
    p.add_argument("--extra", type=str, default="",
                    help="extra CLI args forwarded verbatim to train_adaptive_fps_smdp_mlp.py")
    return p.parse_args()


def load_sweep_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    sweep_params = cfg.pop("sweep", None)
    if not sweep_params:
        raise ValueError(f"No non-empty 'sweep:' section found in {path}")
    return cfg, sweep_params


def sweep_combinations(sweep_params):
    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]
    for values in itertools.product(*value_lists):
        yield dict(zip(keys, values))


def combo_tag(overrides):
    return "_".join(f"{k}{v}" for k, v in overrides.items())


def detect_gpus():
    # CPU-only by default: the Agent is a tiny feedforward MLP (no GPU benefit at
    # this size), and NavModel does batch-1 CNN inference once per physics tick,
    # where GPU kernel-launch/host<->device-copy overhead dominates over the actual
    # compute -- CPU wins both. SyncVectorEnv also runs every env sequentially
    # regardless of device, so GPU's real advantage (batching across envs) is never
    # exploited here either way. Pass --gpus explicitly (e.g. --gpus 0 1) to opt
    # back into GPU per combo.
    return [-1]


def launch(run_cfg_path, cpu, gpu, args, log_dir, tag):
    cmd = ["taskset", "-c", str(cpu), sys.executable, "-u", TRAIN_SCRIPT, "--config", run_cfg_path]
    if gpu == -1:
        cmd += ["--no-cuda"]
    if args.extra:
        cmd += args.extra.split()

    env = os.environ.copy()
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    if gpu >= 0:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    log_path = os.path.join(log_dir, f"{tag}.log")
    log_file = open(log_path, "w")
    print(f"[sweep] launching {tag} on cpu={cpu} gpu={gpu} -> {log_path}")
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return proc, log_file


def main():
    args = parse_args()
    base_cfg, sweep_params = load_sweep_config(args.config)
    combos = list(sweep_combinations(sweep_params))

    gpus = args.gpus if args.gpus is not None else detect_gpus()
    cpus = args.cpus if args.cpus is not None else list(range(len(combos)))
    if len(cpus) < len(combos):
        raise ValueError(f"{len(combos)} combinations need {len(combos)} distinct CPU cores, "
                          f"only got {len(cpus)} via --cpus; omit --cpus to auto-assign one per combo")

    date_str = datetime.now().strftime("%d-%m-%H-%M-%S")
    run_root = os.path.join(SCRIPT_DIR, "runs", f"sweep_{date_str}")
    cfg_dir = os.path.join(run_root, "configs")
    log_dir = os.path.join(run_root, "logs")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for i, overrides in enumerate(combos):
        merged = {**base_cfg, **overrides}
        if args.total_timesteps is not None:
            merged["total_timesteps"] = args.total_timesteps

        tag = combo_tag(overrides)
        run_cfg_path = os.path.join(cfg_dir, f"{tag}.yaml")
        with open(run_cfg_path, "w") as f:
            yaml.safe_dump(merged, f)

        cpu = cpus[i % len(cpus)]
        gpu = gpus[i % len(gpus)]
        proc, log_file = launch(run_cfg_path, cpu, gpu, args, log_dir, tag)
        procs.append((tag, proc, log_file))
        time.sleep(1)  # stagger run_name timestamps so directories can't collide
        if args.sequential:
            proc.wait()
            log_file.close()
            print(f"[sweep] {tag} finished (exit={proc.returncode})")

    if not args.sequential:
        for tag, proc, log_file in procs:
            proc.wait()
            log_file.close()
            print(f"[sweep] {tag} finished (exit={proc.returncode})")

    print(f"[sweep] all runs complete. configs -> {cfg_dir}, logs -> {log_dir}")
    print("[sweep] compare with: tensorboard --logdir runs/adaptive_fps_smdp_mlp")


if __name__ == "__main__":
    main()
