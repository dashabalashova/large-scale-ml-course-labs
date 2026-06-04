#!/usr/bin/env python3
"""Checkpoint I/O benchmark: when does distributed beat vanilla?

The training homework uses a *tiny* model, where a single ``vanilla``
``torch.save`` is fastest -- sharding only adds per-file overhead. This script
shows the crossover point: with a *large* checkpoint written to networked /
parallel storage, writing the shards **concurrently** (one writer per rank,
as ``save_distributed_checkpoint`` does) beats a single sequential writer,
because aggregate write bandwidth scales with the number of writers.

It reuses the three save strategies from ``train_cpu_checkpoint.py`` on a large
synthetic state, writes ``checkpoint_io_benchmark.json`` next to the other
outputs, and deletes the (large) scratch files afterwards. You do not need to
edit this file.

Run (after activating the env), from module_2/HW-code:
    python bench_checkpoint_io.py --out ../HW-solution
"""
import argparse
import json
import os
import shutil
import statistics
import time

import torch

from train_cpu_checkpoint import (
    save_async_checkpoint,
    save_distributed_checkpoint,
    save_vanilla_checkpoint,
)


def build_state(total_mb, num_tensors):
    """A synthetic 'model' state of ~total_mb split across num_tensors params."""
    per_elems = max(1, (total_mb * 1024 * 1024) // (4 * num_tensors))  # float32
    model = {f"layer_{i}.weight": torch.randn(per_elems) for i in range(num_tensors)}
    return {"step": 0, "model": model}


def fs_type(path):
    """Best-effort filesystem type for the mount containing ``path`` (Linux)."""
    try:
        path = os.path.realpath(path)
        best_mnt, best_type = "", "unknown"
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and path.startswith(parts[1]) and len(parts[1]) >= len(best_mnt):
                    best_mnt, best_type = parts[1], parts[2]
        return best_type
    except OSError:
        return "unknown"


def median_ms(fn, reps):
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def median_async_blocking_ms(state, async_dir, reps):
    """Time only the part of the async save that blocks the caller."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _, thread = save_async_checkpoint(state, async_dir, 0)
        samples.append((time.perf_counter() - t0) * 1e3)
        thread.join()
    return statistics.median(samples)


def measure_point(mb, world_size, reps, scratch):
    """Time the three strategies for a ``mb``-sized checkpoint; return a dict."""
    num_tensors = world_size * 2
    state = build_state(mb, num_tensors)
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch, exist_ok=True)
    vanilla_dir = os.path.join(scratch, "vanilla")
    async_dir = os.path.join(scratch, "async")
    dist_dir = os.path.join(scratch, "distributed")

    vanilla_ms = median_ms(lambda: save_vanilla_checkpoint(state, vanilla_dir, 0), reps)
    async_ms = median_async_blocking_ms(state, async_dir, reps)
    dist_ms = median_ms(
        lambda: save_distributed_checkpoint(state, dist_dir, 0, world_size), reps
    )
    shutil.rmtree(scratch, ignore_errors=True)

    return {
        "checkpoint_mb": mb,
        "world_size": world_size,
        "num_tensors": num_tensors,
        "reps": reps,
        "vanilla_ms": round(vanilla_ms, 3),
        "async_blocking_ms": round(async_ms, 3),
        "distributed_parallel_ms": round(dist_ms, 3),
        "distributed_speedup_vs_vanilla": round(vanilla_ms / dist_ms, 3) if dist_ms else None,
        "distributed_beats_vanilla": dist_ms < vanilla_ms,
    }


def main():
    ap = argparse.ArgumentParser(description="checkpoint I/O benchmark")
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=os.path.join(here, "..", "HW-solution"),
                    help="where to write the json result(s)")
    ap.add_argument("--scratch", default=os.path.expanduser("~/.ckpt_io_bench"),
                    help="scratch dir for the large checkpoints (deleted afterwards)")
    ap.add_argument("--mb", type=int, default=256, help="total checkpoint size, MB")
    ap.add_argument("--world-size", type=int, default=4, help="number of shards/ranks")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sweep", default="",
                    help="comma-separated MB sizes; runs a sweep and writes "
                         "checkpoint_io_sweep.json (for plotting the crossover)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fs = fs_type(args.scratch)

    if args.sweep:
        sizes = [int(s) for s in args.sweep.split(",") if s.strip()]
        points = []
        for mb in sizes:
            pt = measure_point(mb, args.world_size, args.reps, args.scratch)
            points.append(pt)
            print(f"  {mb:5d}MB  vanilla={pt['vanilla_ms']:8.1f}ms  "
                  f"distributed={pt['distributed_parallel_ms']:8.1f}ms  "
                  f"speedup={pt['distributed_speedup_vs_vanilla']}x  "
                  f"dist_wins={pt['distributed_beats_vanilla']}")
        result = {"world_size": args.world_size, "reps": args.reps,
                  "scratch_fs": fs, "points": points}
        out_path = os.path.join(args.out, "checkpoint_io_sweep.json")
    else:
        result = measure_point(args.mb, args.world_size, args.reps, args.scratch)
        result["scratch_fs"] = fs
        out_path = os.path.join(args.out, "checkpoint_io_benchmark.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
