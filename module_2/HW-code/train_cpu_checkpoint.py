#!/usr/bin/env python3
"""CPU checkpointing homework -- STARTER.

Fill in the four functions marked with ``# YOUR CODE HERE`` below
(save_vanilla_checkpoint, save_async_checkpoint, save_distributed_checkpoint,
load_checkpoint). Everything else already works.

Trains a tiny MLP on a synthetic regression dataset on CPU and exercises three
checkpoint strategies, then demonstrates resume-from-checkpoint:

  * vanilla         -- a single synchronous ``torch.save`` of the full state.
  * async           -- the same state, but written from a background thread so
                       training is not blocked by disk I/O.
  * distributed     -- the model parameters are sharded round-robin across a
                       few fake "ranks"; each shard is a separate file plus a
                       ``meta.json`` describing the layout.

Outputs written into ``--out-dir``:
  * log.txt          -- human readable log with machine-readable markers
  * metrics.json     -- per-step losses + bookkeeping
  * final_report.json-- summary consumed by the grader
  * checkpoints/      -- vanilla/, async/, distributed/ checkpoints

The script is run twice by ``submit-cpu-checkpoint.sh``: once fresh up to the
checkpoint step, then again with ``--resume`` to continue to the final step.
"""
import argparse
import copy
import json
import os
import threading
import time

import torch
import torch.nn as nn

DEVICE = "cpu"


def log(log_path, msg):
    """Append a timestamped line to the log file and echo it to stdout."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(log_path, "a") as f:
        f.write(line + "\n")


def make_dataset(n_samples, in_dim, seed):
    """Deterministic synthetic linear-ish regression dataset."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_samples, in_dim, generator=g)
    true_w = torch.randn(in_dim, 1, generator=g)
    y = X @ true_w + 0.1 * torch.randn(n_samples, 1, generator=g)
    return X, y


class TinyMLP(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# Checkpoint strategies (these are the functions students implement).
# --------------------------------------------------------------------------- #
def save_vanilla_checkpoint(state, ckpt_dir, step):
    """Save ``state`` to ``<ckpt_dir>/vanilla_step_<step>.pt`` and return the path."""
    # YOUR CODE HERE
    raise NotImplementedError


def save_async_checkpoint(state, ckpt_dir, step):
    """Write ``state`` to ``<ckpt_dir>/async_step_<step>.pt`` from a background
    thread; return ``(path, thread)`` so the caller can ``join`` it."""
    # YOUR CODE HERE
    raise NotImplementedError


def save_distributed_checkpoint(state, ckpt_dir, step, world_size=2):
    """Shard ``state["model"]`` round-robin into ``<ckpt_dir>/step_<step>/shard_<rank>.pt``
    plus a ``meta.json``; return the per-step dir path. Write the shards
    concurrently (one thread per rank) -- that parallel write is what lets
    sharded checkpointing beat vanilla at scale (see bench_checkpoint_io.py)."""
    # YOUR CODE HERE
    raise NotImplementedError


def load_checkpoint(path):
    """Load and return the checkpoint saved at ``path``."""
    # YOUR CODE HERE
    raise NotImplementedError


# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(description="CPU checkpointing homework")
    ap.add_argument("--out-dir", required=True, help="where to write outputs")
    ap.add_argument("--total-steps", type=int, default=40)
    ap.add_argument("--checkpoint-step", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="resume from the vanilla checkpoint at --checkpoint-step")
    ap.add_argument("--report-prefix", default="module_2/HW-solution",
                    help="repo-relative prefix recorded in final_report.json")
    return ap.parse_args()


def main():
    args = parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    ckpt_root = os.path.join(out, "checkpoints")
    vanilla_dir = os.path.join(ckpt_root, "vanilla")
    async_dir = os.path.join(ckpt_root, "async")
    dist_dir = os.path.join(ckpt_root, "distributed")
    log_path = os.path.join(out, "log.txt")
    resume_ckpt = os.path.join(vanilla_dir, f"vanilla_step_{args.checkpoint_step}.pt")

    in_dim, hidden = 16, 32
    torch.manual_seed(args.seed)
    X, y = make_dataset(512, in_dim, args.seed)
    model = TinyMLP(in_dim, hidden)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    start_step = 0
    resumed = False
    losses = []
    pending_threads = []

    if args.resume:
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"cannot resume, missing checkpoint: {resume_ckpt}")
        ckpt = load_checkpoint(resume_ckpt)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"])
        resumed = True
        log(log_path, f"RESUMED_FROM_CHECKPOINT step={start_step} path={resume_ckpt}")
    else:
        log(log_path, f"START_TRAINING seed={args.seed} total_steps={args.total_steps}")

    checkpoint_at = {args.checkpoint_step, args.total_steps}
    timings = []
    for step in range(start_step + 1, args.total_steps + 1):
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        loss_val = float(loss.item())
        losses.append(loss_val)
        if step % 5 == 0 or step == args.total_steps:
            log(log_path, f"step={step} loss={loss_val:.6f}")

        if step in checkpoint_at:
            state = {
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            # Time how long each strategy blocks the training loop. For async
            # this is only the snapshot + thread spawn (the disk write happens
            # off-thread), which is what makes it interesting to compare.
            t0 = time.perf_counter()
            v_path = save_vanilla_checkpoint(state, vanilla_dir, step)
            vanilla_sec = time.perf_counter() - t0

            t0 = time.perf_counter()
            a_path, a_thread = save_async_checkpoint(state, async_dir, step)
            async_blocking_sec = time.perf_counter() - t0
            pending_threads.append(a_thread)

            t0 = time.perf_counter()
            d_path = save_distributed_checkpoint(state, dist_dir, step)
            distributed_sec = time.perf_counter() - t0

            timings.append({
                "step": step,
                "vanilla_sec": vanilla_sec,
                "async_blocking_sec": async_blocking_sec,
                "distributed_sec": distributed_sec,
            })
            log(log_path,
                f"SAVED_CHECKPOINT step={step} vanilla={v_path} "
                f"async={a_path} distributed={d_path}")
            log(log_path,
                f"CHECKPOINT_TIMING step={step} "
                f"vanilla={vanilla_sec * 1e3:.3f}ms "
                f"async_blocking={async_blocking_sec * 1e3:.3f}ms "
                f"distributed={distributed_sec * 1e3:.3f}ms")

    for thread in pending_threads:
        thread.join()

    def _avg(key):
        return sum(t[key] for t in timings) / len(timings) if timings else 0.0

    timing_summary = {
        "vanilla_avg_sec": _avg("vanilla_sec"),
        "async_blocking_avg_sec": _avg("async_blocking_sec"),
        "distributed_avg_sec": _avg("distributed_sec"),
    }
    timing_summary["least_blocking_mode"] = min(
        (("vanilla", timing_summary["vanilla_avg_sec"]),
         ("async", timing_summary["async_blocking_avg_sec"]),
         ("distributed", timing_summary["distributed_avg_sec"])),
        key=lambda kv: kv[1],
    )[0]
    log(log_path,
        f"CHECKPOINT_TIMING_SUMMARY "
        f"vanilla_avg={timing_summary['vanilla_avg_sec'] * 1e3:.3f}ms "
        f"async_blocking_avg={timing_summary['async_blocking_avg_sec'] * 1e3:.3f}ms "
        f"distributed_avg={timing_summary['distributed_avg_sec'] * 1e3:.3f}ms "
        f"least_blocking={timing_summary['least_blocking_mode']}")

    final_step = args.total_steps
    metrics = {
        "losses": losses,
        "start_step": start_step,
        "end_step": final_step,
        "resumed_from_checkpoint": resumed,
        "checkpoint_timings": timings,
    }
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    checkpoint_path = os.path.join(
        args.report_prefix, "checkpoints", "vanilla", f"vanilla_step_{final_step}.pt"
    )
    report = {
        "status": "ok",
        "resumed_from_checkpoint": resumed,
        "total_steps": final_step,
        "final_loss": losses[-1],
        "checkpoint_path": checkpoint_path,
        "checkpoint_modes": ["vanilla", "async", "distributed"],
        "checkpoint_timing_summary": timing_summary,
    }
    with open(os.path.join(out, "final_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    log(log_path, f"FINISHED_TRAINING final_loss={losses[-1]:.6f} total_steps={final_step}")


if __name__ == "__main__":
    main()
