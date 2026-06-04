#!/usr/bin/env python3
"""Plot the checkpoint I/O sweep(s) produced by ``bench_checkpoint_io.py --sweep``.

Renders ``checkpoint_io_benchmark.png`` with two panels:
  * left  -- blocking save time vs checkpoint size on networked storage
             (vanilla vs distributed vs async);
  * right -- distributed speedup over vanilla vs size, for networked and (if
             given) local storage, with the break-even line at 1.0. This shows
             *when* distributed beats vanilla and how it depends on the storage.

Needs matplotlib (``pip install matplotlib``); the grader/CI never run this.

Usage (from module_2/HW-code):
    python plot_checkpoint_io.py \
        --in    ../HW-solution/checkpoint_io_sweep.json \
        --local ../HW-solution/checkpoint_io_sweep_local.json \
        --out   ../HW-solution/checkpoint_io_benchmark.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        data = json.load(f)
    data["points"] = sorted(data["points"], key=lambda p: p["checkpoint_mb"])
    return data


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sol = os.path.join(here, "..", "HW-solution")
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(sol, "checkpoint_io_sweep.json"),
                    help="sweep json on networked/parallel storage")
    ap.add_argument("--local", dest="local", default="",
                    help="optional sweep json on local storage (for comparison)")
    ap.add_argument("--out", dest="out", default=os.path.join(sol, "checkpoint_io_benchmark.png"))
    args = ap.parse_args()

    net = load(args.inp)
    loc = load(args.local) if args.local and os.path.exists(args.local) else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- panel 1: absolute times on networked storage ----------------------
    p = net["points"]
    mb = [x["checkpoint_mb"] for x in p]
    ax1.plot(mb, [x["vanilla_ms"] for x in p], "o-", color="#d62728",
             label="vanilla (1 sequential writer)")
    ax1.plot(mb, [x["distributed_parallel_ms"] for x in p], "s-", color="#1f77b4",
             label=f"distributed ({net['world_size']} parallel writers)")
    ax1.plot(mb, [x["async_blocking_ms"] for x in p], "^--", color="#2ca02c",
             label="async (blocking time only)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(mb)
    ax1.set_xticklabels([str(m) for m in mb])
    ax1.set_xlabel("checkpoint size (MB)")
    ax1.set_ylabel("save time blocking training (ms)")
    ax1.set_title(f"Save time vs size ({net.get('scratch_fs', '?')}, networked)")
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)
    ax1.legend()

    # ---- panel 2: distributed speedup vs vanilla, by storage ----------------
    def speedups(d):
        return ([x["checkpoint_mb"] for x in d["points"]],
                [x["distributed_speedup_vs_vanilla"] for x in d["points"]])

    x, s = speedups(net)
    ax2.plot(x, s, "s-", color="#1f77b4", label=f"networked ({net.get('scratch_fs', '?')})")
    if loc is not None:
        xl, sl = speedups(loc)
        ax2.plot(xl, sl, "o--", color="#ff7f0e", label=f"local ({loc.get('scratch_fs', '?')})")
    ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax2.text(mb[0], 1.02, "break-even (distributed = vanilla)", fontsize=8, color="gray")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(mb)
    ax2.set_xticklabels([str(m) for m in mb])
    ax2.set_xlabel("checkpoint size (MB)")
    ax2.set_ylabel("speedup: vanilla_ms / distributed_ms  (>1 = distributed wins)")
    ax2.set_title(f"When distributed beats vanilla (world_size={net['world_size']})")
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    ax2.legend()

    fig.suptitle("CPU checkpointing: distributed (sharded, parallel) vs vanilla", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
