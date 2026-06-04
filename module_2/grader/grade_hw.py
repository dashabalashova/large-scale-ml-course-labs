#!/usr/bin/env python3
"""Grader for the module_2 CPU checkpointing homework.

Stdlib only -- this intentionally does NOT import torch, does NOT run any
training, and does NOT touch any cluster. It only inspects the files a student
committed under ``module_2/HW-solution/`` and decides pass/fail.

Run from the repository root:
    python module_2/grader/grade_hw.py

Exit code 0 == all checks pass, non-zero == at least one failure.
"""
import glob
import json
import os
import sys

SOL = os.path.join("module_2", "HW-solution")
LOG_MARKERS = [
    "START_TRAINING",
    "SAVED_CHECKPOINT",
    "CHECKPOINT_TIMING",
    "RESUMED_FROM_CHECKPOINT",
    "FINISHED_TRAINING",
]

failures = []
passes = []


def ok(msg):
    passes.append(msg)
    print(f"OK:   {msg}")


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}")


def load_json(path):
    with open(path) as f:
        return json.load(f)


def is_number(x):
    # bool is a subclass of int -- reject it explicitly.
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    # ---- required files exist -------------------------------------------- #
    required = [
        "train_cpu_checkpoint.py",
        "log.txt",
        "metrics.json",
        "final_report.json",
    ]
    for name in required:
        path = os.path.join(SOL, name)
        if os.path.isfile(path):
            ok(f"required file present: {path}")
        else:
            fail(f"missing required file: {path}")

    # ---- solution must not contain placeholders -------------------------- #
    train_path = os.path.join(SOL, "train_cpu_checkpoint.py")
    if os.path.isfile(train_path):
        src = open(train_path, encoding="utf-8").read()
        if "YOUR CODE HERE" in src:
            fail("train_cpu_checkpoint.py still contains 'YOUR CODE HERE'")
        elif "NotImplementedError" in src:
            fail("train_cpu_checkpoint.py still raises NotImplementedError")
        else:
            ok("train_cpu_checkpoint.py has no leftover placeholders")

    # ---- metrics.json ---------------------------------------------------- #
    metrics_path = os.path.join(SOL, "metrics.json")
    if os.path.isfile(metrics_path):
        try:
            metrics = load_json(metrics_path)
        except (json.JSONDecodeError, ValueError) as exc:
            fail(f"metrics.json is not valid JSON: {exc}")
            metrics = None
        if metrics is not None:
            losses = metrics.get("losses")
            if isinstance(losses, list) and losses and all(is_number(v) for v in losses):
                ok(f"metrics.json has {len(losses)} numeric losses")
            else:
                fail("metrics.json 'losses' must be a non-empty list of numbers")
            for key in ("start_step", "end_step", "resumed_from_checkpoint"):
                if key in metrics:
                    ok(f"metrics.json has '{key}'")
                else:
                    fail(f"metrics.json missing '{key}'")

            timings = metrics.get("checkpoint_timings")
            tfields = ("vanilla_sec", "async_blocking_sec", "distributed_sec")
            if (isinstance(timings, list) and timings
                    and all(isinstance(t, dict) and all(is_number(t.get(f)) for f in tfields)
                            for t in timings)):
                ok(f"metrics.json has timing comparison for {len(timings)} checkpoint(s)")
            else:
                fail("metrics.json 'checkpoint_timings' must list per-step "
                     "{vanilla_sec, async_blocking_sec, distributed_sec} numbers")

    # ---- final_report.json ----------------------------------------------- #
    report_path = os.path.join(SOL, "final_report.json")
    report = None
    if os.path.isfile(report_path):
        try:
            report = load_json(report_path)
        except (json.JSONDecodeError, ValueError) as exc:
            fail(f"final_report.json is not valid JSON: {exc}")

    if report is not None:
        if report.get("status") == "ok":
            ok("final_report status == 'ok'")
        else:
            fail(f"final_report status != 'ok' (got {report.get('status')!r})")

        if report.get("resumed_from_checkpoint") is True:
            ok("final_report resumed_from_checkpoint is True")
        else:
            fail("final_report resumed_from_checkpoint must be true")

        total = report.get("total_steps")
        if isinstance(total, int) and not isinstance(total, bool) and total >= 30:
            ok(f"final_report total_steps == {total} (>= 30)")
        else:
            fail(f"final_report total_steps must be an int >= 30 (got {total!r})")

        if is_number(report.get("final_loss")):
            ok(f"final_report final_loss is numeric ({report.get('final_loss')})")
        else:
            fail(f"final_report final_loss must be a number (got {report.get('final_loss')!r})")

        summary = report.get("checkpoint_timing_summary")
        sfields = ("vanilla_avg_sec", "async_blocking_avg_sec", "distributed_avg_sec")
        if (isinstance(summary, dict)
                and all(is_number(summary.get(f)) for f in sfields)
                and isinstance(summary.get("least_blocking_mode"), str)
                and summary.get("least_blocking_mode")):
            ok(f"final_report has checkpoint timing summary "
               f"(least_blocking={summary['least_blocking_mode']})")
        else:
            fail("final_report 'checkpoint_timing_summary' must have numeric "
                 "vanilla/async/distributed averages + least_blocking_mode")

        ckpt_path = report.get("checkpoint_path")
        if isinstance(ckpt_path, str) and ckpt_path:
            ok(f"final_report checkpoint_path set: {ckpt_path}")
            if os.path.isfile(ckpt_path) and os.path.getsize(ckpt_path) > 0:
                ok(f"checkpoint_path points to a real non-empty file")
            else:
                fail(f"checkpoint_path file missing or empty: {ckpt_path}")
        else:
            fail("final_report checkpoint_path must be a non-empty string")

    # ---- checkpoints: vanilla / async / distributed ---------------------- #
    vanilla = glob.glob(os.path.join(SOL, "checkpoints", "vanilla", "*.pt"))
    if vanilla:
        ok(f"found {len(vanilla)} vanilla checkpoint(s)")
    else:
        fail("no vanilla checkpoints under checkpoints/vanilla/*.pt")

    async_ck = glob.glob(os.path.join(SOL, "checkpoints", "async", "*.pt"))
    if async_ck:
        ok(f"found {len(async_ck)} async checkpoint(s)")
    else:
        fail("no async checkpoints under checkpoints/async/*.pt")

    dist_steps = glob.glob(os.path.join(SOL, "checkpoints", "distributed", "step_*"))
    if not dist_steps:
        fail("no distributed checkpoints under checkpoints/distributed/step_*")
    else:
        good = 0
        for step_dir in dist_steps:
            shards = glob.glob(os.path.join(step_dir, "shard_*.pt"))
            meta = os.path.join(step_dir, "meta.json")
            if shards and os.path.isfile(meta):
                try:
                    load_json(meta)
                    good += 1
                except (json.JSONDecodeError, ValueError):
                    fail(f"distributed meta.json invalid in {step_dir}")
            else:
                fail(f"distributed step dir incomplete (need shards + meta.json): {step_dir}")
        if good:
            ok(f"found {good} valid distributed checkpoint step(s)")

    # ---- log markers ----------------------------------------------------- #
    log_path = os.path.join(SOL, "log.txt")
    if os.path.isfile(log_path):
        log_text = open(log_path, encoding="utf-8").read()
        for marker in LOG_MARKERS:
            if marker in log_text:
                ok(f"log.txt contains marker {marker}")
            else:
                fail(f"log.txt missing marker {marker}")

    # ---- verdict --------------------------------------------------------- #
    print("-" * 60)
    print(f"{len(passes)} checks passed, {len(failures)} failed")
    if failures:
        print("RESULT: FAIL")
        sys.exit(1)
    print("RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
