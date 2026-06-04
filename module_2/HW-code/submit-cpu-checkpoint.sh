#!/usr/bin/env bash
#
# Run the CPU checkpointing homework end to end.
#
# Plain bash -- NO Slurm / Kubernetes / srun / sbatch, and NO SSH inside.
# Run this directly on the CPU machine where you want training to happen.
#
# Usage:
#   ./submit-cpu-checkpoint.sh [OUT_DIR]
#
# OUT_DIR defaults to ../HW-solution so that, run from module_2/HW-code, it
# populates the directory the grader inspects. It runs training twice:
#   1) fresh, up to the checkpoint step  -> writes all three checkpoint kinds
#   2) --resume, continuing to the final step -> resumes from the vanilla ckpt
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/../HW-solution}"
TRAIN="$SCRIPT_DIR/train_cpu_checkpoint.py"

CHECKPOINT_STEP=20
TOTAL_STEPS=40
SEED=42

PYTHON="${PYTHON:-python3}"

echo ">>> output dir   : $OUT_DIR"
echo ">>> training file: $TRAIN"
echo ">>> python       : $($PYTHON --version 2>&1)"

mkdir -p "$OUT_DIR"

# Start from a clean slate so the run is reproducible.
rm -f  "$OUT_DIR/log.txt" "$OUT_DIR/metrics.json" "$OUT_DIR/final_report.json"
rm -rf "$OUT_DIR/checkpoints"

echo ">>> [1/2] fresh training to step $CHECKPOINT_STEP"
"$PYTHON" "$TRAIN" \
    --out-dir "$OUT_DIR" \
    --total-steps "$CHECKPOINT_STEP" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --seed "$SEED"

echo ">>> [2/2] resume training to step $TOTAL_STEPS"
"$PYTHON" "$TRAIN" \
    --out-dir "$OUT_DIR" \
    --total-steps "$TOTAL_STEPS" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --seed "$SEED" \
    --resume

echo ">>> [3/3] checkpoint I/O benchmark (distributed-parallel vs vanilla at scale)"
"$PYTHON" "$SCRIPT_DIR/bench_checkpoint_io.py" \
    --out "$OUT_DIR" \
    --mb "${BENCH_MB:-256}" \
    --world-size "${BENCH_WORLD:-4}" \
    --reps "${BENCH_REPS:-3}"

echo ">>> done. outputs in $OUT_DIR"
