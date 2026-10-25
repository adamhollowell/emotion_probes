#!/bin/bash
# Phase-3 orchestrator (disk-light, tmux-launched). Runs the remaining grid
# cells sequentially, validates each (shape + real geometry), STOPS on failure,
# and FREES each model cache after its cell so the model cache never fills the disk.
# Fully resumable: reduced per-emotion files survive; re-launching continues.
#
# Order: 2b-base (convert+reslice L17/L18, frees 49 GB) -> 9b-base -> 9b-it ->
# 27b-it (mode A) -> 27b-base (mode B anchor rerun, reproduction-gated).
set -u
cd "$(dirname "$0")/../.."                      # repository root
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
LOG="${LOG:-gemma_grid/extraction_logs}"; mkdir -p "$LOG"
PY="${PY:-.venv/bin/python}"
HUB="$HF_HOME/hub"

run_cell () {  # $1=cell  $2=mode  $3=model-to-free-after ("" = keep)  [extra args...]
  local cell=$1 mode=$2 model=$3; shift 3
  echo "[$(date -u +%H:%M:%S)] START $cell (mode $mode) $*" >> $LOG/ORCH.log
  $PY gemma_grid/scripts/run_cell_extraction.py --cell gemma_grid/configs/$cell.yaml --mode $mode "$@" > $LOG/$cell.log 2>&1
  local rc=$?
  echo "EXIT=$rc" > $LOG/DONE_$cell
  echo "[$(date -u +%H:%M:%S)] $cell exit $rc" >> $LOG/ORCH.log
  if [ $rc -ne 0 ]; then
    echo "ABORT at $cell: run_cell_extraction rc=$rc (mode B rc=1 => reproduction gate void)" > $LOG/ORCH_ABORTED
    exit 1
  fi
  if [ "$mode" = "A" ]; then
    $PY gemma_grid/scripts/validate_cell.py $cell >> $LOG/ORCH.log 2>&1 \
      || { echo "ABORT at $cell: validate_cell failed (shape/geometry)" > $LOG/ORCH_ABORTED; exit 1; }
  fi
  if [ -n "$model" ]; then
    rm -rf "$HUB/models--google--$model"
    echo "[$(date -u +%H:%M:%S)] freed model cache: $model" >> $LOG/ORCH.log
  fi
}

echo "[$(date -u +%H:%M:%S)] orchestrator up (tmux); disk-light + free-model-after-cell" >> $LOG/ORCH.log

run_cell gemma-2-2b-base  A  gemma-2-2b      --band 18   # convert 171 full->reduced, add L18, free 49 GB
run_cell gemma-2-9b-base  A  gemma-2-9b
run_cell gemma-2-9b-it    A  gemma-2-9b-it
run_cell gemma-2-27b-it   A  gemma-2-27b-it
run_cell gemma-2-27b-base B  ""                          # anchor rerun; re-downloads 27b, gate inside driver

echo "complete" > $LOG/ORCH_COMPLETE
echo "[$(date -u +%H:%M:%S)] ALL CELLS COMPLETE" >> $LOG/ORCH.log
