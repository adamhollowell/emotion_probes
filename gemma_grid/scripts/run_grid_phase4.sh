#!/usr/bin/env bash
# =============================================================================
# run_grid_phase4.sh  The GGE1 analysis pipeline, in order, with its gates.
#
# Recomputes the cross-model geometry of the paper on a CPU from the released
# vectors and neutral activations. Each step calls a released script. The
# pipeline stops at the first failure and logs every step.
#
# PLACEMENT: this file lives at gemma_grid/scripts/ inside the replication
# study's repository. REPO is derived from its location and checked.
#
# ENVIRONMENT
#   Interpreter: $PYBIN (default: python3 on PATH).
#   Pins, checked before any step runs: numpy 2.0.2, scikit-learn 1.6.1, the
#   versions of the analysis environment in RELEASE_MANIFEST.md. The pipeline
#   stops if either differs.
#   Step 1 also needs transformers and a Hugging Face login with the Gemma
#   license accepted (it reads the Gemma 2 tokenizer).
#
# INPUTS (checked by the preflight; see README.md, "Reproducing the paper")
#   data/stories/neutral_dialogues.jsonl        the neutral-dialogue corpus,
#                                               sha256 3838a8f2...
#   data/vectors/emotion_vectors_171.pt         the anchor vectors, base 27B,
#                                               sha256 6de2e293...
#   data/vectors/lens_ingredients.pt            regenerated with the replication
#                                               study's scripts/export_lens_ingredients.py
#   data/ratings/                               the rating norms (README.md)
#   gemma_grid/data/<model>/vectors/            from the Hugging Face folder
#     canonical.pt                  five models other than base 27B
#     <model>__neutral.pt           all six models
#     <model>__per_story_slice.pt   all six models
#     <model>__lens_ingredients.pt  five models other than base 27B, regenerated
#                                   with gemma_grid/scripts/export_lens_ingredients_grid.py
#   BOOT_SCRATCH                    an explicit scratch directory, preferably
#                                   empty (several steps keep resumable state there)
#
# RUN ORDER (step numbers follow the run of record; its step 4 was a check
# against a working file and is omitted here)
#    1  c3_tokenize_neutral.py      token ids of the neutral corpus
#    2  run_rebase.py               cleanup bases and cleaned vectors, all six
#                                   models; then the ANCHOR GATE: the base 27B
#                                   re-clean must match emotion_vectors_171.pt at
#                                   layer 31 to rel <= 1e-6, or the pipeline stops
#    3  run_stageA.py + phase4_stageA_crosstable.py
#    5  run_stageB.py
#    6  run_bootstrap.py            per model
#    7  run_stageC_grade.py
#   7b  phase4_lens_preflight.py    anchor readout assets into BOOT_SCRATCH
#    8  run_c3.py
#   8b  run_c3_savevecs.py
#    9  run_c3_grade.py
#   10  run_lens.py per model + phase4_lens_crosstable.py
#   11  run_extracts.py
#   12  phase4_layerstab_ci.py
#   13  seam_gap_metric.py
#   14  op1_spectrum_measures.py
#   15  norm_robustness_full.py     Table 4 and Appendix E
#   16  fixed_probe_tuning_shifts.py
#17-19  zhang_z1.py, zhang_z2.py, zhang_z3.py
#
# OUTPUTS are written in place, into gemma_grid/geometry_out/ and
# gemma_grid/data/<model>/vectors/. Before step 1 the pipeline records the
# sha256 of every file there. Where a script skips work whose output already
# exists (Stage A summaries, c3_vectors.pt), the existing file is moved under
# the run directory first, so the step recomputes it. At the end the pipeline
# hashes the outputs again and lists which files changed. Files that record a
# run time or a script path change on every run; the values in them should not.
#
# USAGE
#   BOOT_SCRATCH=/path/to/scratch bash gemma_grid/scripts/run_grid_phase4.sh [flags]
#     --dry-run          check inputs and pins, report, run nothing
#     --resume           skip steps whose primary output exists, and set nothing aside
#     --stop-after N     stop after step N (also --stop-after=N)
#     --self-test        test the anchor gate's failure paths on fixtures and exit
#   RUN_DIR=...          where logs and hashes go (default gemma_grid/run_logs)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
[ -d "$REPO/gemma_grid/scripts" ] || { echo "FATAL: $REPO/gemma_grid/scripts not found; this script must live at gemma_grid/scripts/"; exit 1; }
GG="$REPO/gemma_grid"
OUT_TREE="$GG/geometry_out"
RUN_DIR="${RUN_DIR:-$GG/run_logs}"
CELLS=(gemma-2-2b-base gemma-2-2b-it gemma-2-9b-base gemma-2-9b-it gemma-2-27b-base gemma-2-27b-it)
ANCHOR=gemma-2-27b-base
CORPUS_SHA=3838a8f23e726d2433135336477ab0ee0e21f4ecb8ca06613963d26de8a3a04f
ANCHOR_SHA=6de2e293d0e642bd164111e3be24917d7460bd4810fffa6c710cd1a97235971b

DRY=0; RESUME=0; SELF_TEST=0; STOP_AFTER=99
expect_stop=0
for a in "$@"; do
  if [ "$expect_stop" -eq 1 ]; then STOP_AFTER="$a"; expect_stop=0; continue; fi
  case "$a" in
    --dry-run) DRY=1;; --resume) RESUME=1;; --self-test) SELF_TEST=1;;
    --stop-after) expect_stop=1;; --stop-after=*) STOP_AFTER="${a#*=}";;
    *) echo "unknown flag: $a"; exit 2;; esac
done
[ "$expect_stop" -eq 0 ] || { echo "FATAL: --stop-after given without a value"; exit 2; }
case "$STOP_AFTER" in (*[!0-9]*|"") echo "FATAL: --stop-after value must be an integer"; exit 2;; esac

PYBIN="${PYBIN:-python3}"

sha256_of() {  # portable: shasum on macOS, sha256sum on Linux
  if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$@"; else sha256sum "$@"; fi
}

# ---- logging (the self-test logs to a temporary directory) -------------------
if [ "$SELF_TEST" -eq 1 ]; then LOGDIR="$(mktemp -d)"; else LOGDIR="$RUN_DIR"; fi
mkdir -p "$LOGDIR/logs"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGDIR/logs/run.log"; }
die() { log "FATAL: $*"; exit 1; }

# ---- the anchor gate (tested by --self-test) --------------------------------
anchor_gate() {  # $1 = rebase log. Stops unless the [anchor] rel line exists and rel <= 1e-6.
  local logf="$1"
  [ -s "$logf" ] || die "anchor gate: rebase log missing or empty: $logf"
  "$PYBIN" - "$logf" <<'PYEOF' || die "anchor gate FAILED (see the message above): the base 27B re-clean must match emotion_vectors_171.pt to rel <= 1e-6"
import re, sys
txt = open(sys.argv[1]).read()
m = re.search(r"\[anchor\].*?rel ([0-9.eE+\-]+)", txt)
if not m:
    sys.exit("no [anchor] rel line in the rebase log; the cross-check did not run")
rel = float(m.group(1))
k = re.search(r"k@L31=(\d+)", txt)
print(f"anchor gate: rel={rel:.2e}" + (f", k@L31={k.group(1)}" if k else ", no k@L31 line"))
if rel > 1e-6:
    sys.exit(f"rel {rel:.2e} exceeds the 1e-6 bound")
PYEOF
  log "anchor gate: PASS (rel <= 1e-6)"
}

if [ "$SELF_TEST" -eq 1 ]; then
  PYBIN="${PYBIN_TEST:-python3}"   # the gate needs only the standard library
  T="$(mktemp -d)"; fails=0
  expect_die()  { if ( "$@" ) >/dev/null 2>&1; then echo "SELF-TEST FAIL (should have stopped): $*"; fails=1; else echo "  ok (stopped as required): $*"; fi; }
  expect_pass() { if ( "$@" ) >/dev/null 2>&1; then echo "  ok (passed as required): $*"; else echo "SELF-TEST FAIL (should have passed): $*"; fails=1; fi; }
  echo "self-test: anchor_gate"
  printf 'other lines only\n' > "$T/log_norel.txt"
  printf '[anchor] re-clean vs emotion_vectors_171 @L31: rel 1.20e-02 | k@L31=24\n' > "$T/log_bigrel.txt"
  printf '[anchor] re-clean vs emotion_vectors_171 @L31: rel 7.34e-07 | k@L31=24\n' > "$T/log_good.txt"
  expect_die  anchor_gate "$T/log_missing.txt"
  expect_die  anchor_gate "$T/log_norel.txt"
  expect_die  anchor_gate "$T/log_bigrel.txt"
  expect_pass anchor_gate "$T/log_good.txt"
  rm -rf "$T" "$LOGDIR"
  [ "$fails" -eq 0 ] && { echo "SELF-TEST: ALL PASS"; exit 0; } || { echo "SELF-TEST: FAILURES"; exit 1; }
fi

# ---------------- Phase 0: preflight -----------------------------------------
log "Phase 0: preflight (REPO=$REPO)"
req=(
  "data/stories/neutral_dialogues.jsonl"
  "data/vectors/emotion_vectors_171.pt"
  "data/vectors/lens_ingredients.pt"
)
for c in "${CELLS[@]}"; do
  req+=("gemma_grid/data/$c/vectors/${c}__neutral.pt" "gemma_grid/data/$c/vectors/${c}__per_story_slice.pt")
  if [ "$c" != "$ANCHOR" ]; then
    req+=("gemma_grid/data/$c/vectors/canonical.pt" "gemma_grid/data/$c/vectors/${c}__lens_ingredients.pt")
  fi
done
missing=0
for f in "${req[@]}"; do
  if [ ! -s "$REPO/$f" ]; then log "  MISSING: $f"; missing=1; else log "  ok: $f"; fi
done
[ -d "$REPO/data/ratings" ] || { log "  MISSING: data/ratings/ (README.md, rating norms)"; missing=1; }
if [ "$missing" -eq 1 ]; then
  log "  The README's 'Reproducing the paper' section says where each input comes from."
  [ "$DRY" -eq 1 ] && log "(dry-run: continuing despite missing inputs)" || die "preflight failed; see the MISSING lines"
fi

check_hash() {  # $1 = path under REPO, $2 = expected sha256
  [ -s "$REPO/$1" ] || return 0
  local got; got="$(sha256_of "$REPO/$1" | awk '{print $1}')"
  if [ "$got" = "$2" ]; then log "  hash ok: $1"
  else
    [ "$DRY" -eq 1 ] && log "  HASH MISMATCH (dry-run, not fatal): $1 is $got, expected $2" \
                     || die "$1 has sha256 $got, expected $2"
  fi
}
check_hash "data/stories/neutral_dialogues.jsonl" "$CORPUS_SHA"
check_hash "data/vectors/emotion_vectors_171.pt" "$ANCHOR_SHA"

if [ -z "${BOOT_SCRATCH:-}" ]; then
  [ "$DRY" -eq 1 ] && log "(dry-run: BOOT_SCRATCH unset)" || die "set BOOT_SCRATCH to a scratch directory, preferably empty"
else
  mkdir -p "$BOOT_SCRATCH"
  if [ "$RESUME" -eq 0 ] && [ -n "$(ls -A "$BOOT_SCRATCH" 2>/dev/null)" ]; then
    log "  NOTE: BOOT_SCRATCH is not empty; steps that keep state there will resume from it"
  fi
fi
export BOOT_SCRATCH="${BOOT_SCRATCH:-}"
export LENS_ASSET_DIR="${BOOT_SCRATCH:-}"   # phase4_lens.py reads the anchor readout assets from here

# environment pins
command -v "$PYBIN" >/dev/null 2>&1 || [ -x "$PYBIN" ] || { [ "$DRY" -eq 1 ] && log "(dry-run: interpreter $PYBIN not found)" || die "interpreter not found: $PYBIN"; }
if command -v "$PYBIN" >/dev/null 2>&1 || [ -x "$PYBIN" ]; then
  env_out="$("$PYBIN" - <<'PYEOF'
import sys
try:
    import numpy, sklearn
except Exception as e:
    print(f"IMPORT-FAIL {e}"); sys.exit(0)
print(f"python={sys.version.split()[0]} numpy={numpy.__version__} sklearn={sklearn.__version__}")
ok = (numpy.__version__ == "2.0.2") and (sklearn.__version__ == "1.6.1")
print("PINS-OK" if ok else "PINS-MISMATCH (expected numpy 2.0.2, scikit-learn 1.6.1)")
PYEOF
)"
  log "  env: $env_out"
  case "$env_out" in
    *PINS-OK*) : ;;
    *) [ "$DRY" -eq 1 ] && log "(dry-run: pin mismatch reported, not fatal)" || die "environment pins not met (RELEASE_MANIFEST.md)";;
  esac
fi
[ "$DRY" -eq 1 ] && { log "dry-run complete (the run order is in the header of this script)"; exit 0; }

log "  hashing inputs -> $LOGDIR/inputs.sha256"
( cd "$REPO" && sha256_of "${req[@]}" ) > "$LOGDIR/inputs.sha256"
log "  hashing outputs before the run -> $LOGDIR/outputs_before.sha256"
( cd "$REPO" && find gemma_grid/geometry_out gemma_grid/data -type f 2>/dev/null | LC_ALL=C sort | while IFS= read -r f; do sha256_of "$f"; done ) > "$LOGDIR/outputs_before.sha256"

set_aside() {  # move an existing output under the run directory so its step recomputes it
  local rel="$1"
  [ "$RESUME" -eq 1 ] && return 0
  [ -e "$REPO/$rel" ] || return 0
  mkdir -p "$LOGDIR/set_aside/$(dirname "$rel")"
  mv "$REPO/$rel" "$LOGDIR/set_aside/$rel"
  log "  set aside: $rel"
}

# ---------------- Phase 1: run order -----------------------------------------
step() {  # step <n> <label> <primary-output> <cmd...>
  local n="$1" label="$2" out="$3"; shift 3
  [ "$n" -gt "$STOP_AFTER" ] && { log "stop-after $STOP_AFTER reached"; exit 0; }
  if [ "$RESUME" -eq 1 ] && [ -s "$out" ]; then log "step $n [$label]: SKIP (resume; output exists)"; return; fi
  log "step $n [$label]: START"
  ( cd "$REPO" && "$@" ) >"$LOGDIR/logs/step${n}_${label}.log" 2>&1 \
    || die "step $n [$label] failed; see $LOGDIR/logs/step${n}_${label}.log"
  [ -s "$out" ] || die "step $n [$label]: primary output missing: $out"
  log "step $n [$label]: DONE"
}
reached() { [ "$1" -le "$STOP_AFTER" ] || { log "stop-after $STOP_AFTER reached"; exit 0; }; }

S=gemma_grid/scripts
step 1 tokenize "$GG/data/c3_tokens_neutral.npz" "$PYBIN" $S/c3_tokenize_neutral.py

# step 2: rebase, then the enforced anchor gate
reached 2
log "step 2 [rebase]: START"
( cd "$REPO" && "$PYBIN" $S/run_rebase.py ) >"$LOGDIR/logs/step2_rebase.log" 2>&1 \
  || die "step 2 [rebase] failed; see $LOGDIR/logs/step2_rebase.log"
for c in "${CELLS[@]}"; do
  [ -s "$GG/data/$c/vectors/${c}__cleanup_basis.pt" ] || die "step 2 [rebase]: ${c}__cleanup_basis.pt missing"
  [ "$c" = "$ANCHOR" ] || [ -s "$GG/data/$c/vectors/canonical.pt" ] || die "step 2 [rebase]: $c canonical.pt missing"
done
log "step 2 [rebase]: DONE (cleanup bases for six models, canonical.pt for five)"
if grep -q "\[anchor\]" "$LOGDIR/logs/step2_rebase.log"; then
  anchor_gate "$LOGDIR/logs/step2_rebase.log"
else
  # run_rebase.py keeps per-model state in BOOT_SCRATCH; a resumed run may not repeat the anchor check
  [ "$RESUME" -eq 1 ] && log "anchor gate: no [anchor] line in this log (resumed; the base 27B model was done earlier)" \
                      || anchor_gate "$LOGDIR/logs/step2_rebase.log"
fi

# step 3: Stage A per model, then the cross-model table
reached 3
if [ "$RESUME" -eq 1 ] && [ -s "$OUT_TREE/stageA_cross_cell_table.csv" ]; then
  log "step 3 [stageA]: SKIP (resume; cross-model table exists)"
else
  for c in "${CELLS[@]}"; do set_aside "gemma_grid/geometry_out/$c/stageA_summary.json"; done
  log "step 3 [stageA]: START"
  ( cd "$REPO" && "$PYBIN" $S/run_stageA.py ) >"$LOGDIR/logs/step3_stageA.log" 2>&1 \
    || die "step 3 [stageA] failed; see $LOGDIR/logs/step3_stageA.log"
  ( cd "$REPO" && "$PYBIN" $S/phase4_stageA_crosstable.py --tree gemma_grid/geometry_out ) >"$LOGDIR/logs/step3b_crosstable.log" 2>&1 \
    || die "step 3b [crosstable] failed; see $LOGDIR/logs/step3b_crosstable.log"
  [ -s "$OUT_TREE/stageA_cross_cell_table.csv" ] || die "step 3 [stageA]: cross-model table missing"
  log "step 3 [stageA]: DONE"
fi

step 5 stageB "$OUT_TREE/stageB_contrasts.json" "$PYBIN" $S/run_stageB.py

# step 6: bootstrap, one model per call
reached 6
log "step 6 [bootstrap]: START (six models)"
for c in "${CELLS[@]}"; do
  bout="$OUT_TREE/$c/bootstrap_B1000.npz"
  if [ "$RESUME" -eq 1 ] && [ -s "$bout" ]; then log "  step 6 [$c]: SKIP (resume)"; continue; fi
  ( cd "$REPO" && "$PYBIN" $S/run_bootstrap.py "$c" ) >"$LOGDIR/logs/step6_bootstrap_${c}.log" 2>&1 \
    || die "step 6 [bootstrap:$c] failed; see $LOGDIR/logs/step6_bootstrap_${c}.log"
  [ -s "$bout" ] || die "step 6 [bootstrap:$c]: output missing: $bout"
  log "  step 6 [$c]: DONE"
done

step 7 stageC "$OUT_TREE/stageC_grading.json" "$PYBIN" $S/run_stageC_grade.py

# step 7b: the anchor readout assets (base 27B unembedding, final norm, vocabulary,
# emotion token ids) into BOOT_SCRATCH, read by steps 8 and 10
reached 8
assets=(anchor_WU_f16.npy anchor_ln_final.npy anchor_vocab.json emotion_token_ids.json)
have=1; for a in "${assets[@]}"; do [ -s "$BOOT_SCRATCH/$a" ] || have=0; done
if [ "$RESUME" -eq 1 ] && [ "$have" -eq 1 ]; then
  log "step 7b [lens-preflight]: SKIP (resume; assets present)"
else
  log "step 7b [lens-preflight]: START"
  ( cd "$REPO" && "$PYBIN" $S/phase4_lens_preflight.py ) >"$LOGDIR/logs/step7b_lens_preflight.log" 2>&1 \
    || die "step 7b [lens-preflight] failed; see $LOGDIR/logs/step7b_lens_preflight.log"
  for a in "${assets[@]}"; do [ -s "$BOOT_SCRATCH/$a" ] || die "step 7b [lens-preflight]: missing $a"; done
  log "step 7b [lens-preflight]: DONE"
fi

step 8 c3 "$OUT_TREE/c3_summary.json" "$PYBIN" $S/run_c3.py

# step 8b: save the cleaned token-embedding vectors that step 8 computes
reached 9
have=1; for c in "${CELLS[@]}"; do [ -s "$OUT_TREE/$c/c3_vectors.pt" ] || have=0; done
if [ "$RESUME" -eq 1 ] && [ "$have" -eq 1 ]; then
  log "step 8b [c3-savevecs]: SKIP (resume; six files present)"
else
  for c in "${CELLS[@]}"; do set_aside "gemma_grid/geometry_out/$c/c3_vectors.pt"; done
  log "step 8b [c3-savevecs]: START"
  ( cd "$REPO" && "$PYBIN" $S/run_c3_savevecs.py ) >"$LOGDIR/logs/step8b_c3savevecs.log" 2>&1 \
    || die "step 8b [c3-savevecs] failed; see $LOGDIR/logs/step8b_c3savevecs.log"
  for c in "${CELLS[@]}"; do [ -s "$OUT_TREE/$c/c3_vectors.pt" ] || die "step 8b [c3-savevecs]: missing $c/c3_vectors.pt"; done
  log "step 8b [c3-savevecs]: DONE"
fi

step 9 c3grade "$OUT_TREE/c3_grading.json" "$PYBIN" $S/run_c3_grade.py

# step 10: the vocabulary readout per model, then the cross-model table
reached 10
if [ "$RESUME" -eq 1 ] && [ -s "$OUT_TREE/lens_cross_cell.json" ]; then
  log "step 10 [lens]: SKIP (resume; cross-model table exists)"
else
  log "step 10 [lens]: START (six models)"
  for c in "${CELLS[@]}"; do
    if [ "$RESUME" -eq 1 ] && [ -s "$OUT_TREE/$c/lens_summary.json" ]; then log "  step 10 [$c]: SKIP (resume)"; continue; fi
    ( cd "$REPO" && "$PYBIN" $S/run_lens.py --cell "$c" --budget 1000000000 ) >"$LOGDIR/logs/step10_lens_${c}.log" 2>&1 \
      || die "step 10 [lens:$c] failed; see $LOGDIR/logs/step10_lens_${c}.log"
    [ -s "$OUT_TREE/$c/lens_summary.json" ] || die "step 10 [lens:$c]: output missing"
    log "  step 10 [$c]: DONE"
  done
  ( cd "$REPO" && "$PYBIN" $S/phase4_lens_crosstable.py --tree gemma_grid/geometry_out ) >"$LOGDIR/logs/step10b_lens_crosstable.log" 2>&1 \
    || die "step 10b [lens-crosstable] failed; see $LOGDIR/logs/step10b_lens_crosstable.log"
  [ -s "$OUT_TREE/lens_cross_cell.json" ] || die "step 10 [lens]: cross-model table missing"
  log "step 10 [lens]: DONE"
fi

step 11 extracts  "$OUT_TREE/extracts_summary.json"                 "$PYBIN" $S/run_extracts.py
step 12 layerstab "$OUT_TREE/layerstab_ci.json"                     "$PYBIN" $S/phase4_layerstab_ci.py
step 13 seamgap   "$OUT_TREE/seam_gap_metric_v1.json"               "$PYBIN" $S/seam_gap_metric.py
step 14 op1spec   "$OUT_TREE/op1_spectrum_measures_v1.json"         "$PYBIN" $S/op1_spectrum_measures.py
step 15 appE      "$OUT_TREE/norm_robustness_full_results.json"     "$PYBIN" $S/norm_robustness_full.py
step 16 shifts    "$OUT_TREE/fixed_probe_tuning_shifts_results.json" "$PYBIN" $S/fixed_probe_tuning_shifts.py
step 17 z1        "$OUT_TREE/zhang_parallels/zhang_z1_results.json" "$PYBIN" $S/zhang_z1.py
step 18 z2        "$OUT_TREE/zhang_parallels/zhang_z2_results.json" "$PYBIN" $S/zhang_z2.py
step 19 z3        "$OUT_TREE/zhang_parallels/zhang_z3_results.json" "$PYBIN" $S/zhang_z3.py

# ---------------- Phase 2: wrap-up -------------------------------------------
log "Phase 2: hashing outputs after the run -> $LOGDIR/outputs_after.sha256"
( cd "$REPO" && find gemma_grid/geometry_out gemma_grid/data -type f 2>/dev/null | LC_ALL=C sort | while IFS= read -r f; do sha256_of "$f"; done ) > "$LOGDIR/outputs_after.sha256"
"$PYBIN" - "$LOGDIR" <<'PYEOF' | tee -a "$LOGDIR/logs/run.log"
import os, sys
d = sys.argv[1]
def read(name):
    out = {}
    for line in open(os.path.join(d, name)):
        h, p = line.rstrip("\n").split(None, 1)
        out[p.lstrip("*")] = h
    return out
before, after = read("outputs_before.sha256"), read("outputs_after.sha256")   # before includes files later set aside
same = sorted(p for p in after if before.get(p) == after[p])
changed = sorted(p for p in after if p in before and before[p] != after[p])
new = sorted(p for p in after if p not in before)
with open(os.path.join(d, "outputs_changed.txt"), "w") as f:
    for p in changed: f.write(f"changed {p}\n")
    for p in new: f.write(f"new {p}\n")
print(f"outputs: {len(same)} byte-identical to the files before the run, {len(changed)} changed, {len(new)} new")
print(f"the changed and new files are listed in {os.path.join(d, 'outputs_changed.txt')}")
PYEOF
log "DONE. Logs and hashes are in $LOGDIR"
