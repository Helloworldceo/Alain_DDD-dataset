#!/usr/bin/env bash
# Runs the complete SEED-VIG pipeline in order, from the raw recordings to the figures.
# Usage:  cd scripts/seedvig && ./run_pipeline.sh          (set PYTHON=... to choose the interpreter)
# Every step writes its outputs to results/seedvig/ and figures/seedvig/; the long training
# steps (5-7) resume from their saved results if they are interrupted.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== 1. Official features and labels";          $PY build_features.py
echo "== 2. Filtered raw windows";                   $PY build_raw.py
echo "== 3. Artifact cleaning";                      $PY preprocess_clean.py
echo "== 4. Causal feature extraction";              $PY extract_features.py
$PY check_features.py
$PY plot_cleaning.py
$PY plot_montage.py
echo "== 5. Pilot: hyperparameter selection";        $PY pilot.py reg;  $PY pilot.py cls
echo "== 6. Leave-one-person-out experiments";       $PY run_lopo.py reg; $PY run_lopo.py cls
$PY run_lopo.py reg signals=raw                      # cleaning ablation
echo "== 7. Additional analyses"
$PY feature_sets_full.py
$PY sensitivity.py
$PY modality.py
$PY kan_interpret.py
echo "== 8. Statistics, per-class results, figures"
$PY analyze_lopo.py
$PY compare_cleaning.py
$PY per_class.py
echo "Done. Results: results/seedvig/   Figures: figures/seedvig/"
