#!/usr/bin/env bash
# Validation of the deconfounded AUC and the site comparison (needs run_all.sh outputs). About 15 minutes.
# Run from the project root:  bash scripts/run_validation.sh
set -o pipefail
step() { echo "== $(date +%H:%M) $*"; "$@" || echo "!! FAILED: $*"; }
step python -m src.site_shortcut compare
step python -m src.validate_metric simulate --reps 500
step python -m src.validate_metric sweep --resamples 20
step python -m src.figures validation
step python -m src.supp_tables
echo "#### VALIDATION DONE $(date +%H:%M)"
