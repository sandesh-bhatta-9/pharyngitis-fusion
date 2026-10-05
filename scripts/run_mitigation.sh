#!/usr/bin/env bash
# Third batch: deconfounded AUC and shortcut mitigation (needs the outputs of run_all.sh and run_extra.sh).
# Run from the project root:  bash scripts/run_mitigation.sh   (about 10 minutes on an Apple M5)
set -o pipefail
step() { echo "== $(date +%H:%M) $*"; "$@" || echo "!! FAILED: $*"; }

step python -m src.mitigate extract   # Shades-of-Gray embeddings -> data/features/<bb>__sog.npz
step python -m src.mitigate run       # 13 methods x 2 backbones -> results/mitigation/oof/
step python -m src.mitigate cross     # cross-phone transfer with colour constancy / per-phone standardisation
step python -m src.mitigate audit     # deconfounded AUC, within-phone AUC, probes, tests -> results/metrics/mitigation_*.csv
step python -m src.figures mitigation
echo "#### MITIGATION DONE $(date +%H:%M)"
