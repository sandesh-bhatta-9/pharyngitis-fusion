#!/usr/bin/env bash
# End-to-end mitigation: fine-tuned ConvNeXt-Tiny with ERM, GroupDRO and adversarial training (1 repeat each).
# Run from the project root:  bash scripts/run_finetune_mitigation.sh   (about 1-2 hours on an Apple M5)
set -o pipefail
step() { echo "== $(date +%H:%M) $*"; "$@" || echo "!! FAILED: $*"; }
for M in erm gdro adv; do
  step nice -n 10 python -m src.finetune_mitigate --method "$M" --model convnext_tiny.fb_in22k_ft_in1k --repeats 1
done
step python -m src.mitigate audit
echo "#### FINETUNE MITIGATION DONE $(date +%H:%M)"
