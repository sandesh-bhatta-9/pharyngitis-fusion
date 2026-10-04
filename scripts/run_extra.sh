#!/usr/bin/env bash
# Second batch: phone/site shortcut experiments, tie-rule sensitivity, E1 fine-tuning (1 repeat), final metrics.
# Run from the project root:  bash scripts/run_extra.sh
# Each step runs even if an earlier one fails; failures are marked "!! FAILED" in the log.
set -o pipefail
step() { echo "== $(date +%H:%M) $*"; "$@" || echo "!! FAILED: $*"; }
DINO=vit_small_patch14_dinov2
CNX=convnext_tiny

echo "#### 1. Phone/site shortcut"
step python -m src.site_shortcut size
step python -m src.site_shortcut within
step python -m src.site_shortcut cross
step python -m src.site_shortcut lowlevel
step python -m src.site_shortcut largeimg --models "img_lr_$DINO" "img_lr_$CNX" "stack_$DINO" "gated_$DINO" "gated_$CNX" phone sym_lgbm

echo "#### 2. Tie-rule sensitivity"
for RULE in nonbacterial drop; do
  C=(--config "configs/tie_$RULE.yaml")
  step python -m src.prepare_data "${C[@]}"
  step python -m src.make_folds "${C[@]}"
  step python -m src.baselines "${C[@]}"
  step python -m src.fusion "${C[@]}" --backbone "$DINO"
  step python -m src.fusion "${C[@]}" --backbone "$CNX"
  step python -m src.evaluate "${C[@]}"
  step python -m src.stats "${C[@]}" --hypotheses --proposed "gated_$DINO"
  step python -m src.label_dependence "${C[@]}" strata --models phone sym_lgbm "img_lr_$DINO" "stack_$DINO" "gated_$DINO"
done

echo "#### 3. E1 fine-tuning (1 repeat = 5 folds per model, low priority)"
step nice -n 10 python -m src.finetune_image --model densenet121 --repeats 1
step nice -n 10 python -m src.finetune_image --model mobilenetv3_large_100 --repeats 1
step nice -n 10 python -m src.finetune_image --model convnext_tiny.fb_in22k_ft_in1k --repeats 1 --save-full
step python -m src.explain gradcam --model "$CNX"

echo "#### 4. Final metrics, tests and figures (main analysis, ties = bacterial)"
step python -m src.evaluate
step python -m src.stats --hypotheses --proposed "gated_$DINO"
cp results/metrics/hypotheses.csv results/metrics/hypotheses_dinov2.csv
step python -m src.stats --hypotheses --proposed "gated_$CNX"
cp results/metrics/hypotheses.csv results/metrics/hypotheses_convnext.csv
step python -m src.stats --pair "img_lr_$DINO" phone --pair "img_lr_$CNX" phone --pair "stack_$DINO" "img_lr_$DINO" \
  --pair "gated_$CNX" "gated_${CNX}__hard" --pair "gated_$CNX" "concat_$CNX" --pair "img_lr_$DINO" size \
  --pair ft_densenet121 phone --pair ft_convnext_tiny phone
step python -m src.label_dependence strata --models phone size lowlevel sym_lgbm "img_lr_$CNX" "img_lr_$DINO" "stack_$DINO" \
  "gated_$CNX" "gated_$DINO" "gated_${CNX}__shuffled" ft_densenet121 ft_mobilenetv3_large_100 ft_convnext_tiny
step python -m src.figures roc --models "stack_$DINO" "img_lr_$DINO" phone sym_lgbm \
  --labels "Stacked fusion (DINOv2)" "Image only (DINOv2)" "Phone model only" "Symptoms (LightGBM)"
step python -m src.figures subgroups --model "img_lr_$DINO"
step python -m src.figures shortcut
echo "#### ALL DONE $(date +%H:%M)"
