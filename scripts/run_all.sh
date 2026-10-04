#!/usr/bin/env bash
# Full pipeline, in plan order. Run from the project root:  bash scripts/run_all.sh
# Smoke test on fake data:                                   bash scripts/run_all.sh configs/synthetic.yaml
# Each step writes files the next step reads, so you can also run the lines one at a time.
set -eo pipefail   # no -u: macOS bash 3.2 treats empty arrays as unbound
CFG="${1:-configs/default.yaml}"
C=(--config "$CFG")
BB=$(python -c "import yaml; print(' '.join(b.split('.')[0] for b in yaml.safe_load(open('$CFG'))['features']['backbones']))")
MAIN=$(echo "$BB" | awk '{print $1}')   # first backbone = main one for ablations
SMOKE=()
[[ "$CFG" == *synthetic* ]] && SMOKE=(--no-pretrained)

echo "== Data and folds"
python -m src.prepare_data "${C[@]}"
python -m src.make_folds "${C[@]}"
python -m src.extract_features "${C[@]}" "${SMOKE[@]}"

echo "== E2 baselines and fusion"
python -m src.baselines "${C[@]}"
for B in $BB; do
  python -m src.fusion "${C[@]}" --backbone "$B"
done

echo "== E3 ablations (main backbone)"
python -m src.fusion "${C[@]}" --backbone "$MAIN" --fusion concat
python -m src.fusion "${C[@]}" --backbone "$MAIN" --fusion film
python -m src.fusion "${C[@]}" --backbone "$MAIN" --hard-labels --tag hard
python -m src.fusion "${C[@]}" --backbone "$MAIN" --aux 0 --tag noAux
python -m src.fusion "${C[@]}" --backbone "$MAIN" --modality-dropout 0 --tag noModDrop

echo "== E5 symptom knock-outs and E7 shuffled-image control"
# Edit these names to match your symptom columns (see data/processed/columns.json).
KO1=$(python -c "import yaml; c=yaml.safe_load(open('$CFG'))['centor']; print(' '.join(v for k,v in c.items() if v and k in ('fever','cough')))")
KO2=$(python -c "import yaml; c=yaml.safe_load(open('$CFG'))['centor']; print(' '.join(v for v in c.values() if v))")
TAGS=()
if [[ -n "$KO1" ]]; then
  python -m src.baselines "${C[@]}" --models sym_lr --drop-symptoms $KO1 --tag noFeverCough
  python -m src.fusion "${C[@]}" --backbone "$MAIN" --drop-symptoms $KO1 --tag noFeverCough
  TAGS+=(noFeverCough)
fi
if [[ -n "$KO2" && "$KO2" != "$KO1" ]]; then
  python -m src.baselines "${C[@]}" --models sym_lr --drop-symptoms $KO2 --tag noCentor
  python -m src.fusion "${C[@]}" --backbone "$MAIN" --drop-symptoms $KO2 --tag noCentor
  TAGS+=(noCentor)
fi
python -m src.fusion "${C[@]}" --backbone "$MAIN" --shuffle-images --tag shuffled

echo "== E1 image-only fine-tuning (slowest step, run last so frozen-feature results come first)"
for M in $(python -c "import yaml; print(' '.join(yaml.safe_load(open('$CFG'))['finetune']['models']))"); do
  SAVE=()
  [[ "$M" == convnext* || "$M" == resnet* ]] && SAVE=(--save-full)   # CNN kept for Grad-CAM (E6)
  python -m src.finetune_image "${C[@]}" --model "$M" "${SMOKE[@]}" "${SAVE[@]}"
done

echo "== Metrics and statistics"
python -m src.evaluate "${C[@]}"
python -m src.stats "${C[@]}" --hypotheses --proposed "gated_$MAIN"
python -m src.label_dependence "${C[@]}" strata --models phone sym_lgbm "img_lr_$MAIN" "late_$MAIN" "gated_$MAIN" "gated_${MAIN}__shuffled"
python -m src.label_dependence "${C[@]}" residual --backbone "$MAIN" --sym-model sym_lr
if (( ${#TAGS[@]} )); then
  python -m src.label_dependence "${C[@]}" knockouts --fusion "gated_$MAIN" --symptom sym_lr --tags "${TAGS[@]}"
fi

echo "== E6 explainability and figures"
python -m src.explain "${C[@]}" shap
python -m src.explain "${C[@]}" gate --model "gated_$MAIN"
MODELS_DIR=$(python -c "from src.common import load_config; print(load_config('$CFG')['paths']['models'])")
for CK in "$MODELS_DIR"/*_full.pt; do
  [[ -f "$CK" ]] || continue
  python -m src.explain "${C[@]}" gradcam --model "$(basename "$CK" _full.pt)" || true
done
python -m src.figures "${C[@]}" roc --models "gated_$MAIN" sym_lgbm "img_lr_$MAIN" "late_$MAIN" \
  --labels "Gated fusion" "Symptoms (LightGBM)" "Image (frozen + LR)" "Late fusion"
python -m src.figures "${C[@]}" calibration --models "gated_$MAIN" sym_lgbm "img_lr_$MAIN"
python -m src.figures "${C[@]}" dca --models "gated_$MAIN" sym_lgbm "img_lr_$MAIN"
python -m src.figures "${C[@]}" ablation --models "gated_$MAIN" "concat_$MAIN" "film_$MAIN" \
  "gated_${MAIN}__hard" "gated_${MAIN}__noAux" "gated_${MAIN}__noModDrop" "gated_${MAIN}__shuffled"
python -m src.figures "${C[@]}" subgroups --model "gated_$MAIN"
echo "Done. Tables in results/metrics, figures in figures/."
