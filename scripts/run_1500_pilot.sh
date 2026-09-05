#!/usr/bin/env bash
set -euo pipefail

RAW_DATA_PATH="${RAW_DATA_PATH:-data/raw/sdt_results_1500.json}"
PAIRS_PATH="data/processed/dpo_pairs_1500.jsonl"
REPORT_PATH="data/processed/pair_report_1500.json"
BASELINE_MODEL="LiquidAI/LFM2.5-1.2B-Instruct"
DPO_MODEL="outputs/lfm25-1500-full-dpo"
MAX_LENGTH="2048"

if [[ ! -f "$RAW_DATA_PATH" ]]; then
  echo "Missing pilot data: $RAW_DATA_PATH"
  exit 1
fi

sdt-build-pairs \
  --input "$RAW_DATA_PATH" \
  --output "$PAIRS_PATH" \
  --report "$REPORT_PATH" \
  --label-mode single_judge_pilot \
  --use-supplied-aggregates \
  --min-common-judges 1 \
  --min-margin 0.10 \
  --train-share 0.80 \
  --validation-share 0.10 \
  --seed 42

python -m unittest discover -s tests -v
sdt-train-dpo --config configs/pilot_1500_lfm.yaml

if [[ "${RUN_LOCKED_TEST:-0}" != "1" ]]; then
  echo "Training and validation complete. Test remains locked."
  echo "After freezing the configuration, rerun test commands from PILOT_1500_PIPELINE.md."
  exit 0
fi

sdt-verify-checkpoint \
  --baseline-model "$BASELINE_MODEL" \
  --trained-model "$DPO_MODEL" \
  --output outputs/pilot-1500-checkpoint-change.json

sdt-evaluate-pairs \
  --pairs "$PAIRS_PATH" \
  --model "$BASELINE_MODEL" \
  --split test \
  --max-length "$MAX_LENGTH" \
  --all-confidence \
  --output outputs/pilot-1500-baseline-test.json \
  --details outputs/pilot-1500-baseline-test-pairs.jsonl

sdt-evaluate-pairs \
  --pairs "$PAIRS_PATH" \
  --model "$DPO_MODEL" \
  --split test \
  --max-length "$MAX_LENGTH" \
  --all-confidence \
  --output outputs/pilot-1500-dpo-test.json \
  --details outputs/pilot-1500-dpo-test-pairs.jsonl

sdt-compare-evaluations \
  --baseline-details outputs/pilot-1500-baseline-test-pairs.jsonl \
  --dpo-details outputs/pilot-1500-dpo-test-pairs.jsonl \
  --output outputs/pilot-1500-fixed-pair-comparison.json \
  --details-output outputs/pilot-1500-relative-test-pairs.jsonl \
  --beta 0.10

sdt-generate-responses \
  --pairs "$PAIRS_PATH" \
  --model "$BASELINE_MODEL" \
  --output outputs/pilot-1500-baseline-generations.jsonl \
  --split test \
  --max-input-length "$MAX_LENGTH"

sdt-generate-responses \
  --pairs "$PAIRS_PATH" \
  --model "$DPO_MODEL" \
  --output outputs/pilot-1500-dpo-generations.jsonl \
  --split test \
  --max-input-length "$MAX_LENGTH"

echo "Fixed-pair testing and fresh generation are complete."
echo "Run sdt-judge-generations with an independent judge endpoint as documented."
