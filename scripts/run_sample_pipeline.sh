#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after: python -m pip install -e .
sdt-build-pairs \
  --input data/raw/sdt_100_llama.json \
  --output data/processed/dpo_pairs.jsonl \
  --report data/processed/pair_report.json \
  --min-common-judges 2 \
  --min-margin 0.10 \
  --min-confidence 0.60 \
  --train-share 0.70 \
  --validation-share 0.15 \
  --seed 42

python -m unittest discover -s tests -v
sdt-train-dpo --config configs/full.yaml

# Open the locked test split only after the training choice is frozen.
sdt-evaluate-pairs \
  --pairs data/processed/dpo_pairs.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --output outputs/base-test.json \
  --details outputs/base-test-pairs.jsonl

sdt-evaluate-pairs \
  --pairs data/processed/dpo_pairs.jsonl \
  --model outputs/dpo-full \
  --output outputs/dpo-test.json \
  --details outputs/dpo-test-pairs.jsonl

sdt-compare-evaluations \
  --baseline-details outputs/base-test-pairs.jsonl \
  --dpo-details outputs/dpo-test-pairs.jsonl \
  --output outputs/base-vs-dpo.json
