# SDT full-parameter DPO pipeline

This repository converts the SDT judgment JSON into auditable preference pairs,
trains **all parameters** of `Qwen/Qwen2.5-0.5B-Instruct` with DPO, and compares the
frozen baseline and full-DPO checkpoint on the same locked test set.

This is not LoRA or PEFT training. `train.py` verifies at runtime that every model
parameter is trainable and stops if any part of the model is frozen.

## Methodology

- Failed judge blocks are treated as missing evidence, never as zero scores.
- All pairs from one prompt are assigned to the same split before pair expansion.
- Near-ties and low-confidence comparisons are excluded.
- Retained pairs are trained with equal weight using only `prompt`, `chosen`, and
  `rejected`; judge scores and explanations are annotation metadata.
- Validation selects the checkpoint. The test labels are not evaluated until the
  training configuration and checkpoint are frozen.
- Baseline and DPO models are scored on identical test pair IDs with the same chat
  template and length-normalized conditional response log-probability.

## Repository layout

```text
configs/full.yaml          Full-parameter DPO settings
data/raw/                  Supplied 100-record structural sample
data/processed/            Generated pairs and preparation audit
src/sdt_dpo/pairs.py       Raw JSON -> chosen/rejected pairs and splits
src/sdt_dpo/train.py       Full-parameter DPO trainer
src/sdt_dpo/evaluate.py    Baseline or full-checkpoint evaluator
src/sdt_dpo/compare.py     Paired baseline-vs-DPO statistics
scripts/run_sample_pipeline.sh
tests/                     Data, leakage, and metric tests
outputs/                   Created during training and evaluation
```

## Recommended: run in Google Colab

Select **Runtime > Change runtime type > GPU**, then check the assigned accelerator:

```python
!nvidia-smi
```

Clone or upload this repository, enter its directory, and install it:

```python
%cd /content/SDT
%pip install -e .
```

Run the complete workflow:

```bash
!bash scripts/run_sample_pipeline.sh
```

The default configuration uses batch size 1, gradient accumulation, gradient
checkpointing, mixed precision on CUDA, and precomputed reference log-probabilities.
These choices make full DPO of the 0.49B model practical on a modest Colab GPU.

Colab storage is temporary. Copy `outputs/` to Google Drive or download it after the
run if the checkpoint must be preserved.

## Run locally

```bash
cd "/Users/ranaezzeddine/Desktop/SDT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
bash scripts/run_sample_pipeline.sh
```

Apple Silicon uses PyTorch MPS automatically; an Intel Mac falls back to CPU. Full
DPO consumes substantially more memory than LoRA, so Colab GPU is the recommended
environment even for this small model.

## Workflow details

### 1. Prepare DPO pairs

The all-in-one script runs this command first:

```bash
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
```

The sample has 300 possible response comparisons and retains 188 reliable pairs:
139 train, 27 validation, and 22 test. The test pairs come from only 10 prompts, so
their results demonstrate pipeline behavior rather than reliable model improvement.

### 2. Test preparation logic

```bash
python -m unittest discover -s tests -v
```

### 3. Train all model parameters

```bash
sdt-train-dpo --config configs/full.yaml
```

Important defaults in `configs/full.yaml`:

```yaml
model_id: Qwen/Qwen2.5-0.5B-Instruct
training_method: full_parameter_dpo
learning_rate: 5.0e-7
num_train_epochs: 1
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_length: 1024
precompute_ref_log_probs: true
```

The complete trained model and tokenizer are written to `outputs/dpo-full/`. Unlike
a LoRA adapter, this directory is a complete checkpoint that can be loaded directly
with `AutoModelForCausalLM.from_pretrained()`.

### 4. Evaluate baseline and DPO checkpoint

```bash
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
```

The reports include pair preference accuracy, equal-weight-per-prompt accuracy,
chosen-minus-rejected model margin, confidence/source subgroups, a prompt-cluster
bootstrap interval, and the paired change from baseline to DPO. Prompt-clustered
results are primary because several comparison pairs can come from one prompt.

## Expected outputs

```text
outputs/
├── dpo-full/                 Complete trained model and tokenizer
├── base-test.json            Baseline summary
├── base-test-pairs.jsonl     Baseline per-pair scores
├── dpo-test.json             Full-DPO summary
├── dpo-test-pairs.jsonl      Full-DPO per-pair scores
└── base-vs-dpo.json          Paired comparison
```

## Scaling to the complete dataset

The 100 records are appropriate for verifying parsing, training, saving, reloading,
and evaluation. They are not sufficient to determine whether full DPO improves the
model. For the complete 20k-50k records, create new prompt-grouped splits, add
near-duplicate clustering and metadata stratification, select settings using only
train/validation, and keep the final test set locked until all choices are frozen.
