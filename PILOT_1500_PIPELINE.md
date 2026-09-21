# 1,500-record single-judge DPO pilot

## Purpose

This branch validates the complete SDT DPO workflow before the final nine
judgments are available. It is a pilot, not the final alignment experiment.
The raw 1,500-record JSON contains one judge model with three repeated attempts
per response. The final data is expected to contain judgments from multiple
models.

## What changed from the original 100-record workflow

| Area | Original multi-judge mode | 1,500-record pilot mode |
|---|---|---|
| Evidence | Multiple judge models | One judge model, three attempts |
| Score used | Attempt-derived mean by default | Supplied `score_avg` |
| Repeat variability | Used in the old confidence heuristic | Supplied `score_std`, diagnostic only |
| Agreement | Calculated across judge models | Not defined (`null`) |
| Confidence filtering | Enabled | Disabled |
| Minimum common judges | 2 | 1 |
| Target/reference checkpoint | Qwen2.5-0.5B-Instruct | Phi-4-mini-instruct for the selected pilot run |
| Evaluation | Stored preference pairs | Stored pairs plus fresh blind generation comparison |

The DPO loss, equal pair weighting, prompt/chosen/rejected format, prompt-level
split, validation-only selection, and locked-test policy are unchanged.

## Data policy

For each successful response judgment, pair preparation reads the supplied
`judgments[].score_avg` and `judgments[].score_std`. Individual attempt text and
scores are retained only as source evidence. A block whose status is not
`success` is missing evidence even if its placeholder score is `0.0`.

Each record has `baseline`, `sdt`, and `sdt-augmented` responses, producing three
possible comparisons. The higher supplied average is chosen and the lower is
rejected. Empty prompts/responses, identical responses, missing successful
common evidence, exact ties, and margins below `0.10` are excluded. The margin
rule is configurable and should be accompanied by a sensitivity report; it is
not a universal DPO constant.

Pilot rows carry:

```json
{
  "label_mode": "single_judge_pilot",
  "label_confidence": "single_judge_pilot",
  "confidence_score": null,
  "judge_agreement": null,
  "evidence_type": "single_model_repeated_attempts",
  "pair_weight": 1.0
}
```

This avoids presenting one-of-one judge coverage as robust cross-judge
confidence. The top-level `preferred_response` remains an audit check and is not
given to the model.

## Split and leakage policy

The deterministic split is 80% train, 10% validation, and 10% test. Split
assignment happens from the normalized prompt before expanding a record into
pairs, so all comparisons from one instruction stay together. The final study
must also cluster semantic near-duplicates before splitting.

The test split remains locked while filters, hyperparameters, and the checkpoint
are selected. Because this pilot is used for development, its prompts must not
be reused in the final experiment's test partition.

## Target model and selected pilot configuration

The focused pilot notebook uses `microsoft/Phi-4-mini-instruct`. The unchanged
checkpoint is both the baseline and frozen DPO reference. The instruct
checkpoint is used because there is no preceding SFT stage. Training updates
every policy parameter; this is not LoRA.

The completed validation sweep selected a learning rate of `1e-6`, beta `0.10`,
and one epoch. The focused reproduction does not repeat the broad sweep. It
evaluates about four times during the epoch and loads the checkpoint with the
highest TRL validation reward accuracy before saving the final model. The
external prompt-macro comparison remains the experiment-level validation
summary.

The configured maximum sequence length is 1,536 to keep full-parameter Phi
training within the available A100 memory. The Colab notebook reports the
actual Phi-tokenized length distribution before training. Any rows removed for
length are recorded in the training manifest.

## Exact pipeline

1. Upload the raw JSON without committing it to Git.
2. Create all three pair types in `single_judge_pilot` mode using supplied
   aggregates.
3. Review the judgment audit, exclusions, split counts, and duplicate-attempt
   counts.
4. Run unit tests and inspect Phi token lengths.
5. Reproduce the already selected `1e-6`, beta `0.10`, one-epoch configuration.
   Validate and save at regular intervals, then load the best within-epoch
   checkpoint by validation reward accuracy.
6. Freeze one checkpoint and verify that its parameters changed.
7. Unlock the test once and evaluate baseline and DPO on identical stored pairs.
8. Generate one fresh baseline response and one fresh DPO response for every
   eligible test prompt with identical decoding settings.
9. Record whitespace-identical generations as deterministic ties. Randomize A/B
   order for the remaining responses and use an independent local Hugging Face
   judge to report wins, ties, and seven SDT dimension deltas.
10. Create a unique run ID from the data, model, configuration, Git commit, and
    timestamp. Copy the raw input, processed pairs, pair audit, resolved config,
    dependency snapshot, final selected checkpoint, and evaluation artifacts to
    Drive. Exclude large temporary step checkpoints, verify every copied file by
    size, and require a nonempty final `*.safetensors` checkpoint.

## Run in Colab

Open `notebooks/SDT_1500_DPO_Pilot_Colab.ipynb` from the
`1500-record-dpo-pipeline` branch. Select an A100 GPU runtime, run setup, upload
the updated 1,500-record JSON, and execute through validation. The notebook
uses `configs/pilot_1500_phi.yaml` and saves the selected model to Drive as soon
as training and validation finish. Later test and generation artifacts are
synced immediately after their stage. Generation and LLM judging are resumable;
judge outputs are written directly to Drive. Keep
`RUN_LOCKED_TEST = False` until the configuration is frozen, then change it to
`True` and run the locked-test sections once.

For generated-response judging, the notebook uses
`Qwen/Qwen2.5-7B-Instruct` locally through Transformers. It requires no API key
and has no per-request API charge, although it consumes Colab GPU compute units
and downloads the model weights. The judge is optional for pipeline debugging
but required for the planned behavioral comparison. The judge cache is bound
to the input hashes, model, decoding limit, prompt version, and A/B seed so
incompatible runs cannot be silently mixed.

The Drive run is complete only when `backup-verification.json` lists the final
weight files and the `reproducibility/` directory contains the raw data, pairs,
pair report, exact notebook, resolved YAML, Git commit, package snapshot, and
input hashes.

## Command-line pair preparation

After `python -m pip install -e .`:

```bash
sdt-build-pairs \
  --input data/raw/sdt_results_1500.json \
  --output data/processed/dpo_pairs_1500.jsonl \
  --report data/processed/pair_report_1500.json \
  --label-mode single_judge_pilot \
  --use-supplied-aggregates \
  --min-common-judges 1 \
  --min-margin 0.10 \
  --train-share 0.80 \
  --validation-share 0.10 \
  --seed 42

python -m unittest discover -s tests -v
sdt-train-dpo --config configs/pilot_1500_phi.yaml
```

The raw dataset and trained checkpoints are intentionally ignored by Git.

## Interpretation boundary

A successful pilot demonstrates correct parsing, leakage control, Phi/TRL
compatibility, checkpoint movement, and measurable preliminary preference and
behavioral signals. It does not establish robust SDT alignment because its
labels come from one judge model and many repeated judge texts are duplicated.
The final multi-model judgments must be used to rebuild labels and run a new
frozen experiment.
