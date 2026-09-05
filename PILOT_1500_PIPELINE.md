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
| Target/reference checkpoint | Qwen2.5-0.5B-Instruct | LFM2.5-1.2B-Instruct |
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

## Target model

The pilot uses `LiquidAI/LFM2.5-1.2B-Instruct`. The unchanged checkpoint is both
the baseline and frozen DPO reference. The instruct checkpoint is used because
there is no preceding SFT stage. Training updates every policy parameter; this
is not LoRA.

The configured maximum sequence length is 2,048. The Colab notebook reports the
actual LFM-tokenized length distribution before training. Any rows removed for
length are recorded in the training manifest.

## Exact pipeline

1. Upload the raw JSON without committing it to Git.
2. Create all three pair types in `single_judge_pilot` mode using supplied
   aggregates.
3. Review the judgment audit, exclusions, split counts, and duplicate-attempt
   counts.
4. Run unit tests and inspect LFM token lengths.
5. Evaluate candidate settings only on validation prompts. Optional tuning
   compares candidates by prompt-macro implicit reward accuracy, then implicit
   margin, then validation loss.
6. Freeze one checkpoint and verify that its parameters changed.
7. Unlock the test once and evaluate baseline and DPO on identical stored pairs.
8. Generate one fresh baseline response and one fresh DPO response for every
   eligible test prompt with identical decoding settings.
9. Randomize A/B order and use an independent OpenAI-compatible judge endpoint
   to report wins, ties, and seven SDT dimension deltas.
10. Save the configuration, manifests, pair-level details, generations, raw
    judge output, and summaries.

## Run in Colab

Open `notebooks/SDT_1500_DPO_Pilot_Colab.ipynb` from the
`1500-record-dpo-pipeline` branch. Select a GPU runtime, run setup, upload
`sdt_results_1500.json`, and execute through validation. Keep
`RUN_LOCKED_TEST = False` until the configuration is frozen, then change it to
`True` and run the locked-test sections once.

For generated-response judging, add `JUDGE_API_KEY` as a Colab secret and set an
independent judge model and OpenAI-compatible chat-completions endpoint in the
final cell. The judge is optional for pipeline debugging but required for the
planned behavioral comparison.

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
sdt-train-dpo --config configs/pilot_1500_lfm.yaml
```

The raw dataset and trained checkpoints are intentionally ignored by Git.

## Interpretation boundary

A successful pilot demonstrates correct parsing, leakage control, LFM/TRL
compatibility, checkpoint movement, and measurable preliminary preference and
behavioral signals. It does not establish robust SDT alignment because its
labels come from one judge model and many repeated judge texts are duplicated.
The final multi-model judgments must be used to rebuild labels and run a new
frozen experiment.
