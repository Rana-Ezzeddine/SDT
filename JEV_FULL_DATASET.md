# Full Jev dataset workflow

This branch extends the 1500-record pipeline, preserving its historical parser and notebooks.
Updated 2026-11-11 on `jev-dataset-adaptation`. Local integration checks and a
single-T4 Qwen2.5-0.5B development pilot have completed on JEV data. Full-dataset
training and distributed/Ibex execution have not been run. See
[pilot results and remaining behavioral evaluation](JEV_BEHAVIOR_PILOT.md).

Release recheck: 58 unit/regression tests passed, and a fresh offline synthetic
training/save/reload/resume/generation/evaluation canary passed. The four clean
JEV notebooks are included; raw records, model weights, exported responses and
temporary outputs are deliberately excluded. Existing unrelated legacy Gemini/V2
working-tree edits are not part of this release.

## Preparation

Install with `python -m pip install -e .`. Keep raw data outside Git.

```bash
sdt-build-jev-pairs --input /path/sdt_gpt_fulljev_CLEAN.json --input /path/sdt_opus_fulljev_CLEAN.json --output data/processed/jev_dpo_pairs.jsonl --report data/processed/jev_pair_report.json
```

Uses supplied SDT/content averages, not overall score. Content average covers four quality dimensions.
The threshold is >=0.80. Pass beats fail; two passers are ordered by SDT average.
Both-failed, exact SDT ties and identical texts are excluded. These are explicit proposed pair rules,
pending team approval, not evidence that Ghadi has prescribed every pairwise comparison.
Stored preferred_response must be a passing SDT maximum; any tied maximum is accepted for audit,
but ties are not turned into training labels. Missing/invalid scores fail loudly, never become zero.
All three comparisons are audited. Instructions and nonempty input are included in the model prompt.
IDs include generator identity; normalized full prompts share a deterministic 80/10/10 split across files.
Near-duplicate clustering is not implemented: review semantic duplicates before freezing the final split.

## Training

```bash
sdt-train-dpo --config configs/jev_phi4_dpo.yaml
```

The supplied configuration is an initial Phi baseline, not a validated optimum for this dataset.
Use separate output directories per experiment. min_sdt_margin applies only to training/validation
SDT-based pairs, never quality-gate pairs. Do not tune the test population for each configuration.
Report validation selection and test counts separately. Length filtering remains model-tokenizer-specific.
Keep a common evaluable set when comparing models with different context limits.

For small smoke tests add max_train_samples and max_validation_samples to a copied configuration.
For distributed training, launch through Accelerate on an allocated compute node. A deepspeed
configuration path can now be passed to the trainer. Its TRL/reference-cache compatibility and memory
use must be tested before large runs. Exact Gemma/Qwen checkpoint IDs are not confirmed, so no
invented production configs are provided. Full 27B training is not GPU-validated on this machine.

## Locked fixed-pair evaluation

After selecting a configuration using validation, evaluate the original checkpoint and the saved
full model on the same test pairs. Jev records now bypass legacy confidence/margin
filters automatically; the explicit legacy flags below are harmless. Keep the Jev
test population fixed with the default `--min-sdt-margin 0`:

```bash
sdt-evaluate-pairs --pairs data/processed/jev_dpo_pairs.jsonl --model microsoft/Phi-4-mini-instruct --split test --all-confidence --min-margin 0 --max-length 1536 --output outputs/base-test.json --details outputs/base-test-pairs.jsonl
sdt-evaluate-pairs --pairs data/processed/jev_dpo_pairs.jsonl --model outputs/jev-phi4 --split test --all-confidence --min-margin 0 --max-length 1536 --output outputs/dpo-test.json --details outputs/dpo-test-pairs.jsonl
sdt-compare-evaluations --baseline-details outputs/base-test-pairs.jsonl --dpo-details outputs/dpo-test-pairs.jsonl --beta 0.1 --output outputs/jev-comparison.json
```

Implicit accuracy measures movement relative to the reference, not generated-answer correctness.
Absolute accuracy measures the model's ranking of stored candidates. Prompt macro and cluster
intervals account for multiple pairs per instruction. Report skipped examples and identical pair IDs.
Fresh-response generation/judging scripts remain available but are not Jev API integration.
General-capability benchmark evaluation is still required before claiming capabilities are preserved.

## Preservation

Save raw data hashes, pair report, configs, Git commit and package versions with each run.
Archive full weights and tokenizer as well as resumable optimizer checkpoints. Verify reload and
generation from the archived path before deleting local files. Never equate a metrics file or config.json
with a complete checkpoint. Ibex storage retention/backup policy must be confirmed with support.
# Local readiness audit — 2026-11-10

Historical local audit: data preparation, unit tests, and the tiny-model
training/save/reload/resume integration test passed. The subsequent single-T4
pilot also passed; multi-GPU and Ibex-specific execution remain unverified.

- 54 local unit tests pass (`PYTHONPATH=src python3 -m unittest discover -s tests -q`).
- Both supplied files were processed: 23,728 records, 71,184 possible pairs,
  69,216 retained (55,434 train / 6,752 validation / 7,030 test).
- Unique pair IDs, normalized-prompt split isolation, chosen quality >= 0.80,
  and strict SDT ordering for pass/pass comparisons were checked.
- Training and evaluation share Jev eligibility: quality-gate pairs are not
  discarded by legacy confidence or SDT-margin thresholds. `--min-sdt-margin`
  applies only to pass/pass comparisons; default zero. Legacy flags remain for
  older datasets. Evaluation now reports generator and preference-basis subgroups.
- Full-data training requires matching data SHA-256 and configuration before
  reusing a run directory. Unknown existing directories are refused, not deleted.
- Only the main distributed process writes final tokenizer/JSON metadata.

## Remaining release gates (do not call these completed)

1. Local tested versions are recorded in `constraints-smoke-tested.txt`. Repeat
   environment validation on Ibex's CUDA stack; the local CPU test cannot certify it.
2. Repeat the integration canary on the allocated GPU before a full job. Local
   real optimizer updates, changed weights, checkpoint resume, reload, inference,
   and fixed-pair evaluation have passed. Recovery was from a completed intermediate
   checkpoint, not from a forcibly terminated process or partially written files.
3. Audit near-duplicate prompts across splits. Exact normalization is implemented;
   semantic/near-duplicate clustering is not. Agree on the final frozen split.
4. Measure tokenizer-specific length exclusions for each target model. Do not
   silently compare different retained test subsets across models.
5. Validate multi-GPU behavior and memory on actual hardware. The DeepSpeed
   configuration hook is not a tested Ibex setup; reference precomputation and
   ZeRO compatibility must be checked before deployment.
6. Finalize a fresh-response evaluation using the team's Jev rubric and a separate
   capability-retention evaluation. Existing fixed-pair metrics alone do not
   establish improved generated SDT behavior or preserved general capability.
7. Verify durable checkpoint storage by hashes and a reload from the saved copy,
   not merely existence or file size. Retain data/configuration hashes, dependency
   versions, logs, selection decision, and evaluation outputs with every run.

Scientific acceptance: select configurations on validation only; freeze before
test evaluation. Report prompt-cluster uncertainty, SDT dimensions, content quality,
and capability changes, including negative results. Passing software tests does
not guarantee that DPO improves the research task.
