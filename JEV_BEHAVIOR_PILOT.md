# Jev behavioral development pilot

Updated 2026-11-11. **Both training runs, fixed-pair validation, generation and
Drive flush/remount/hash verification completed. Fresh-response JEV scoring is
pending; the test remains locked.** Every candidate started independently from
Qwen2.5-0.5B-Instruct. This is the November pilot, not the earlier 1,500-record run.

Completed recovery notebook: https://colab.research.google.com/drive/1ufbRlO5-ES_wS-5FI12VacDEkfNQqz69

## Completed validation results

All three models use the same 583 pairs from 200 instructions.

| Metric | Baseline | LR 5e-7 | LR 1e-6 |
| --- | ---: | ---: | ---: |
| Correct absolute rankings | 168 | 243 | 271 |
| Absolute pair accuracy | 28.82% | 41.68% | 46.48% |
| Absolute prompt-macro accuracy | 28.75% | 41.92% | 46.83% |
| Implicit reward accuracy | not applicable | 86.45% | 85.76% |
| Implicit prompt-macro accuracy | not applicable | 86.50% | 85.75% |
| Mean implicit reward margin | not applicable | 2.3754 | 2.9131 |

The 1e-6 candidate corrects 112 baseline errors and introduces 9: net +103 correct
pairs (+17.67 percentage points). Its prompt-macro improvement is +18.08 points
with a prompt-cluster 95% interval of [+14.50, +21.67]. For 5e-7 the prompt-macro
gain is +13.17 points, interval [+10.08, +16.25]. These intervals are versus
baseline, not a direct significance test between candidates. Absolute ranking
uses average token log-probabilities; implicit accuracy measures sequence-level
movement relative to the reference. Neither is generated-answer SDT correctness.

All 600 validation responses (200 per model) passed completeness checks and were
exported as `validation-jev-blinded.jsonl`. Give only that file to Ghadi; retain
`validation-jev-private-map.json` privately. JEV scoring, final selection,
locked-test evaluation and general-capability retention checks are still pending.

## Reproduce and recover

Build the short smoke bundle from private inputs (not included in Git):

```bash
PYTHONPATH=src python scripts/build_colab_qwen_bundle.py --input /path/sdt_gpt_fulljev_CLEAN.json --input /path/sdt_opus_fulljev_CLEAN.json
PYTHONPATH=src python scripts/build_jev_behavior_pilot.py --input /path/sdt_gpt_fulljev_CLEAN.json --input /path/sdt_opus_fulljev_CLEAN.json --exclude-bundle outputs/colab-qwen-smoke/sdt_colab_qwen_bundle.zip --output outputs/jev-behavior-pilot
python scripts/build_jev_behavior_notebook.py
```

The committed clean recovery notebooks can be used directly. To import a newly
downloaded browser-edited recovery notebook, explicitly provide its path:
`python scripts/build_behavior_resume_notebook.py --source /path/recovery.ipynb`.
Then `python scripts/build_behavior_account_transfer.py` rebuilds its backup-restore
variant. These recovery files are tied to the documented run by default, not
general automatic discovery. Inspect paths and use only authorized compute.

Live notebook: https://colab.research.google.com/drive/1YUcrfYavHOL5tMpUjHX8Kx4v9S7jPCU1

Drive run folder: `MyDrive/SDT_JEV_Behavior_Pilot_v2/20261110T190910Z`, restored
in the destination account from the separate private backup. Both restored models
passed SHA-256 checks after unmount/remount. The original backup was not changed.
Drive is mounted at `/content/pilot-drive`. The first `/content/drive` mount failed;
local-only preparation folder `20261110T190742Z` has no trained checkpoint.
Training began only after a successful mount check.

## Actual prepared sample

| Partition | Unique prompts | Retained pairs |
| --- | ---: | ---: |
| Training | 2,000 | 5,825 |
| Validation | 200 | 583 |
| Locked test | 300 | 867 |

Both commercial response generators are included: 3,737 Opus pairs and 3,538 GPT pairs overall. There are 5,685 SDT-ranking pairs and 1,590 content-gate pairs. Pair labels still use the team's quality threshold 0.8 and supplied SDT scores, not candidate-source names. All previous smoke-test prompt IDs are excluded.

Selection is deterministic at normalized full-prompt level, preserving production splits across response sources. Every retained pair of a selected prompt must fit 1,024 tokens. During sampling, 53 train / 9 validation / 11 test prompt groups failed that length rule. This is a length-bounded development subset, not a claim of full-dataset representativeness. Semantic near-duplicate leakage is not yet audited; normalized exact matching is the currently enforced guarantee. Topics were not used as sampling strata because this builder does not have an agreed topic taxonomy.

## Workflow

1. Open `notebooks/SDT_Jev_Behavior_Pilot_Colab.ipynb` on a GPU and upload `outputs/jev-behavior-pilot/jev_behavior_pilot.zip`.
2. Use a new folder under `MyDrive/SDT_JEV_Behavior_Pilot_v2`; save its printed path. Resume requires that exact path and matching input files.
3. Train two independent full-parameter candidates: learning rate 5e-7 or 1e-6, beta 0.1, up to two epochs, microbatch 1, accumulation 8, maximum length 1,024. Epoch-end validation selects the best epoch within each run. Each completed model is hashed. Interrupted runs can resume from available optimizer checkpoints.
4. Compare validation fixed pairs and generate answers on all 200 validation prompts for baseline and both candidates. Decoding is deterministic, with a 512-new-token limit for all models. Inspect length-limit hits before interpreting quality. No extra SDT prompt is given to the trained model.
5. Export anonymous single-response records for JEV. Keep the mapping private. The evaluator protocol/API is not implemented here: obtain Ghadi's agreed evaluator, or send the blinded validation file to Ghadi. Do not substitute an unvalidated prompt and call it the same JEV evaluation.
6. Select a candidate (or neither) using validation SDT scores and content quality. Review autonomy, competence, relatedness separately; report paired per-prompt changes and uncertainty. Do not equate DPO relative accuracy with generated-answer quality. An acceptable quality-degradation tolerance must be agreed before selecting based on these scores.
7. Enter the selected candidate and selection reason, then explicitly unlock the 300-prompt test once. Compare fixed pairs and export blinded test generations. Report unchanged answers, failures and truncated responses; do not silently omit them.
8. Flush Drive uploads, remount and verify both models' SHA256 values. Retain source bundle, sample, configuration, logs and output files.

## Storage, runtime and limits

### Independent Drive backup (2026-11-11)

Backup folder: https://drive.google.com/drive/folders/1UC9BWJsIzQOj2058PS8SiZwqeletJCBj

Both final models were confirmed through the Drive connector (independent of the Colab mount), then server-side copied into separate `lr5e7` and `lr1e6` backup folders, with model configuration, tokenizer, metrics and training metadata. Both copied `model.safetensors` files were read back in Drive listings at **1,976,163,472 bytes each**. This is server-side existence/size verification, not a fresh byte-level hash download. Optimizer checkpoint subfolders are not duplicated; final inference model files are. This is a separate copy in the same Drive account, not an offline or separate-account backup. Runtime was not interrupted. At inspection its UI displayed “Resuming execution”; final Colab flush/remount audit was not yet confirmed.

Reserve at least 30 GB of free Drive capacity for resumable checkpoints and final
models across both candidates (about 2 GB per FP32 final model). Colab can interrupt
runs. The 1,024-token pilot completed on T4; this does not certify larger models or
multi-GPU execution. Do not silently reduce lengths after viewing results; create
a documented new run if the configuration changes.

This setup is designed to produce a fair behavioral pilot, not a guaranteed positive outcome. Before a final paper experiment, add a semantic near-duplicate audit, general-capability retention evaluation and an independent repeat seed. Keep this pilot's test results out of subsequent tuning decisions.
