# Pretrained model + real Jev data integration check

This is separate from the random-model unit/integration test and from the final
Phi/Gemma/Qwen experiment. It uses `HuggingFaceTB/SmolLM2-135M-Instruct` and actual
records from the supplied commercial-generator datasets, with no rewritten labels.

Run from the repository root with the installed project environment:

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=src .venv-smoke/bin/python scripts/pretrained_jev_smoke.py \
  --input /Users/ranaezzeddine/Downloads/sdt_gpt_fulljev_CLEAN.json \
  --input /Users/ranaezzeddine/Downloads/sdt_opus_fulljev_CLEAN.json
```

- The first run downloads approximately 269 MB of pretrained model weights.
- Uses the existing quality-gate/SDT ranking logic and original hash-based splits.
- Samples 4 training, 2 validation, and 2 test prompts from each input file.
- Keeps all eligible pairs for each sampled prompt, provided both full responses
  fit within 256 tokens. Text is not shortened to make it fit.
- One epoch, full parameters, learning rate 1e-6, beta 0.1, microbatch 1,
  gradient accumulation 4. No search or hyperparameter selection on these test results.
- Uses MPS float32 on a compatible Mac, otherwise the trainer's normal device selection.
- Runs actual training, checkpoint-change verification, baseline/DPO fixed-pair
  evaluation and two short fresh generations per model (not judged).
- Writes a new `outputs/pretrained-jev-*` directory per run with source hashes,
  immutable downloaded-model snapshot path, selected pairs, config, subprocess
  logs, model weights, tokenizer, optimizer checkpoint, metrics, generations,
  comparison, and successful-completion record with weight hashes.

Do not interpret these few short examples as an unbiased benchmark or claim they
predict Phi performance. Retain the sampled prompt IDs as development exposure;
if their outcomes guide later changes, exclude them from a final confirmatory test.
The local test does not validate CUDA, distributed training, or remote backup.

## Completed local run — 2026-11-10

Artifacts: `outputs/pretrained-jev-nji0bm_o/`. All stages passed.

- Actual sample: 24 training pairs / 12 validation pairs / 12 test pairs,
  from 8 / 4 / 4 separate prompts respectively.
- Full training: 134,515,008 trainable parameters, six optimizer updates, one epoch.
  Reported training runtime 46.62 seconds (excludes download, reference caching,
  model loading, and later evaluation).
- Training loss (run average): 0.675575. Validation loss: 0.698051;
  validation implicit reward accuracy: 4/12 = 33.3%.
- Test relative movement: 8/12 pairs (66.7%) shifted toward the chosen response
  relative to the starting model; mean implicit margin 0.009159.
- Absolute stored-response ranking: baseline 3/12, DPO 3/12 (25% both).
  No pair's absolute ranking flipped. This is NOT generated-answer correctness.
- All 12 evaluated pairs had changed completion log probabilities; none skipped.
- Saved weights reloaded in independent evaluation and generation processes.
  Two test prompts were generated from each model, without an LLM judge.
- `completion.json` records the weight SHA-256. Weights, tokenizer, optimizer
  checkpoint, sample/config, and logs are local in the artifacts directory.

Interpretation: pipeline validation succeeded. There is no demonstrated absolute
accuracy improvement; four test prompts and a short-example sample cannot establish
SDT generalization. Neither the 66.7% relative movement nor this model's poor absolute
ranking should be extrapolated to the full dataset or Phi. No hyperparameter search
was performed. This is a development smoke test, not a publishable result.
