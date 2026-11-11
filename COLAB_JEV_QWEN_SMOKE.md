# Real Jev / Qwen full-DPO Colab test

Completed 2026-11-10 through the Colab browser on a Tesla T4.

## Run and artifacts

- Live executed notebook: https://colab.research.google.com/drive/1vLgyhxfMI20hKLSy8z_Oqlu-IY3ND687
- Drive run: `MyDrive/SDT_JEV_Colab_v1/qwen05-full-fp16-20261110T160112Z`
- Model: `trained/`; resumable checkpoint: `trained/checkpoint-24/`.
- Final weights: `trained/model.safetensors`, **1,976,163,472 bytes**.
- SHA256: `04f4af03c2840ae9f59f43b63fc8c81d1f3f7c5d0fd7dd404bef085921ccce22`.
- Source bundle, precision-patched source files, sample pairs/report, resolved configuration, training log, checkpoint-change report, validation metrics, test comparison, per-pair results, two generations/model and completion inventory were saved in the run.
- Drive uploads were flushed, Drive unmounted/remounted, and final weight size and SHA256 rechecked successfully. This is stronger evidence than checking a mounted-path file before flushing.

Use the generated local `notebooks/SDT_Jev_Qwen_Full_DPO_Colab.ipynb` with the rebuilt bundle for a clean sequential rerun. The live notebook retains diagnostic cells and the failed attempt for traceability; do not blindly Run All in that historical notebook.

## Configuration and sample

- `Qwen/Qwen2.5-0.5B-Instruct`: 494,032,768 parameters, all trainable; no LoRA.
- Learning rate 1e-6; beta 0.1; sigmoid DPO; one epoch; microbatch 1; accumulation 8; maximum sequence length 512; seed 42; 24 optimizer updates; one warmup update.
- FP32 trainable parameters with FP16 AMP on T4; cached reference log probabilities.
- Actual GPT/Opus Jev records, balanced by response generator, selected deterministically within the production prompt-hash splits.
- 64 training prompts / 187 pairs; 16 validation prompts / 48 pairs; 16 test prompts / 48 pairs. No further length drops at training.
- Sample deliberately restricted to short examples: a development integration test, not a representative full-dataset benchmark. Do not use its test outcomes for tuning and then call that same set a final confirmatory test.
- Recorded stack: Python 3.13.16, torch 2.14.1+cu130, transformers 5.5.3, TRL 1.17.1, datasets 4.8.5, accelerate 1.15.0.

## Observed results

| Metric | Value |
| --- | --- |
| Trainer training runtime (includes epoch-end overhead) | 113.3 seconds |
| Mean training loss | 0.5148 |
| Validation DPO loss | 0.640694 |
| Validation implicit preference accuracy | 38/48 = 79.17% |
| Test implicit preference accuracy | 35/48 = 72.92% |
| Test prompt-cluster 95% accuracy interval | 58.33–85.42% |
| Mean test implicit reward margin | 0.72677 |
| Margin prompt-cluster 95% interval | 0.22874–1.19669 |
| Baseline absolute length-normalized ranking accuracy | 15/48 = 31.25% |
| DPO absolute length-normalized ranking accuracy | 13/48 = 27.08% |
| Absolute accuracy difference | −4.17 percentage points |
| Prompt-cluster 95% interval for accuracy difference | −14.58 to +4.17 percentage points |
| Baseline-correct → DPO-wrong / reverse | 3 / 1 |
| Exact McNemar p | 0.625 |
| Mean absolute margin, baseline → DPO | −0.28431 → −0.24123 |

All 48 test pairs had changed completion log probabilities. Both models were loaded in separate evaluation processes; baseline and DPO each generated two held-out responses in separate generation processes. The embedding-weight comparison also found changed weights. Generated responses were not judged for SDT quality in this smoke test.

The implicit metric means the chosen-versus-rejected probability ratio moved in the preferred direction relative to the starting model. It does **not** mean 72.92% of generated responses are SDT-compliant. Absolute ranking did not improve. This verifies functioning optimization, save/reload, paired evaluation and generation, not successful end-task alignment or capability retention.

## Fixes and checks

1. Modern PyTorch reported BF16 support on T4 despite lack of native BF16 GEMM support; TRL backward failed before the first update. Precision selection now also requires CUDA compute capability >=8 for BF16. FP16 kernel tested explicitly on the T4; corrected run completed.
2. Retained FP32 master parameters for FP16 AMP, avoiding half-precision gradient-unscale errors.
3. Editable-install imports made explicit in notebook setup; upload uses returned bytes, avoiding filename-renaming ambiguity.
4. Archived 100-record fixture is not distributed in the bundle. Only its fixture-dependent test was excluded remotely; all other 53 tests passed. Updated local suite: 55 tests passed, including a native-BF16 regression test.
5. Failed run remains separate at `qwen05-full-20261110T155614Z`.
6. Reusable notebook includes Drive flush/remount/checksum verification. The bundle was rebuilt with corrected source.

## Before Ibex

The single-GPU path works on real Jev data. Multi-GPU launch/sharding, long-sequence memory use, full-dataset throughput, and restart on Ibex still need a small cluster smoke test before launching the main experiment. Final claims require generated-response SDT evaluation plus general-capability retention checks on prespecified held-out sets.
