# Local end-to-end DPO test — 2026-11-10

## Release recheck — 2026-11-11

The current `jev-dataset-adaptation` release was rechecked with **58 passing
unit/regression tests**. A fresh offline integration run at
`outputs/smoke-jev-ez_mcvop/smoke-report.json` also passed training, weight-change
verification, step-2-to-step-4 resume, paired evaluation, generation, and copied
backup reload. Resumed versus uninterrupted weights matched exactly; backup
evaluation/generation matched exactly. This is synthetic CPU evidence, not a
new research result. Historical observations below are retained for provenance.

**PASS, on CPU.** This is a software integration test, not an SDT experiment.
It uses a locally constructed random Qwen2 model (27,936 parameters), a local
tokenizer and synthetic Jev-shaped records. No research checkpoints were overwritten.

## Reproduce

From the repository root, with the project dependencies installed:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -q
PYTHONPATH=src python3 scripts/smoke_jev_pipeline.py
```

The environment used here is `.venv-smoke/bin/python`; core tested package versions
are in `constraints-smoke-tested.txt`. Install using
`python -m pip install -c constraints-smoke-tested.txt -e .` in a suitable environment.
This does not establish compatibility with Ibex's CUDA/driver/DeepSpeed versions.

Each invocation creates a new ignored `outputs/smoke-jev-*` directory. It preserves
baseline and trained weights, resumable checkpoints, a copied backup, configuration,
pair data, subprocess logs, evaluation outputs, generated responses and a JSON report.
The script runs on CPU intentionally; it does not need a pretrained-model download.

## Observed results

Final complete run: `outputs/smoke-jev-cmfxe8o6/smoke-report.json`.

| Check | Result |
|---|---|
| Unit/regression tests | 54 passed |
| Actual TRL DPO training | 4 optimizer updates on 8 eligible training pairs |
| Validation / test | 4 pairs each, separately partitioned |
| Training losses | 0.693147, 0.667248, 0.650452, 0.648306; all finite |
| Weights changed from baseline | Confirmed by checkpoint verifier |
| Overlength canary | 1 excluded before training, as intended |
| Resume from intermediate checkpoint | Step 2 resumed through step 4 |
| Resumed vs uninterrupted step-4 weights | Maximum absolute difference 0.0 |
| Selected final checkpoint reload | Successful |
| Copied backup weights | SHA-256 matched original |
| Backup greedy generation | Exact match to original checkpoint |
| Backup fixed-pair evaluation | Exact match to original checkpoint |
| Baseline/DPO comparison | Same test pair IDs; comparison command completed |
| Fresh-generation commands | Both completed on the same 2 unique test prompts |

Weight SHA-256:
`ae32dc4bf5eba69e9a879a7e152598723d5960bd87dbe4452cac53a3d7f7594b`

## Fixes made during the test

- Explicitly request token lists for training length checks. Transformers 5.5.3
  otherwise returns a BatchEncoding whose length is two fields, not token count.
- Added CPU and max-step configuration controls for tiny real integration tests.
- Warmup calculation honors max_steps when supplied.
- Fresh-generation prompt selection now follows the same Jev eligibility rules
  as training and evaluation; legacy margin defaults cannot drop quality-gate pairs.
- Added regression tests for both length filtering and Jev generation selection.

## Boundaries of this evidence

This verifies the real training/evaluation software path on a tiny model. It does
not validate Phi-scale memory use, mixed precision, multi-GPU, SDT generalization,
or Jev judging. Resume used a fully saved intermediate checkpoint, not a damaged
or half-written checkpoint. Backup was a local filesystem copy, not Google Drive
or remote Ibex storage. Repeat a short real-model GPU run and archive/reload check
on Ibex before the full experiment. See JEV_FULL_DATASET.md for remaining gates.
