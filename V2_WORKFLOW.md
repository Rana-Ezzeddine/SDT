# SDT DPO pilot V2 workflow

V2 separates irreversible training/generation from expensive judging and stores
both workflows under a new Google Drive root:

`MyDrive/SDT_DPO_Pilot_V2/<RUN_NAME>/`

## Notebook 1: training and generation

Run `notebooks/SDT_1500_DPO_Pilot_V2_Training_Colab.ipynb` from top to bottom.
It:

1. records the Git commit and Python environment;
2. stores the raw JSON, prepared pairs, pair audit, and resolved configuration;
3. trains the frozen Phi-4-mini full-DPO configuration with checkpoints on Drive;
4. hashes every final model-weight file;
5. reloads the model from Drive and performs a smoke generation;
6. compares baseline and DPO on validation;
7. opens the locked fixed-pair test once; and
8. generates matched baseline/DPO responses for seeds 42, 202, and 303.

The final cell creates `artifact-manifest.json`. A run without that file is not
complete. If Colab loses its runtime, rerun the notebook and set
`RESUME_RUN_NAME` in cell 2 to the already-created Drive run name. Training uses
the latest complete Trainer checkpoint when one exists.

## Notebook 2: fresh-response evaluation

After Notebook 1 passes its final audit, run
`notebooks/SDT_1500_DPO_Pilot_V2_Evaluation_Colab.ipynb`. Paste the exact
`RUN_NAME`, or leave it blank to select the newest completed V2 run.

The notebook verifies model-file sizes, verifies all matched generation files,
and judges every response pair using three stronger local instruct models. Each
judge sees both A/B orders. A failed or order-inconsistent vote cannot create a
win. Decisions are aggregated across judges for each generation seed and then
across the three seeds for each unique prompt.

## Completion evidence

The following files are the minimum evidence that the experiment is preserved:

- `model-integrity.json`
- `training/model/*.safetensors`
- `checkpoint-reload-smoke-test.json`
- `checkpoint-change.json`
- `validation/comparison.json`
- `test/fixed-pair-comparison.json`
- six generation JSONL files under `test/fresh-generations/`
- `artifact-manifest.json`
- `test/fresh-evaluation-v2/fresh-evaluation-summary.json`
- `test/fresh-evaluation-v2/evaluation-artifact-manifest.json`

Do not delete the V1 run. It is the historical pilot and provides an audit trail
for why V2 added stronger persistence and evaluation controls.
