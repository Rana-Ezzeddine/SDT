"""Build the standalone enhanced fresh-response evaluation notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/SDT_Enhanced_Fresh_Response_Evaluation_Colab.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown(
        """# Enhanced fresh-response evaluation

Evaluation only—this notebook never trains or loads the lost DPO checkpoint. It
reuses the 111 paired baseline and DPO responses already saved in Drive, then
applies three independent local judges, forward/reversed A/B judging, strict
majority decisions, and prompt-level bootstrap intervals. Outputs resume safely
in a new Drive folder; the original test results remain unchanged.
"""
    ),
    markdown("## 1. GPU and evaluation code\n"),
    code(
        """import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import subprocess, sys, json, hashlib, re
from pathlib import Path
import torch

assert torch.cuda.is_available(), "Select Runtime > Change runtime type > GPU."
subprocess.run(["nvidia-smi"], check=True)

REPO_URL = "https://github.com/Rana-Ezzeddine/SDT.git"
BRANCH = "1500-record-dpo-pipeline"
REPO_DIR = Path("/content/SDT")
if REPO_DIR.exists():
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "switch", BRANCH], cwd=REPO_DIR, check=True)
    subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=REPO_DIR, check=True)
else:
    subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
print("Commit:", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
"""
    ),
    markdown("## 2. Locate the saved run and configure the evaluation\n"),
    code(
        """from google.colab import drive
drive.mount("/content/drive")

RUN_NAME = "pilot-1500-e8058f88d1c9-9806f0c8-65559296-3bb7250-20260921T055400Z"
DRIVE_RUN = Path("/content/drive/MyDrive/SDT_DPO_Pilot") / RUN_NAME
BASELINE_GENERATIONS = DRIVE_RUN / "test/baseline-generations.jsonl"
DPO_GENERATIONS = DRIVE_RUN / "test/dpo-generations.jsonl"
assert BASELINE_GENERATIONS.exists(), f"Missing saved baseline responses: {BASELINE_GENERATIONS}"
assert DPO_GENERATIONS.exists(), f"Missing saved DPO responses: {DPO_GENERATIONS}"

EVAL_ROOT = DRIVE_RUN / "test/enhanced-existing-generations-evaluation-v1"
EVAL_ROOT.mkdir(parents=True, exist_ok=True)
GENERATION_SEEDS = [42]
JUDGE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct",
]
print({"run": RUN_NAME, "saved_generation_seed": GENERATION_SEEDS[0], "judges": JUDGE_MODELS})
"""
    ),
    markdown(
        """## 3. Verify the saved matched responses

These are the original fresh responses generated before the checkpoint was
lost. This notebook does not need the model weights and does not regenerate or
overwrite either file.
"""
    ),
    code(
        """def read_generation_ids(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [str(row["prompt_id"]) for row in rows]
    assert len(ids) == len(set(ids)), f"Duplicate prompt IDs in {path}"
    assert all(str(row.get("response", "")).strip() for row in rows), f"Empty response in {path}"
    return rows, set(ids)

baseline_rows, baseline_ids = read_generation_ids(BASELINE_GENERATIONS)
dpo_rows, dpo_ids = read_generation_ids(DPO_GENERATIONS)
assert baseline_ids == dpo_ids and baseline_ids, "Baseline/DPO prompt coverage differs."
baseline_prompts = {str(row["prompt_id"]): str(row["prompt"]) for row in baseline_rows}
dpo_prompts = {str(row["prompt_id"]): str(row["prompt"]) for row in dpo_rows}
assert baseline_prompts == dpo_prompts, "Prompt text differs between saved generation files."
print("Verified paired saved responses:", len(baseline_ids))
"""
    ),
    markdown("## 4. Inspect response overlap\n"),
    code(
        """baseline_by_id = {str(row["prompt_id"]): " ".join(str(row["response"]).split()) for row in baseline_rows}
dpo_by_id = {str(row["prompt_id"]): " ".join(str(row["response"]).split()) for row in dpo_rows}
identical = sum(baseline_by_id[prompt_id] == dpo_by_id[prompt_id] for prompt_id in baseline_ids)
print({"prompts": len(baseline_ids), "identical_responses": identical,
       "responses_requiring_judgment": len(baseline_ids) - identical})
"""
    ),
    markdown(
        """## 5. Blind three-judge panel with position reversal

Every judge sees each response pair twice: once in its deterministic randomized
A/B order and once with the positions reversed. A judge vote contributes to a
win only if it selects the same underlying model in both orientations.
"""
    ),
    code(
        """def judge_slug(index, model):
    short = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[-48:]
    return f"judge-{index + 1}-{short}"

manifest_entries = []
for judge_index, judge_model in enumerate(JUDGE_MODELS):
    slug = judge_slug(judge_index, judge_model)
    for generation_seed in GENERATION_SEEDS:
        output_dir = EVAL_ROOT / "judgments" / slug / f"seed-{generation_seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for orientation, reverse in [("forward", False), ("reverse", True)]:
            details = output_dir / f"{orientation}.jsonl"
            summary = output_dir / f"{orientation}-summary.json"
            failures = output_dir / f"{orientation}-failures.jsonl"
            command = [
                "sdt-judge-generations-local",
                "--baseline", str(BASELINE_GENERATIONS),
                "--dpo", str(DPO_GENERATIONS),
                "--details", str(details), "--summary", str(summary),
                "--failures", str(failures), "--judge-model", judge_model,
                "--seed", "42", "--max-new-tokens", "512", "--max-retries", "3",
                "--continue-on-failure",
            ]
            if reverse:
                command.append("--reverse-order")
            subprocess.run(command, check=True)
            files[orientation] = str(details)
        manifest_entries.append({
            "generation_seed": generation_seed,
            "judge_model": judge_model,
            "forward": files["forward"],
            "reverse": files["reverse"],
        })

MANIFEST = EVAL_ROOT / "evaluation-manifest.json"
MANIFEST.write_text(json.dumps({
    "generation_seeds": GENERATION_SEEDS,
    "judge_models": JUDGE_MODELS,
    "source": "saved_original_fresh_generations",
    "position_reversal": True,
    "judgments": manifest_entries,
}, indent=2) + "\\n")
print("Saved", MANIFEST)
"""
    ),
    markdown("## 6. Aggregate strictly at the prompt level\n"),
    code(
        """FINAL_SUMMARY = EVAL_ROOT / "enhanced-summary.json"
PROMPT_DETAILS = EVAL_ROOT / "enhanced-prompt-details.jsonl"
subprocess.run([
    "sdt-aggregate-fresh-evaluation", "--manifest", str(MANIFEST),
    "--output", str(FINAL_SUMMARY), "--details", str(PROMPT_DETAILS),
    "--bootstrap-samples", "10000",
], check=True)
report = json.loads(FINAL_SUMMARY.read_text())
"""
    ),
    markdown("## 7. Results\n"),
    code(
        """import pandas as pd
import matplotlib.pyplot as plt

display(pd.DataFrame([{
    "prompts": report["n_prompts"],
    "DPO wins": report["prompt_level_dpo_wins"],
    "Baseline wins": report["prompt_level_baseline_wins"],
    "Ties": report["prompt_level_ties"],
    "Tie-adjusted DPO score": report["prompt_macro_tie_adjusted_dpo_score"],
    "95% low": report["prompt_bootstrap_95"][0],
    "95% high": report["prompt_bootstrap_95"][1],
}]))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(["DPO wins", "Baseline wins", "Ties"],
            [report["prompt_level_dpo_wins"], report["prompt_level_baseline_wins"], report["prompt_level_ties"]],
            color=["#2E86AB", "#D1495B", "#9E9E9E"])
axes[0].set_title("Prompt-level outcomes")
dimensions = report["dimension_deltas"]
names = list(dimensions)
values = [dimensions[name]["mean_dpo_minus_baseline"] for name in names]
axes[1].barh(names, values, color=["#2E86AB" if value >= 0 else "#D1495B" for value in values])
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Mean dimension delta: DPO − baseline")
plt.tight_layout()
plt.show()

display(pd.DataFrame([
    {"judge": judge, **metrics}
    for judge, metrics in report["judge_diagnostics"].items()
]))
"""
    ),
    markdown("## 8. Final artifact audit\n"),
    code(
        """required = [BASELINE_GENERATIONS, DPO_GENERATIONS, MANIFEST, FINAL_SUMMARY, PROMPT_DETAILS]
required += [Path(entry[key]) for entry in manifest_entries for key in ("forward", "reverse")]
for path in required:
    assert path.exists() and path.stat().st_size > 0, f"Missing: {path}"
print(f"Enhanced evaluation complete: {len(required)} core artifacts verified")
print("Results:", FINAL_SUMMARY)
print("All outputs:", EVAL_ROOT)
os.sync()
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
