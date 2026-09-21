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

Evaluation only—this notebook never trains or modifies the DPO checkpoint. It
uses three matched stochastic generations per prompt, three independent local
judge models, forward/reversed A/B judging, strict majority decisions, and
prompt-level bootstrap intervals. Outputs resume safely in a new Drive folder;
the original test results remain unchanged.
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
PAIRS = DRIVE_RUN / "reproducibility/dpo_pairs_1500.jsonl"
BASELINE_MODEL = "microsoft/Phi-4-mini-instruct"
DPO_MODEL = DRIVE_RUN / "selected_lr1e6_b010_ep1/model"

assert PAIRS.exists(), f"Missing saved pairs: {PAIRS}"
assert (DPO_MODEL / "config.json").exists(), f"Missing DPO config: {DPO_MODEL}"
weights = list(DPO_MODEL.glob("*.safetensors"))
assert weights and sum(path.stat().st_size for path in weights) > 100 * 1024 * 1024, "DPO weights are missing."

EVAL_ROOT = DRIVE_RUN / "test/enhanced-fresh-evaluation-v1"
EVAL_ROOT.mkdir(parents=True, exist_ok=True)
GENERATION_SEEDS = [101, 202, 303]
TEMPERATURE = 0.7
TOP_P = 0.9
JUDGE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct",
]
print({"run": RUN_NAME, "weights_gb": sum(p.stat().st_size for p in weights) / 1e9,
       "seeds": GENERATION_SEEDS, "judges": JUDGE_MODELS})
"""
    ),
    markdown(
        """## 3. Generate matched responses

Each prompt receives a deterministic prompt-specific random seed. Baseline and
DPO therefore use the same random seed for each matched comparison. Existing
complete files are reused after a disconnect.
"""
    ),
    code(
        """for seed in GENERATION_SEEDS:
    seed_dir = EVAL_ROOT / "generations" / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for name, model in [("baseline", BASELINE_MODEL), ("dpo", str(DPO_MODEL))]:
        output = seed_dir / f"{name}.jsonl"
        subprocess.run([
            "sdt-generate-responses", "--pairs", str(PAIRS), "--model", model,
            "--output", str(output), "--split", "test", "--max-input-length", "1536",
            "--max-new-tokens", "512", "--temperature", str(TEMPERATURE),
            "--top-p", str(TOP_P), "--seed", str(seed),
        ], check=True)
        print("Complete:", output)
"""
    ),
    markdown("## 4. Verify matched coverage\n"),
    code(
        """expected_ids = None
for seed in GENERATION_SEEDS:
    seed_dir = EVAL_ROOT / "generations" / f"seed-{seed}"
    baseline = {json.loads(x)["prompt_id"] for x in (seed_dir / "baseline.jsonl").read_text().splitlines() if x.strip()}
    dpo = {json.loads(x)["prompt_id"] for x in (seed_dir / "dpo.jsonl").read_text().splitlines() if x.strip()}
    assert baseline == dpo and baseline
    expected_ids = baseline if expected_ids is None else expected_ids
    assert baseline == expected_ids, "Generation seeds covered different prompts."
    print(f"Seed {seed}: {len(baseline)} matched prompts")
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
        generation_dir = EVAL_ROOT / "generations" / f"seed-{generation_seed}"
        output_dir = EVAL_ROOT / "judgments" / slug / f"seed-{generation_seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for orientation, reverse in [("forward", False), ("reverse", True)]:
            details = output_dir / f"{orientation}.jsonl"
            summary = output_dir / f"{orientation}-summary.json"
            failures = output_dir / f"{orientation}-failures.jsonl"
            command = [
                "sdt-judge-generations-local",
                "--baseline", str(generation_dir / "baseline.jsonl"),
                "--dpo", str(generation_dir / "dpo.jsonl"),
                "--details", str(details), "--summary", str(summary),
                "--failures", str(failures), "--judge-model", judge_model,
                "--seed", "42", "--max-new-tokens", "512", "--max-retries", "3",
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
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
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
        """required = [MANIFEST, FINAL_SUMMARY, PROMPT_DETAILS]
required += [
    EVAL_ROOT / "generations" / f"seed-{seed}" / f"{name}.jsonl"
    for seed in GENERATION_SEEDS for name in ("baseline", "dpo")
]
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
