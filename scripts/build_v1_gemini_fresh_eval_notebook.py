"""Build a concise, evaluation-only Colab notebook for the saved V1 generations."""

from __future__ import annotations

import json
from pathlib import Path


OUTPUT = Path("notebooks/SDT_V1_Gemini_Fresh_Response_Evaluation_Colab.ipynb")


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


cells = [
    markdown(
        """# V1 fresh-response rescue evaluation

This notebook **does not retrain or regenerate**. It reuses the 111 matched
baseline/DPO responses already saved from the completed V1 pilot. Gemini judges
each non-identical pair blindly in both A/B orders using the current concise
rubric. A model win is accepted only when the same underlying winner is selected
after the positions are swapped.

The result is a supplementary pilot evaluation: one generation seed and one
independent judge. Report the judge identity, identical-response count,
position-consistency rate, ties, and uncertainty interval with any slide result.
"""
    ),
    markdown("## 1. Mount Drive and install the current branch\n"),
    code(
        """from google.colab import drive
drive.mount('/content/drive')

import os, subprocess, sys
from pathlib import Path

REPO_URL = "https://github.com/Rana-Ezzeddine/SDT.git"
BRANCH = "1500-record-dpo-pipeline"
REPO = Path("/content/SDT")
if not REPO.exists():
    subprocess.run(["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO)], check=True)
else:
    subprocess.run(["git", "fetch", "origin", BRANCH], cwd=REPO, check=True)
    subprocess.run(["git", "switch", BRANCH], cwd=REPO, check=True)
    subprocess.run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=REPO, check=True)
os.chdir(REPO)
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
GIT_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("Commit:", GIT_COMMIT)
"""
    ),
    markdown("## 2. Select the preserved V1 run and verify the matched responses\n"),
    code(
        """import json, hashlib

RUN_NAME = "pilot-1500-e8058f88d1c9-9806f0c8-65559296-3bb7250-20260921T055400Z"
DRIVE_RUN = Path("/content/drive/MyDrive/SDT_DPO_Pilot") / RUN_NAME
TEST_DIR = DRIVE_RUN / "test"
BASELINE = TEST_DIR / "baseline-generations.jsonl"
DPO = TEST_DIR / "dpo-generations.jsonl"
EVAL_ROOT = TEST_DIR / "fresh-evaluation-v4-gemini"
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

def read_rows(path):
    assert path.exists(), f"Missing: {path}"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows and len({str(row["prompt_id"]) for row in rows}) == len(rows)
    assert all(str(row.get("response", "")).strip() for row in rows)
    return {str(row["prompt_id"]): row for row in rows}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

baseline, dpo = read_rows(BASELINE), read_rows(DPO)
assert set(baseline) == set(dpo)
assert all(str(baseline[p]["prompt"]) == str(dpo[p]["prompt"]) for p in baseline)
identical = sum(" ".join(str(baseline[p]["response"]).split()) == " ".join(str(dpo[p]["response"]).split()) for p in baseline)
print({
    "prompts": len(baseline), "identical_responses": identical,
    "responses_requiring_judgment": len(baseline) - identical,
    "baseline_sha256": sha256(BASELINE), "dpo_sha256": sha256(DPO),
})
"""
    ),
    markdown(
        """## 3. Configure the independent Gemini judge

Create a Gemini API key in Google AI Studio, add it to Colab Secrets as
`GEMINI_API_KEY`, and enable notebook access. This evaluation uses an API and
does not require a GPU. If the named model is unavailable to your account,
replace `JUDGE_MODEL` with a current model shown in AI Studio before starting;
do not change it midway through a run.
"""
    ),
    code(
        """from google.colab import userdata

JUDGE_MODEL = "gemini-3.8-flash"
API_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
api_key = userdata.get("GEMINI_API_KEY")
assert api_key, "Add GEMINI_API_KEY to Colab Secrets and enable notebook access."
os.environ["JUDGE_API_KEY"] = api_key
print({"judge": JUDGE_MODEL, "api": API_URL, "gpu_required": False})
"""
    ),
    markdown(
        """## 4. Judge forward and reversed orders

This cell is resumable: each completed prompt is appended immediately to Drive.
There are 22 deterministic identical-response ties in the saved sample, so only
the remaining pairs call the API. Failed calls are recorded as unavailable and
cannot create a win.
"""
    ),
    code(
        """import subprocess

judgment_files = {}
for orientation, reverse in [("forward", False), ("reverse", True)]:
    details = EVAL_ROOT / f"{orientation}.jsonl"
    summary = EVAL_ROOT / f"{orientation}-summary.json"
    failures = EVAL_ROOT / f"{orientation}-failures.jsonl"
    command = [
        "sdt-judge-generations",
        "--baseline", str(BASELINE), "--dpo", str(DPO),
        "--details", str(details), "--summary", str(summary),
        "--failures", str(failures), "--judge-model", JUDGE_MODEL,
        "--api-url", API_URL, "--api-key-env", "JUDGE_API_KEY",
        "--seed", "42", "--timeout", "180", "--max-retries", "5",
        "--retry-backoff", "3", "--continue-on-failure",
    ]
    if reverse:
        command.append("--reverse-order")
    print("Running", orientation)
    subprocess.run(command, check=True)
    judgment_files[orientation] = str(details)
"""
    ),
    markdown("## 5. Aggregate only position-consistent judgments\n"),
    code(
        """MANIFEST = EVAL_ROOT / "evaluation-manifest.json"
MANIFEST.write_text(json.dumps({
    "run_name": RUN_NAME,
    "generation_seeds": [42],
    "judge_models": [JUDGE_MODEL],
    "position_reversal": True,
    "source": "saved_v1_generations",
    "judgments": [{
        "generation_seed": 42,
        "judge_model": JUDGE_MODEL,
        "forward": judgment_files["forward"],
        "reverse": judgment_files["reverse"],
    }],
}, indent=2) + "\n")

SUMMARY = EVAL_ROOT / "fresh-evaluation-summary.json"
DETAILS = EVAL_ROOT / "fresh-evaluation-prompt-details.jsonl"
subprocess.run([
    "sdt-aggregate-fresh-evaluation", "--manifest", str(MANIFEST),
    "--output", str(SUMMARY), "--details", str(DETAILS),
    "--bootstrap-samples", "10000",
], check=True)
report = json.loads(SUMMARY.read_text())
print(json.dumps(report, indent=2))
"""
    ),
    markdown("## 6. Slide-ready result and figure\n"),
    code(
        """import pandas as pd
import matplotlib.pyplot as plt

dpo_wins = report["prompt_level_dpo_wins"]
baseline_wins = report["prompt_level_baseline_wins"]
ties = report["prompt_level_ties"]
decisive = dpo_wins + baseline_wins
diagnostics = report["judge_diagnostics"][JUDGE_MODEL]

result_table = pd.DataFrame([{
    "Prompts": report["n_prompts"],
    "DPO wins": dpo_wins,
    "Baseline wins": baseline_wins,
    "Ties/unresolved": ties,
    "DPO win rate among decisive": dpo_wins / decisive if decisive else None,
    "Tie-adjusted DPO score": report["prompt_macro_tie_adjusted_dpo_score"],
    "95% interval low": report["prompt_bootstrap_95"][0],
    "95% interval high": report["prompt_bootstrap_95"][1],
    "Position consistency": diagnostics.get("position_consistency_rate"),
}])
display(result_table)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].bar(["DPO wins", "Baseline wins", "Ties/unresolved"],
            [dpo_wins, baseline_wins, ties],
            color=["#2E86AB", "#D1495B", "#9E9E9E"])
axes[0].set_ylabel("Unique test prompts")
axes[0].set_title("Fresh-response outcomes")
dimensions = report["dimension_deltas"]
names = list(dimensions)
values = [dimensions[name]["mean_dpo_minus_baseline"] or 0 for name in names]
axes[1].barh(names, values,
             color=["#2E86AB" if value >= 0 else "#D1495B" for value in values])
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Judge score: DPO minus baseline")
plt.tight_layout()
FIGURE = EVAL_ROOT / "fresh-evaluation-overview.png"
plt.savefig(FIGURE, dpi=200, bbox_inches="tight")
plt.show()

slide_summary = {
    "scope": "V1 saved fresh responses; one generation seed; one Gemini judge; both A/B orders",
    "n_prompts": report["n_prompts"],
    "dpo_wins": dpo_wins, "baseline_wins": baseline_wins,
    "ties_or_unresolved": ties,
    "tie_adjusted_dpo_score": report["prompt_macro_tie_adjusted_dpo_score"],
    "prompt_bootstrap_95": report["prompt_bootstrap_95"],
    "position_consistency_rate": diagnostics.get("position_consistency_rate"),
    "interpretation_rule": (
        "Call this positive only if the DPO score is above 0.5 and its 95% interval "
        "does not include 0.5; otherwise describe it as inconclusive or negative."
    ),
}
(EVAL_ROOT / "slide-summary.json").write_text(json.dumps(slide_summary, indent=2) + "\n")
print(json.dumps(slide_summary, indent=2))
"""
    ),
    markdown("## 7. Final artifact audit\n"),
    code(
        """required = [
    Path(judgment_files["forward"]), Path(judgment_files["reverse"]),
    MANIFEST, SUMMARY, DETAILS, FIGURE, EVAL_ROOT / "slide-summary.json",
]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
assert not missing, "Missing artifacts:\n" + "\n".join(missing)
audit = {
    "git_commit": GIT_COMMIT,
    "run_name": RUN_NAME,
    "judge_model": JUDGE_MODEL,
    "artifacts": [{"name": path.name, "bytes": path.stat().st_size} for path in required],
}
(EVAL_ROOT / "evaluation-artifact-manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
os.sync()
print("Evaluation complete and saved to:", EVAL_ROOT)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "NONE",
        "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(OUTPUT)
