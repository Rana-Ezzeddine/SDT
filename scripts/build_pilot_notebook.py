"""Build the concise, restart-safe Colab notebook for the 1,500-record pilot."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/SDT_1500_DPO_Pilot_Colab.ipynb"


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
        """# SDT 1,500-record DPO pilot

Reproduce the validation-selected full-parameter Phi-4-mini DPO run, verify it,
open the test once, generate paired answers, and optionally run a blinded LLM
judge. Drive backups are verified after every irreversible stage. Temporary step
checkpoints are excluded; the final selected model weights are required.
"""
    ),
    markdown("## 1. GPU, branch, and installation\n"),
    code(
        """import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import subprocess, sys
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
GIT_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip(), "Repository must be clean."
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
print("Branch:", subprocess.check_output(["git", "branch", "--show-current"], text=True).strip())
print("Commit:", GIT_COMMIT)
"""
    ),
    markdown("## 2. Upload the updated raw JSON\n\nUpload only `sdt_1500_updated.json`.\n"),
    code(
        """from google.colab import files

uploaded = files.upload()
assert len(uploaded) == 1, "Upload only the updated 1,500-record JSON."
RAW_DATA = Path("data/raw/sdt_results_1500.json")
RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
RAW_DATA.write_bytes(next(iter(uploaded.values())))
print("Saved", RAW_DATA, "bytes=", RAW_DATA.stat().st_size)
"""
    ),
    markdown("## 3. Build provisional pairs and inspect the audit\n"),
    code(
        """import json

PAIRS = Path("data/processed/dpo_pairs_1500.jsonl")
PAIR_REPORT = Path("data/processed/pair_report_1500.json")
subprocess.run([
    "sdt-build-pairs", "--input", str(RAW_DATA), "--output", str(PAIRS),
    "--report", str(PAIR_REPORT), "--label-mode", "single_judge_pilot",
    "--use-supplied-aggregates", "--min-common-judges", "1",
    "--min-margin", "0.10", "--train-share", "0.80",
    "--validation-share", "0.10", "--seed", "42",
], check=True)
report = json.loads(PAIR_REPORT.read_text())
for key in ["records", "possible_pairs", "retained_pairs", "retained_prompts", "retained_by_split", "retained_by_comparison_type", "exclusion_reasons"]:
    print(f"{key}: {report.get(key)}")
print("judgment_evidence:", json.dumps(report["judgment_evidence"], indent=2))
"""
    ),
    markdown("## 4. Run tests and inspect Phi token lengths\n"),
    code(
        """subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)

import yaml
from transformers import AutoTokenizer

BASE_CONFIG_PATH = Path("configs/pilot_1500_phi.yaml")
base_config = yaml.safe_load(BASE_CONFIG_PATH.read_text())
BASELINE_MODEL = str(base_config["model_id"])
MAX_LENGTH = int(base_config["max_length"])
tokenizer = AutoTokenizer.from_pretrained(BASELINE_MODEL)
lengths = []
for line in PAIRS.read_text().splitlines():
    row = json.loads(line)
    if not row.get("retain"):
        continue
    for response in (row["chosen"], row["rejected"]):
        messages = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": response}]
        lengths.append(len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)))
print({"n": len(lengths), "max": max(lengths), "over_max_length": sum(x > MAX_LENGTH for x in lengths), "max_length": MAX_LENGTH})
"""
    ),
    markdown(
        """## 5. Reproduce the frozen validation winner

This notebook does not repeat the earlier hyperparameter sweep. It reproduces
the already selected `1e-6`, beta `0.10`, one-epoch configuration and loads the
best within-epoch validation checkpoint before saving the final model.
"""
    ),
    code(
        """from google.colab import drive

drive.mount("/content/drive")
DRIVE_BACKUP_ROOT = Path("/content/drive/MyDrive/SDT_DPO_Pilot")
DRIVE_BACKUP_ROOT.mkdir(parents=True, exist_ok=True)

EXPERIMENT = {
    "name": "selected_lr1e6_b010_ep1",
    "learning_rate": 1e-6,
    "beta": 0.10,
    "num_train_epochs": 1,
    "eval_strategy": "steps",
    "eval_steps": 96,
    "save_strategy": "steps",
    "save_steps": 96,
    "save_total_limit": 2,
    "metric_for_best_model": "eval_rewards/accuracies",
    "greater_is_better": True,
}
print("Frozen experiment:", EXPERIMENT)
"""
    ),
    code(
        """import hashlib, shutil
from datetime import datetime, timezone
import pandas as pd

def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def run_files(root):
    return [
        path for path in Path(root).rglob("*")
        if path.is_file() and not any(part.startswith("checkpoint-") for part in path.relative_to(root).parts)
    ]

def sync_run_to_drive(stage, require_model=False):
    destination = DRIVE_BACKUP_ROOT / RUN_ROOT.name
    destination.mkdir(parents=True, exist_ok=True)
    for source in run_files(RUN_ROOT):
        relative = source.relative_to(RUN_ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        same = target.exists() and target.stat().st_size == source.stat().st_size
        if not same:
            shutil.copy2(source, target)
    source_sizes = {str(path.relative_to(RUN_ROOT)): path.stat().st_size for path in run_files(RUN_ROOT)}
    for relative, size in source_sizes.items():
        target = destination / relative
        assert target.exists() and target.stat().st_size == size, f"Backup verification failed: {relative}"
    weights = []
    if require_model:
        saved_model = destination / Path(DPO_MODEL).relative_to(RUN_ROOT)
        assert (saved_model / "config.json").exists(), "Saved model config is missing"
        weights = sorted(saved_model.glob("*.safetensors"))
        assert weights, "Saved model weights are missing"
        assert sum(path.stat().st_size for path in weights) > 100 * 1024 * 1024, "Saved weights are unexpectedly small"
    verification = {
        "stage": stage,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": GIT_COMMIT,
        "run_name": RUN_ROOT.name,
        "file_count": len(source_sizes),
        "model_weight_files": [{"name": path.name, "bytes": path.stat().st_size} for path in weights],
    }
    (destination / "backup-verification.json").write_text(json.dumps(verification, indent=2) + "\\n")
    print("Verified Drive backup:", destination, "stage=", stage)
    return destination

data_fingerprint = sha256_file(RAW_DATA)[:12]
model_fingerprint = hashlib.sha256(BASELINE_MODEL.encode()).hexdigest()[:8]
resolved = dict(base_config)
resolved.update({k: v for k, v in EXPERIMENT.items() if k != "name"})
config_fingerprint = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()[:8]
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN_ROOT = Path("outputs") / f"pilot-1500-{data_fingerprint}-{model_fingerprint}-{config_fingerprint}-{GIT_COMMIT[:7]}-{timestamp}"
RUN_ROOT.mkdir(parents=True, exist_ok=False)

REPRO = RUN_ROOT / "reproducibility"
REPRO.mkdir()
shutil.copy2(RAW_DATA, REPRO / "sdt_1500_updated.json")
shutil.copy2(PAIRS, REPRO / "dpo_pairs_1500.jsonl")
shutil.copy2(PAIR_REPORT, REPRO / "pair_report_1500.json")
shutil.copy2(BASE_CONFIG_PATH, REPRO / "pilot_1500_phi.yaml")
shutil.copy2(Path("notebooks/SDT_1500_DPO_Pilot_Colab.ipynb"), REPRO / "SDT_1500_DPO_Pilot_Colab.ipynb")
(REPRO / "git-commit.txt").write_text(GIT_COMMIT + "\\n")
(REPRO / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
(REPRO / "input-sha256.json").write_text(json.dumps({
    "raw_data": sha256_file(RAW_DATA), "pairs": sha256_file(PAIRS), "pair_report": sha256_file(PAIR_REPORT)
}, indent=2) + "\\n")

model_dir = RUN_ROOT / EXPERIMENT["name"] / "model"
DPO_MODEL = str(model_dir)
resolved.update({"model_id": BASELINE_MODEL, "pairs_file": str(PAIRS.resolve()), "max_length": MAX_LENGTH, "output_dir": str(model_dir.resolve())})
config_path = REPRO / "resolved-training-config.yaml"
config_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
sync_run_to_drive("prepared", require_model=False)

baseline_val = RUN_ROOT / "baseline-validation.json"
baseline_val_details = RUN_ROOT / "baseline-validation-pairs.jsonl"
subprocess.run(["sdt-evaluate-pairs", "--pairs", str(PAIRS), "--model", BASELINE_MODEL,
                "--split", "validation", "--max-length", str(MAX_LENGTH), "--all-confidence",
                "--output", str(baseline_val), "--details", str(baseline_val_details)], check=True)
subprocess.run(["sdt-train-dpo", "--config", str(config_path)], check=True)

candidate_details = model_dir.parent / "validation-pairs.jsonl"
subprocess.run(["sdt-evaluate-pairs", "--pairs", str(PAIRS), "--model", DPO_MODEL,
                "--split", "validation", "--max-length", str(MAX_LENGTH), "--all-confidence",
                "--output", str(model_dir.parent / "validation-summary.json"), "--details", str(candidate_details)], check=True)
comparison_path = model_dir.parent / "validation-comparison.json"
subprocess.run(["sdt-compare-evaluations", "--baseline-details", str(baseline_val_details),
                "--dpo-details", str(candidate_details), "--output", str(comparison_path),
                "--beta", str(resolved["beta"])], check=True)

comparison = json.loads(comparison_path.read_text())
training_metrics = json.loads((model_dir / "validation_metrics.json").read_text())
run_manifest = json.loads((model_dir / "run_manifest.json").read_text())
primary = comparison["primary_dpo_relative_metrics"]
secondary = comparison["secondary_absolute_likelihood_metrics"]
selected = {**EXPERIMENT, "model": DPO_MODEL,
            "implicit_accuracy": primary["implicit_reward_accuracy"],
            "prompt_macro_implicit_accuracy": primary["implicit_reward_prompt_macro_accuracy"],
            "mean_implicit_margin": primary["mean_implicit_reward_margin"],
            "absolute_pair_delta": secondary["accuracy_delta_dpo_minus_baseline"],
            "absolute_prompt_macro_delta": secondary["prompt_macro_accuracy_delta"],
            "mean_absolute_margin_delta": secondary["mean_model_margin_delta"],
            "validation_loss": training_metrics.get("validation_loss"),
            "best_within_epoch_reward_accuracy": run_manifest.get("best_validation_metric"),
            "best_within_epoch_checkpoint": run_manifest.get("best_model_checkpoint")}
(RUN_ROOT / "selected-configuration.json").write_text(json.dumps(selected, indent=2, default=str) + "\\n")
display(pd.DataFrame([selected]))
DRIVE_RUN = sync_run_to_drive("validation_complete", require_model=True)
print("Frozen checkpoint:", DPO_MODEL)
"""
    ),
    markdown("## 6. Verify the checkpoint and its Drive copy\n"),
    code(
        """subprocess.run(["sdt-verify-checkpoint", "--baseline-model", BASELINE_MODEL,
                "--trained-model", DPO_MODEL, "--output", str(RUN_ROOT / "checkpoint-change.json")], check=True)
print(json.loads((RUN_ROOT / "checkpoint-change.json").read_text()))
DRIVE_RUN = sync_run_to_drive("checkpoint_verified", require_model=True)
"""
    ),
    markdown(
        """## 7. Locked fixed-pair test

The configuration is already frozen. Keep this `False` until the validation and
backup checks above pass, then set it to `True` and run this cell once.
"""
    ),
    code(
        """RUN_LOCKED_TEST = False
TEST_DIR = RUN_ROOT / "test"
if not RUN_LOCKED_TEST:
    print("Test remains locked. Set RUN_LOCKED_TEST=True after the validation backup is verified.")
else:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    for name, model in [("baseline", BASELINE_MODEL), ("dpo", DPO_MODEL)]:
        subprocess.run(["sdt-evaluate-pairs", "--pairs", str(PAIRS), "--model", model,
                        "--split", "test", "--max-length", str(MAX_LENGTH), "--all-confidence",
                        "--output", str(TEST_DIR / f"{name}-summary.json"),
                        "--details", str(TEST_DIR / f"{name}-pairs.jsonl")], check=True)
    subprocess.run(["sdt-compare-evaluations", "--baseline-details", str(TEST_DIR / "baseline-pairs.jsonl"),
                    "--dpo-details", str(TEST_DIR / "dpo-pairs.jsonl"),
                    "--output", str(TEST_DIR / "fixed-pair-comparison.json"),
                    "--details-output", str(TEST_DIR / "relative-pairs.jsonl"),
                    "--beta", str(resolved["beta"])], check=True)
    fixed = json.loads((TEST_DIR / "fixed-pair-comparison.json").read_text())
    print(json.dumps(fixed["primary_dpo_relative_metrics"], indent=2))
    print(json.dumps(fixed["secondary_absolute_likelihood_metrics"], indent=2))
    DRIVE_RUN = sync_run_to_drive("fixed_pair_test_complete", require_model=True)
"""
    ),
    markdown("## 8. Fresh baseline and DPO generations on the same test prompts\n"),
    code(
        """if not RUN_LOCKED_TEST:
    print("Skipped because the test is locked.")
else:
    for name, model in [("baseline", BASELINE_MODEL), ("dpo", DPO_MODEL)]:
        output = TEST_DIR / f"{name}-generations.jsonl"
        subprocess.run(["sdt-generate-responses", "--pairs", str(PAIRS), "--model", model,
                        "--output", str(output), "--split", "test",
                        "--max-input-length", str(MAX_LENGTH), "--max-new-tokens", "512",
                        "--temperature", "0", "--seed", "42"], check=True)
        DRIVE_RUN = sync_run_to_drive(f"{name}_generation_complete", require_model=True)
    baseline_ids = {json.loads(line)["prompt_id"] for line in (TEST_DIR / "baseline-generations.jsonl").read_text().splitlines() if line.strip()}
    dpo_ids = {json.loads(line)["prompt_id"] for line in (TEST_DIR / "dpo-generations.jsonl").read_text().splitlines() if line.strip()}
    assert baseline_ids and baseline_ids == dpo_ids
    print("Paired generated prompts:", len(baseline_ids))
"""
    ),
    markdown(
        """## 9. Optional free local LLM-as-a-judge

Run an independent Qwen judge locally on the Colab GPU, with no API key or
per-request API charge. Identical responses are recorded deterministically as
ties. Each completed judgment is written directly to Drive, so the cell can
resume after a disconnect when its model, settings, prompt version, and inputs
are unchanged. This still uses Colab GPU compute units.
"""
    ),
    code(
        """RUN_LOCAL_LLM_JUDGE = False
LOCAL_JUDGE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

if RUN_LOCAL_LLM_JUDGE:
    assert RUN_LOCKED_TEST and LOCAL_JUDGE_MODEL, "Generate the locked-test answers and choose a local judge."
    drive_test = DRIVE_RUN / "test"
    details = drive_test / "generation-judgments.jsonl"
    summary = drive_test / "generation-judge-summary.json"
    failures = drive_test / "generation-judge-failures.jsonl"
    subprocess.run(["sdt-judge-generations-local",
                    "--baseline", str(drive_test / "baseline-generations.jsonl"),
                    "--dpo", str(drive_test / "dpo-generations.jsonl"),
                    "--details", str(details), "--summary", str(summary),
                    "--failures", str(failures), "--judge-model", LOCAL_JUDGE_MODEL,
                    "--seed", "42", "--max-new-tokens", "512",
                    "--max-retries", "3"], check=True)
    print(summary.read_text())
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    for source in [details, summary, failures]:
        if source.exists():
            shutil.copy2(source, TEST_DIR / source.name)
    DRIVE_RUN = sync_run_to_drive("llm_judge_complete", require_model=True)
else:
    print("Local LLM judge disabled. Set RUN_LOCAL_LLM_JUDGE=True when ready.")
"""
    ),
    markdown("## 10. Final Drive audit\n"),
    code(
        """DRIVE_RUN = sync_run_to_drive("final_audit", require_model=True)
verification = json.loads((DRIVE_RUN / "backup-verification.json").read_text())
print(json.dumps(verification, indent=2))
required = [
    "reproducibility/sdt_1500_updated.json",
    "reproducibility/dpo_pairs_1500.jsonl",
    "reproducibility/pair_report_1500.json",
    "reproducibility/resolved-training-config.yaml",
    "selected-configuration.json",
    "checkpoint-change.json",
]
for relative in required:
    path = DRIVE_RUN / relative
    assert path.exists(), f"Missing required Drive artifact: {relative}"
    print("OK", relative, path.stat().st_size, "bytes")
print("Drive run is complete:", DRIVE_RUN)
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
