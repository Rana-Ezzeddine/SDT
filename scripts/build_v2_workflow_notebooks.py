"""Build the restart-safe V2 training and fresh-response evaluation notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINING_OUTPUT = ROOT / "notebooks/SDT_1500_DPO_Pilot_V2_Training_Colab.ipynb"
EVALUATION_OUTPUT = ROOT / "notebooks/SDT_1500_DPO_Pilot_V2_Evaluation_Colab.ipynb"


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


def notebook(cells: list[dict[str, object]]) -> dict[str, object]:
    return {
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


training_cells = [
    markdown(
        """# SDT DPO pilot V2 — training, preservation, and generation

This notebook repeats the frozen 1,500-record Phi-4-mini full-DPO pilot. All
irreplaceable artifacts are written to a new `SDT_DPO_Pilot_V2` Drive folder.
The run is not considered trained until the Drive checkpoint is hashed, reloaded,
and used for a smoke generation. It then performs the locked fixed-pair test and
creates three matched baseline/DPO generation samples per test prompt.
"""
    ),
    markdown("## 1. GPU, exact branch, and installation\n"),
    code(
        """import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import subprocess, sys, json, hashlib, shutil, time
from datetime import datetime, timezone
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
GIT_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("Commit:", GIT_COMMIT)
"""
    ),
    markdown("## 2. Mount Drive and upload the enhanced 1,500-record JSON\n"),
    code(
        """from google.colab import drive, files
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/SDT_DPO_Pilot_V2")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
RESUME_RUN_NAME = ""  # After a runtime loss, paste the existing V2 run name here.
if RESUME_RUN_NAME:
    RUN_NAME = RESUME_RUN_NAME
    DRIVE_RUN = DRIVE_ROOT / RUN_NAME
    RAW_DATA = DRIVE_RUN / "reproducibility" / "sdt_1500_updated.json"
    assert RAW_DATA.exists(), f"Cannot resume; raw data is missing: {RAW_DATA}"
    raw_bytes = RAW_DATA.read_bytes()
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
else:
    uploaded = files.upload()
    assert len(uploaded) == 1, "Upload only sdt_1500_updated.json."
    raw_bytes = next(iter(uploaded.values()))
    raw_sha = hashlib.sha256(raw_bytes).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RUN_NAME = f"pilot-1500-v2-{raw_sha[:12]}-{GIT_COMMIT[:7]}-{timestamp}"
    DRIVE_RUN = DRIVE_ROOT / RUN_NAME
REPRO = DRIVE_RUN / "reproducibility"
MODEL_DIR = DRIVE_RUN / "training" / "model"
TEST_DIR = DRIVE_RUN / "test"
LOG_DIR = DRIVE_RUN / "logs"
for folder in (REPRO, MODEL_DIR.parent, TEST_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RAW_DATA = REPRO / "sdt_1500_updated.json"
if not RAW_DATA.exists():
    RAW_DATA.write_bytes(raw_bytes)
assert hashlib.sha256(RAW_DATA.read_bytes()).hexdigest() == raw_sha
print("V2 run:", DRIVE_RUN)
print("Raw data SHA-256:", raw_sha)
"""
    ),
    markdown("## 3. Build, audit, and freeze the preference pairs\n"),
    code(
        """PAIRS = REPRO / "dpo_pairs_1500.jsonl"
PAIR_REPORT = REPRO / "pair_report_1500.json"
subprocess.run([
    "sdt-build-pairs", "--input", str(RAW_DATA), "--output", str(PAIRS),
    "--report", str(PAIR_REPORT), "--label-mode", "single_judge_pilot",
    "--use-supplied-aggregates", "--min-common-judges", "1",
    "--min-margin", "0.10", "--train-share", "0.80",
    "--validation-share", "0.10", "--seed", "42",
], check=True)
pair_report = json.loads(PAIR_REPORT.read_text())
for key in ["records", "possible_pairs", "retained_pairs", "retained_prompts", "retained_by_split", "exclusion_reasons"]:
    print(f"{key}: {pair_report.get(key)}")
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=True)
"""
    ),
    markdown("## 4. Freeze configuration, environment, and run state\n"),
    code(
        """import yaml

BASE_CONFIG = yaml.safe_load(Path("configs/pilot_1500_phi.yaml").read_text())
BASELINE_MODEL = str(BASE_CONFIG["model_id"])
MAX_LENGTH = int(BASE_CONFIG["max_length"])
RESOLVED_CONFIG = dict(BASE_CONFIG)
RESOLVED_CONFIG.update({
    "pairs_file": str(PAIRS),
    "output_dir": str(MODEL_DIR),
    "learning_rate": 1e-6,
    "beta": 0.10,
    "num_train_epochs": 1,
    "resume_from_checkpoint": "auto",
    "save_total_limit": 1,
})
CONFIG_PATH = REPRO / "resolved-training-config.yaml"
CONFIG_PATH.write_text(yaml.safe_dump(RESOLVED_CONFIG, sort_keys=False))
(REPRO / "git-commit.txt").write_text(GIT_COMMIT + "\\n")
(REPRO / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
(REPRO / "gpu-info.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
shutil.copy2(Path("notebooks/SDT_1500_DPO_Pilot_V2_Training_Colab.ipynb"), REPRO / "training-notebook.ipynb")

STATE_FILE = DRIVE_RUN / "run-state.json"
def mark_stage(stage, **extra):
    previous = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {"stages": []}
    previous["run_name"] = RUN_NAME
    previous["git_commit"] = GIT_COMMIT
    previous["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    if stage not in previous["stages"]:
        previous["stages"].append(stage)
    previous.update(extra)
    STATE_FILE.write_text(json.dumps(previous, indent=2, default=str) + "\\n")
    os.sync()
    print("Recorded stage:", stage)

mark_stage("prepared", raw_sha256=raw_sha, pairs_sha256=hashlib.sha256(PAIRS.read_bytes()).hexdigest())
print(json.dumps(RESOLVED_CONFIG, indent=2))
"""
    ),
    markdown("## 5. Baseline validation\n"),
    code(
        """BASELINE_VAL = DRIVE_RUN / "validation" / "baseline-summary.json"
BASELINE_VAL_DETAILS = DRIVE_RUN / "validation" / "baseline-pairs.jsonl"
BASELINE_VAL.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([
    "sdt-evaluate-pairs", "--pairs", str(PAIRS), "--model", BASELINE_MODEL,
    "--split", "validation", "--max-length", str(MAX_LENGTH), "--all-confidence",
    "--output", str(BASELINE_VAL), "--details", str(BASELINE_VAL_DETAILS),
], check=True)
mark_stage("baseline_validation_complete")
print(json.loads(BASELINE_VAL.read_text())["overall"])
"""
    ),
    markdown(
        """## 6. Train the frozen DPO configuration

The output directory is on Drive. If a valid Trainer checkpoint already exists,
the command resumes it. The complete stdout/stderr stream is also saved on Drive.
Do not start a second copy of this cell concurrently.
"""
    ),
    code(
        """TRAIN_LOG = LOG_DIR / "dpo-training.log"

def run_logged(command, log_path):
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write(f"\\n[{datetime.now(timezone.utc).isoformat()}] COMMAND: {command}\\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

weight_files = sorted(MODEL_DIR.glob("*.safetensors"))
final_model_complete = all((MODEL_DIR / name).exists() for name in [
    "config.json", "run_manifest.json", "train_metrics.json", "validation_metrics.json",
]) and bool(weight_files)
if not final_model_complete:
    run_logged(["sdt-train-dpo", "--config", str(CONFIG_PATH)], TRAIN_LOG)
else:
    print("A final Drive model already exists; training was not repeated.")
mark_stage("training_process_complete")
"""
    ),
    markdown("## 7. Hash, reload, and smoke-test the Drive checkpoint\n"),
    code(
        """from transformers import AutoModelForCausalLM, AutoTokenizer

def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

weight_files = sorted(MODEL_DIR.glob("*.safetensors"))
assert (MODEL_DIR / "config.json").exists(), "Drive model config is missing."
assert weight_files, "Drive model weight files are missing."
total_weight_bytes = sum(path.stat().st_size for path in weight_files)
assert total_weight_bytes > 1_000_000_000, "Drive model weights are unexpectedly small."
weight_manifest = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in weight_files]
INTEGRITY = DRIVE_RUN / "model-integrity.json"
INTEGRITY.write_text(json.dumps({
    "model_dir": str(MODEL_DIR), "total_weight_bytes": total_weight_bytes,
    "weights": weight_manifest, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
}, indent=2) + "\\n")

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
reloaded = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True, torch_dtype=dtype).to("cuda")
inputs = tokenizer.apply_chat_template([{"role": "user", "content": "Reply with exactly: checkpoint loaded"}], add_generation_prompt=True, return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = reloaded.generate(inputs, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
smoke_text = tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=True).strip()
del reloaded
torch.cuda.empty_cache()
(DRIVE_RUN / "checkpoint-reload-smoke-test.json").write_text(json.dumps({"response": smoke_text}, indent=2) + "\\n")
mark_stage("drive_checkpoint_reloaded", model_weight_bytes=total_weight_bytes)
print("Reloaded Drive checkpoint successfully:", smoke_text)
"""
    ),
    markdown("## 8. Validate DPO on validation and verify parameter movement\n"),
    code(
        """DPO_VAL = DRIVE_RUN / "validation" / "dpo-summary.json"
DPO_VAL_DETAILS = DRIVE_RUN / "validation" / "dpo-pairs.jsonl"
subprocess.run([
    "sdt-evaluate-pairs", "--pairs", str(PAIRS), "--model", str(MODEL_DIR),
    "--split", "validation", "--max-length", str(MAX_LENGTH), "--all-confidence",
    "--output", str(DPO_VAL), "--details", str(DPO_VAL_DETAILS),
], check=True)
VALIDATION_COMPARISON = DRIVE_RUN / "validation" / "comparison.json"
subprocess.run([
    "sdt-compare-evaluations", "--baseline-details", str(BASELINE_VAL_DETAILS),
    "--dpo-details", str(DPO_VAL_DETAILS), "--output", str(VALIDATION_COMPARISON),
    "--beta", str(RESOLVED_CONFIG["beta"]),
], check=True)
CHECKPOINT_REPORT = DRIVE_RUN / "checkpoint-change.json"
subprocess.run([
    "sdt-verify-checkpoint", "--baseline-model", BASELINE_MODEL,
    "--trained-model", str(MODEL_DIR), "--output", str(CHECKPOINT_REPORT),
], check=True)
assert not json.loads(CHECKPOINT_REPORT.read_text())["exactly_equal"]
(DRIVE_RUN / "selected-configuration.json").write_text(json.dumps({
    "model": BASELINE_MODEL,
    "training_method": "full_parameter_dpo",
    "learning_rate": RESOLVED_CONFIG["learning_rate"],
    "beta": RESOLVED_CONFIG["beta"],
    "num_train_epochs": RESOLVED_CONFIG["num_train_epochs"],
    "validation_comparison": json.loads(VALIDATION_COMPARISON.read_text()),
}, indent=2) + "\\n")
mark_stage("validation_and_checkpoint_verification_complete")
print(json.dumps(json.loads(VALIDATION_COMPARISON.read_text())["primary_dpo_relative_metrics"], indent=2))
"""
    ),
    markdown("## 9. Open the locked fixed-pair test once\n"),
    code(
        """RUN_LOCKED_TEST = True
assert RUN_LOCKED_TEST, "Keep the test locked until the configuration is frozen."
for name, model in [("baseline", BASELINE_MODEL), ("dpo", str(MODEL_DIR))]:
    summary = TEST_DIR / f"{name}-summary.json"
    details = TEST_DIR / f"{name}-pairs.jsonl"
    if not (summary.exists() and details.exists()):
        subprocess.run([
            "sdt-evaluate-pairs", "--pairs", str(PAIRS), "--model", model,
            "--split", "test", "--max-length", str(MAX_LENGTH), "--all-confidence",
            "--output", str(summary), "--details", str(details),
        ], check=True)
FIXED_COMPARISON = TEST_DIR / "fixed-pair-comparison.json"
RELATIVE_PAIRS = TEST_DIR / "relative-pairs.jsonl"
subprocess.run([
    "sdt-compare-evaluations", "--baseline-details", str(TEST_DIR / "baseline-pairs.jsonl"),
    "--dpo-details", str(TEST_DIR / "dpo-pairs.jsonl"), "--output", str(FIXED_COMPARISON),
    "--details-output", str(RELATIVE_PAIRS), "--beta", str(RESOLVED_CONFIG["beta"]),
], check=True)
mark_stage("locked_fixed_pair_test_complete")
fixed = json.loads(FIXED_COMPARISON.read_text())
print(json.dumps(fixed["primary_dpo_relative_metrics"], indent=2))
print(json.dumps(fixed["secondary_absolute_likelihood_metrics"], indent=2))
"""
    ),
    markdown(
        """## 10. Generate three matched fresh samples per test prompt

Baseline and DPO use the same prompts, decoding settings, and per-prompt random
seeds. Files are written directly to Drive and each command resumes completed rows.
"""
    ),
    code(
        """GENERATION_SEEDS = [42, 202, 303]
GENERATION_ROOT = TEST_DIR / "fresh-generations"
GENERATION_ROOT.mkdir(parents=True, exist_ok=True)
for seed in GENERATION_SEEDS:
    seed_dir = GENERATION_ROOT / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    for name, model in [("baseline", BASELINE_MODEL), ("dpo", str(MODEL_DIR))]:
        output = seed_dir / f"{name}-generations.jsonl"
        subprocess.run([
            "sdt-generate-responses", "--pairs", str(PAIRS), "--model", model,
            "--output", str(output), "--split", "test",
            "--max-input-length", str(MAX_LENGTH), "--max-new-tokens", "512",
            "--temperature", "0.7", "--top-p", "0.9", "--seed", str(seed),
        ], check=True)
    baseline_rows = [json.loads(x) for x in (seed_dir / "baseline-generations.jsonl").read_text().splitlines() if x.strip()]
    dpo_rows = [json.loads(x) for x in (seed_dir / "dpo-generations.jsonl").read_text().splitlines() if x.strip()]
    assert [str(x["prompt_id"]) for x in baseline_rows] == [str(x["prompt_id"]) for x in dpo_rows]
    assert all(str(x["response"]).strip() for x in baseline_rows + dpo_rows)
    mark_stage(f"generation_seed_{seed}_complete", **{f"generation_seed_{seed}_prompts": len(baseline_rows)})
print("Completed matched generation seeds:", GENERATION_SEEDS)
"""
    ),
    markdown("## 11. Final V2 audit — do not skip\n"),
    code(
        """required = [
    RAW_DATA, PAIRS, PAIR_REPORT, CONFIG_PATH, TRAIN_LOG, INTEGRITY,
    MODEL_DIR / "config.json", DRIVE_RUN / "checkpoint-reload-smoke-test.json",
    MODEL_DIR / "run_manifest.json", MODEL_DIR / "train_metrics.json",
    MODEL_DIR / "validation_metrics.json", DRIVE_RUN / "selected-configuration.json",
    CHECKPOINT_REPORT, VALIDATION_COMPARISON, FIXED_COMPARISON, RELATIVE_PAIRS,
]
required += weight_files
for seed in GENERATION_SEEDS:
    required += [
        GENERATION_ROOT / f"seed-{seed}" / "baseline-generations.jsonl",
        GENERATION_ROOT / f"seed-{seed}" / "dpo-generations.jsonl",
    ]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
assert not missing, "Missing V2 artifacts:\\n" + "\\n".join(missing)
FINAL_MANIFEST = DRIVE_RUN / "artifact-manifest.json"
FINAL_MANIFEST.write_text(json.dumps({
    "run_name": RUN_NAME, "git_commit": GIT_COMMIT,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "artifacts": [{"path": str(p.relative_to(DRIVE_RUN)), "bytes": p.stat().st_size} for p in required],
}, indent=2) + "\\n")
mark_stage("training_testing_generation_complete", final_manifest=str(FINAL_MANIFEST))
os.sync()
print("V2 training and generation are complete:", DRIVE_RUN)
print("Copy this RUN_NAME into the evaluation notebook:", RUN_NAME)
"""
    ),
]


evaluation_cells = [
    markdown(
        """# SDT DPO pilot V2 — multi-sample fresh-response evaluation

This notebook never trains. It evaluates the three saved matched generation
samples from the V2 run with three stronger local judges. Every pair is judged
blindly in both A/B orders. Failed and order-inconsistent votes cannot create a
win. Final aggregation is first across judges for each generation seed, then
across generation seeds for each unique prompt.
"""
    ),
    markdown("## 1. GPU, exact branch, and installation\n"),
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
GIT_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print("Commit:", GIT_COMMIT)
"""
    ),
    markdown("## 2. Select and verify the completed V2 run\n"),
    code(
        """from google.colab import drive
drive.mount("/content/drive")

DRIVE_ROOT = Path("/content/drive/MyDrive/SDT_DPO_Pilot_V2")
RUN_NAME = ""  # Paste the exact RUN_NAME printed by the training notebook; blank selects latest complete run.
if RUN_NAME:
    DRIVE_RUN = DRIVE_ROOT / RUN_NAME
else:
    candidates = sorted(
        [p for p in DRIVE_ROOT.glob("pilot-1500-v2-*") if (p / "artifact-manifest.json").exists()],
        key=lambda p: p.stat().st_mtime,
    )
    assert candidates, "No completed V2 run found."
    DRIVE_RUN = candidates[-1]
    RUN_NAME = DRIVE_RUN.name
assert (DRIVE_RUN / "artifact-manifest.json").exists(), "Training/generation final audit is missing."
MODEL_DIR = DRIVE_RUN / "training" / "model"
INTEGRITY = json.loads((DRIVE_RUN / "model-integrity.json").read_text())
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

for item in INTEGRITY["weights"]:
    path = MODEL_DIR / item["name"]
    assert path.exists() and path.stat().st_size == item["bytes"], f"Missing/corrupt weight: {path}"
    assert sha256_file(path) == item["sha256"], f"Weight hash mismatch: {path}"
print("Verified V2 run:", DRIVE_RUN)
print("Saved model bytes:", INTEGRITY["total_weight_bytes"])
"""
    ),
    markdown("## 3. Verify all matched generation samples\n"),
    code(
        """GENERATION_SEEDS = [42, 202, 303]
GENERATION_ROOT = DRIVE_RUN / "test" / "fresh-generations"
generation_files = {}
for seed in GENERATION_SEEDS:
    baseline = GENERATION_ROOT / f"seed-{seed}" / "baseline-generations.jsonl"
    dpo = GENERATION_ROOT / f"seed-{seed}" / "dpo-generations.jsonl"
    def read_rows(path):
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        assert rows and len({str(x["prompt_id"]) for x in rows}) == len(rows)
        assert all(str(x.get("response", "")).strip() for x in rows)
        return rows
    baseline_rows, dpo_rows = read_rows(baseline), read_rows(dpo)
    assert [(str(x["prompt_id"]), x["prompt"]) for x in baseline_rows] == [(str(x["prompt_id"]), x["prompt"]) for x in dpo_rows]
    generation_files[seed] = {"baseline": baseline, "dpo": dpo}
    identical = sum(" ".join(a["response"].split()) == " ".join(b["response"].split()) for a, b in zip(baseline_rows, dpo_rows))
    print({"seed": seed, "prompts": len(baseline_rows), "identical": identical})
"""
    ),
    markdown(
        """## 4. Blind stronger-judge panel with position reversal

The models run sequentially, so only one judge occupies GPU memory at a time.
Outputs are written directly to Drive and resume row-by-row. On an A100 40/80 GB,
the default 12–14B judges should fit in bfloat16. If a model is unavailable, use a
comparably capable ungated instruct model and start a new evaluation version.
"""
    ),
    code(
        """JUDGE_MODELS = [
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "allenai/OLMo-2-1124-13B-Instruct",
]
EVAL_ROOT = DRIVE_RUN / "test" / "fresh-evaluation-v2"
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

def judge_slug(index, model):
    short = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")[-56:]
    return f"judge-{index + 1}-{short}"

COMBINED_ROOT = EVAL_ROOT / "combined-inputs"
COMBINED_ROOT.mkdir(parents=True, exist_ok=True)
combined_files = {}
for model_name in ("baseline", "dpo"):
    combined = COMBINED_ROOT / f"{model_name}-generations.jsonl"
    with combined.open("w", encoding="utf-8") as handle:
        for seed in GENERATION_SEEDS:
            for line in generation_files[seed][model_name].read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                original_prompt_id = str(row["prompt_id"])
                row["original_prompt_id"] = original_prompt_id
                row["generation_seed"] = seed
                row["prompt_id"] = f"{seed}::{original_prompt_id}"
                handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
    combined_files[model_name] = combined

# Combining the three generation seeds lets each 12–14B judge load only twice
# (forward and reverse), rather than six times. Split files below restore the
# original prompt IDs for seed-aware aggregation.
manifest_entries = []
for judge_index, judge_model in enumerate(JUDGE_MODELS):
    judge_root = EVAL_ROOT / "judgments" / judge_slug(judge_index, judge_model)
    judge_root.mkdir(parents=True, exist_ok=True)
    combined_details = {}
    for orientation, reverse in [("forward", False), ("reverse", True)]:
        details = judge_root / f"combined-{orientation}.jsonl"
        command = [
            "sdt-judge-generations-local",
            "--baseline", str(combined_files["baseline"]),
            "--dpo", str(combined_files["dpo"]),
            "--details", str(details),
            "--summary", str(judge_root / f"combined-{orientation}-summary.json"),
            "--failures", str(judge_root / f"combined-{orientation}-failures.jsonl"),
            "--judge-model", judge_model, "--seed", "42",
            "--max-new-tokens", "512", "--max-retries", "3", "--continue-on-failure",
        ]
        if reverse:
            command.append("--reverse-order")
        print("Running", judge_model, orientation, "across all generation seeds")
        subprocess.run(command, check=True)
        combined_details[orientation] = details

    for seed in GENERATION_SEEDS:
        seed_dir = judge_root / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        prefix = f"{seed}::"
        for orientation in ("forward", "reverse"):
            split_path = seed_dir / f"{orientation}.jsonl"
            with split_path.open("w", encoding="utf-8") as handle:
                for line in combined_details[orientation].read_text().splitlines():
                    row = json.loads(line)
                    combined_id = str(row["prompt_id"])
                    if combined_id.startswith(prefix):
                        row["prompt_id"] = combined_id[len(prefix):]
                        handle.write(json.dumps(row, ensure_ascii=False) + "\\n")
            assert split_path.exists() and split_path.stat().st_size > 0
            paths[orientation] = str(split_path)
        manifest_entries.append({"generation_seed": seed, "judge_model": judge_model, **paths})

MANIFEST = EVAL_ROOT / "evaluation-manifest.json"
MANIFEST.write_text(json.dumps({
    "run_name": RUN_NAME, "generation_seeds": GENERATION_SEEDS,
    "judge_models": JUDGE_MODELS, "position_reversal": True,
    "failed_votes_treated_as_unavailable": True, "judgments": manifest_entries,
}, indent=2) + "\\n")
print("Saved:", MANIFEST)
"""
    ),
    markdown("## 5. Aggregate at the unique-prompt level\n"),
    code(
        """FINAL_SUMMARY = EVAL_ROOT / "fresh-evaluation-summary.json"
PROMPT_DETAILS = EVAL_ROOT / "fresh-evaluation-prompt-details.jsonl"
subprocess.run([
    "sdt-aggregate-fresh-evaluation", "--manifest", str(MANIFEST),
    "--output", str(FINAL_SUMMARY), "--details", str(PROMPT_DETAILS),
    "--bootstrap-samples", "10000",
], check=True)
report = json.loads(FINAL_SUMMARY.read_text())
print(json.dumps({k: report[k] for k in [
    "n_prompts", "prompt_level_dpo_wins", "prompt_level_baseline_wins",
    "prompt_level_ties", "prompt_macro_tie_adjusted_dpo_score", "prompt_bootstrap_95",
]}, indent=2))
"""
    ),
    markdown("## 6. Results and diagnostics\n"),
    code(
        """import pandas as pd
import matplotlib.pyplot as plt

display(pd.DataFrame([{"judge": judge, **metrics} for judge, metrics in report["judge_diagnostics"].items()]))
display(pd.DataFrame([{
    "prompts": report["n_prompts"], "DPO wins": report["prompt_level_dpo_wins"],
    "Baseline wins": report["prompt_level_baseline_wins"], "Ties": report["prompt_level_ties"],
    "Tie-adjusted DPO score": report["prompt_macro_tie_adjusted_dpo_score"],
    "95% low": report["prompt_bootstrap_95"][0], "95% high": report["prompt_bootstrap_95"][1],
}]))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(["DPO", "Baseline", "Tie"], [report["prompt_level_dpo_wins"], report["prompt_level_baseline_wins"], report["prompt_level_ties"]], color=["#2E86AB", "#D1495B", "#999999"])
axes[0].set_title("Prompt-level fresh-response outcomes")
names = list(report["dimension_deltas"])
values = [report["dimension_deltas"][name]["mean_dpo_minus_baseline"] for name in names]
axes[1].barh(names, values, color=["#2E86AB" if value >= 0 else "#D1495B" for value in values])
axes[1].axvline(0, color="black", linewidth=1)
axes[1].set_title("Mean judge-score delta: DPO − baseline")
plt.tight_layout()
FIGURE = EVAL_ROOT / "fresh-evaluation-overview.png"
plt.savefig(FIGURE, dpi=180, bbox_inches="tight")
plt.show()
"""
    ),
    markdown("## 7. Final evaluation audit\n"),
    code(
        """required = [MANIFEST, FINAL_SUMMARY, PROMPT_DETAILS, FIGURE]
required += [Path(entry[key]) for entry in manifest_entries for key in ("forward", "reverse")]
missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
assert not missing, "Missing evaluation artifacts:\\n" + "\\n".join(missing)
AUDIT = EVAL_ROOT / "evaluation-artifact-manifest.json"
AUDIT.write_text(json.dumps({
    "run_name": RUN_NAME, "git_commit": GIT_COMMIT,
    "completed_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    "artifacts": [{"path": str(p.relative_to(DRIVE_RUN)), "bytes": p.stat().st_size} for p in required],
}, indent=2) + "\\n")
os.sync()
print("V2 evaluation complete:", EVAL_ROOT)
print("Final result:", FINAL_SUMMARY)
"""
    ),
]


TRAINING_OUTPUT.write_text(json.dumps(notebook(training_cells), indent=1) + "\n", encoding="utf-8")
EVALUATION_OUTPUT.write_text(json.dumps(notebook(evaluation_cells), indent=1) + "\n", encoding="utf-8")
print(TRAINING_OUTPUT)
print(EVALUATION_OUTPUT)
