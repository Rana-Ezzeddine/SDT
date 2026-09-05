"""Train an equal-weight, full-parameter DPO model without opening the test set."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .pairs import validate_no_prompt_leakage


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping")
    return config


def _repo_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.resolve().parent.parent / path).resolve()


def _package_versions(names: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _precision(torch: Any) -> tuple[Any, bool, bool, str]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False, "cuda-bfloat16"
        return torch.float16, False, True, "cuda-float16"
    if torch.backends.mps.is_available():
        # Float32 is slower but is the safest default for an initial Mac run.
        return torch.float32, False, False, "mps-float32"
    return torch.float32, False, False, "cpu-float32"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if config.get("training_method") != "full_parameter_dpo":
        raise ValueError("This repository supports full-parameter DPO only")

    import torch
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer, __version__ as transformers_version, set_seed
    from trl import DPOConfig, DPOTrainer

    seed = int(config.get("seed", 42))
    set_seed(seed)

    pairs_path = _repo_path(config_path, str(config["pairs_file"]))
    output_dir = _repo_path(config_path, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = load_dataset("json", data_files=str(pairs_path), split="train")
    all_rows = [dict(row) for row in all_pairs]
    validate_no_prompt_leakage(all_rows)
    expected_label_mode = config.get("label_mode")
    observed_label_modes = {
        str(row.get("label_mode", "multi_judge")) for row in all_rows
    }
    if expected_label_mode and observed_label_modes != {str(expected_label_mode)}:
        raise ValueError(
            "Pair label mode does not match the training configuration: "
            f"expected={expected_label_mode!r}, observed={sorted(observed_label_modes)}"
        )

    filter_by_confidence = bool(config.get("filter_by_confidence", True))
    allowed_confidence = set(config.get("allowed_confidence", ["high", "medium"]))
    minimum_margin = float(config.get("min_label_margin", 0.0))

    def eligible(row: dict[str, Any]) -> bool:
        confidence_allowed = (
            not filter_by_confidence
            or row.get("label_confidence") in allowed_confidence
        )
        return bool(row["retain"]) and confidence_allowed and (
            float(row["label_margin"]) >= minimum_margin
        )

    eligible_pairs = all_pairs.filter(eligible)
    split_datasets: dict[str, Dataset] = {
        split: eligible_pairs.filter(lambda row, split=split: row["split"] == split)
        for split in ("train", "validation", "test")
    }
    eligible_counts_before_caps = {
        split: len(dataset) for split, dataset in split_datasets.items()
    }
    if len(split_datasets["train"]) == 0:
        raise ValueError("No eligible training pairs remain")
    if len(split_datasets["validation"]) == 0:
        raise ValueError(
            "No eligible validation pairs remain. For a smoke test, change the split seed; "
            "do not tune on the test partition."
        )
    if len(split_datasets["test"]) == 0:
        raise ValueError("No eligible test pairs remain; change the split seed before training")

    max_train = config.get("max_train_samples")
    max_validation = config.get("max_validation_samples")
    if max_train is not None:
        split_datasets["train"] = split_datasets["train"].shuffle(seed=seed).select(
            range(min(int(max_train), len(split_datasets["train"])))
        )
    if max_validation is not None:
        split_datasets["validation"] = split_datasets["validation"].shuffle(seed=seed).select(
            range(min(int(max_validation), len(split_datasets["validation"])))
        )

    sanity_use_train_as_validation = bool(
        config.get("sanity_use_train_as_validation", False)
    )
    if sanity_use_train_as_validation:
        # Deliberately evaluate on the same tiny subset only for the optional overfit
        # canary. This must never be enabled for a real experiment.
        split_datasets["validation"] = split_datasets["train"]

    model_id = str(config["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(config.get("max_length", 1024))

    def within_length(row: dict[str, Any]) -> bool:
        prompt = [{"role": "user", "content": row["prompt"]}]
        chosen = prompt + [{"role": "assistant", "content": row["chosen"]}]
        rejected = prompt + [{"role": "assistant", "content": row["rejected"]}]
        chosen_ids = tokenizer.apply_chat_template(
            chosen, tokenize=True, add_generation_prompt=False
        )
        rejected_ids = tokenizer.apply_chat_template(
            rejected, tokenize=True, add_generation_prompt=False
        )
        return max(len(chosen_ids), len(rejected_ids)) <= max_length

    before_lengths = {split: len(dataset) for split, dataset in split_datasets.items()}
    split_datasets = {
        split: dataset.filter(within_length)
        for split, dataset in split_datasets.items()
    }
    dropped_for_length = {
        split: before_lengths[split] - len(dataset)
        for split, dataset in split_datasets.items()
    }
    if len(split_datasets["train"]) == 0 or len(split_datasets["validation"]) == 0:
        raise ValueError("Length filtering removed all training or validation rows")

    def to_dpo(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "prompt": [{"role": "user", "content": row["prompt"]}],
            "chosen": [{"role": "assistant", "content": row["chosen"]}],
            "rejected": [{"role": "assistant", "content": row["rejected"]}],
        }

    train_dpo = split_datasets["train"].map(
        to_dpo, remove_columns=split_datasets["train"].column_names
    )
    validation_dpo = split_datasets["validation"].map(
        to_dpo, remove_columns=split_datasets["validation"].column_names
    )

    dtype, use_bf16, use_fp16, precision_name = _precision(torch)
    eval_strategy = str(config.get("eval_strategy", "epoch"))
    save_strategy = str(config.get("save_strategy", eval_strategy))
    dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
    warmup_steps_value = config.get("warmup_steps")
    warmup_steps = int(warmup_steps_value) if warmup_steps_value is not None else 0
    warmup_ratio = (
        0.0
        if warmup_steps_value is not None
        else float(config.get("warmup_ratio", 0.0))
    )
    training_args = DPOConfig(
        output_dir=str(output_dir),
        beta=float(config.get("beta", 0.1)),
        loss_type=str(config.get("loss_type", "sigmoid")),
        learning_rate=float(config.get("learning_rate", 5e-7)),
        warmup_steps=warmup_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=float(config.get("num_train_epochs", 1)),
        per_device_train_batch_size=int(config.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        max_length=max_length,
        eval_strategy=eval_strategy,
        eval_steps=int(config.get("eval_steps", 25)),
        save_strategy=save_strategy,
        save_steps=int(config.get("save_steps", 25)),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=int(config.get("logging_steps", 5)),
        report_to="none",
        bf16=use_bf16,
        fp16=use_fp16,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_cache=False,
        dataloader_pin_memory=False,
        precompute_ref_log_probs=bool(config.get("precompute_ref_log_probs", True)),
        model_init_kwargs={dtype_key: dtype},
        seed=seed,
        data_seed=seed,
    )

    trainer = DPOTrainer(
        model=model_id,
        ref_model=None,
        args=training_args,
        train_dataset=train_dpo,
        eval_dataset=validation_dpo,
        processing_class=tokenizer,
    )

    total_parameters = sum(parameter.numel() for parameter in trainer.model.parameters())
    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError(
            "Full DPO requires every model parameter to be trainable, but "
            f"only {trainable_parameters:,}/{total_parameters:,} are trainable"
        )
    print(f"Full DPO trainable parameters: {trainable_parameters:,}/{total_parameters:,}")

    train_result = trainer.train()
    # Reuse the initialized eval dataset because its reference log-probabilities were
    # cached before training. Passing validation_dpo here would be a new uncached dataset
    # after the reference model has already been released.
    validation_metrics = trainer.evaluate(metric_key_prefix="validation")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    (output_dir / "train_metrics.json").write_text(
        json.dumps(train_result.metrics, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "validation_metrics.json").write_text(
        json.dumps(validation_metrics, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_method": "full_parameter_dpo",
        "sanity_use_train_as_validation": sanity_use_train_as_validation,
        "model_id": model_id,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "precision": precision_name,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_versions": _package_versions(
            ["torch", "transformers", "datasets", "trl", "accelerate"]
        ),
        "pairs_file": str(pairs_path),
        "label_mode": expected_label_mode,
        "filter_by_confidence": filter_by_confidence,
        "allowed_confidence": sorted(allowed_confidence) if filter_by_confidence else None,
        "eligible_pairs_before_caps": eligible_counts_before_caps,
        "eligible_pairs": {split: len(dataset) for split, dataset in split_datasets.items()},
        "dropped_for_length": dropped_for_length,
        "test_evaluated": False,
        "test_policy": "Run the baseline and frozen full DPO checkpoint together with sdt-evaluate-pairs.",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
