"""Evaluate a baseline or full DPO checkpoint on a locked preference-pair split."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    """Return a Wilson 95% interval for a binomial proportion."""

    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _prompt_macro_accuracy(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["prompt_id"])].append(float(row["correct"]))
    return statistics.fmean(statistics.fmean(values) for values in groups.values())


def _prompt_bootstrap_interval(
    rows: list[dict[str, Any]], *, seed: int = 42, samples: int = 10_000
) -> list[float] | None:
    if not rows or samples <= 0:
        return None
    import random

    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["prompt_id"])].append(float(row["correct"]))
    prompt_scores = [statistics.fmean(values) for values in groups.values()]
    randomizer = random.Random(seed)
    estimates = [
        statistics.fmean(
            prompt_scores[randomizer.randrange(len(prompt_scores))]
            for _ in range(len(prompt_scores))
        )
        for _ in range(samples)
    ]
    estimates.sort()
    return [
        estimates[int(0.025 * (samples - 1))],
        estimates[int(0.975 * (samples - 1))],
    ]


def _summary(rows: list[dict[str, Any]], *, include_prompt_interval: bool = False) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "preference_accuracy": None, "accuracy_wilson_95": None}
    wins = sum(row["correct"] for row in rows)
    margins = [float(row["model_margin"]) for row in rows]
    result = {
        "n": len(rows),
        "unique_prompts": len({row["prompt_id"] for row in rows}),
        "correct": wins,
        "ties": sum(row["tie"] for row in rows),
        "preference_accuracy": wins / len(rows),
        "accuracy_wilson_95": wilson_interval(wins, len(rows)),
        "prompt_macro_accuracy": _prompt_macro_accuracy(rows),
        "mean_model_margin": statistics.fmean(margins),
        "median_model_margin": statistics.median(margins),
        "model_margin_std": statistics.pstdev(margins) if len(margins) > 1 else 0.0,
        "mean_chosen_logprob_per_token": statistics.fmean(
            float(row["chosen_logprob_per_token"]) for row in rows
        ),
        "mean_rejected_logprob_per_token": statistics.fmean(
            float(row["rejected_logprob_per_token"]) for row in rows
        ),
    }
    if include_prompt_interval:
        result["prompt_cluster_bootstrap_95"] = _prompt_bootstrap_interval(rows)
    return result


def summarize_details(
    evaluated: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    *,
    split: str,
    model: str,
    max_length: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "label_confidence": defaultdict(list),
        "comparison_type": defaultdict(list),
        "label_margin_bucket": defaultdict(list),
    }
    for row in evaluated:
        grouped["label_confidence"][str(row["label_confidence"])].append(row)
        grouped["comparison_type"][str(row["comparison_type"])].append(row)
        margin = float(row["label_margin"])
        bucket = "[0.10,0.25)" if margin < 0.25 else "[0.25,0.50)" if margin < 0.50 else "[0.50,+inf)"
        grouped["label_margin_bucket"][bucket].append(row)

    return {
        "model": model,
        "split": split,
        "scoring_rule": (
            "mean conditional log-probability per response token under the same chat template; "
            "a pair is correct when chosen > rejected"
        ),
        "max_length": max_length,
        "overall": _summary(evaluated, include_prompt_interval=True),
        "skipped": {"n": len(skipped), "reasons": _count(row["skip_reason"] for row in skipped)},
        "subgroups": {
            field: {value: _summary(rows) for value, rows in sorted(values.items())}
            for field, values in grouped.items()
        },
        "uncertainty_note": (
            "Pairs from the same prompt are correlated. Treat prompt-macro accuracy and its "
            "prompt-cluster bootstrap interval as the primary uncertainty summary; the Wilson "
            "pair interval is descriptive."
        ),
    }


def _count(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _device_and_dtype(torch: Any) -> tuple[Any, Any, str]:
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype, str(dtype).replace("torch.", "cuda-")
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32, "mps-float32"
    return torch.device("cpu"), torch.float32, "cpu-float32"


def _sequence(tokenizer: Any, prompt: str, response: str) -> tuple[list[int], int]:
    if not response:
        raise ValueError("empty_response")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("fast_tokenizer_required_for_response_offsets")

    # A generation-prompt rendering is not guaranteed to be a token prefix of the
    # completed chat rendering. Mark the exact start of the assistant content in the
    # rendered conversation, remove the marker, and use tokenizer character offsets
    # to find the first response token without relying on that invalid assumption.
    marker = "<|sdt_response_boundary|>"
    while marker in prompt or marker in response:
        marker += "_"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": marker + response},
    ]
    rendered_with_marker = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    marker_start = rendered_with_marker.find(marker)
    if marker_start < 0 or rendered_with_marker.count(marker) != 1:
        raise ValueError("response_boundary_marker_not_found")

    rendered = (
        rendered_with_marker[:marker_start]
        + rendered_with_marker[marker_start + len(marker) :]
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    response_start = next(
        (
            index
            for index, (start, end) in enumerate(offsets)
            if end > marker_start and end > start
        ),
        None,
    )
    if response_start is None or response_start >= len(full):
        raise ValueError("empty_tokenized_response")
    return full, response_start


def _score_response(
    model: Any,
    torch: Any,
    token_ids: list[int],
    response_start: int,
    device: Any,
) -> dict[str, float | int]:
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False).logits
        token_log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = ids[:, 1:]
        gathered = token_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        response_log_probs = gathered[:, response_start - 1 :]
    return {
        "mean": float(response_log_probs.mean().item()),
        "sum": float(response_log_probs.sum().item()),
        "token_count": int(response_log_probs.numel()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model ID or local full-checkpoint directory.",
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--confidence", nargs="+", default=["high", "medium"])
    parser.add_argument(
        "--all-confidence",
        action="store_true",
        help="Do not filter retained pairs by label_confidence (required for the one-judge pilot).",
    )
    parser.add_argument("--min-margin", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

    rows: list[dict[str, Any]] = []
    with args.pairs.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (
                row.get("retain")
                and row.get("split") == args.split
                and (
                    args.all_confidence
                    or row.get("label_confidence") in set(args.confidence)
                )
                and float(row.get("label_margin", 0.0)) >= args.min_margin
            ):
                rows.append(row)
    if not rows:
        raise ValueError("No eligible pairs matched the requested evaluation split and filters")

    device, dtype, precision = _device_and_dtype(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(args.model, **{dtype_key: dtype})
    model.to(device)
    model.eval()

    evaluated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        try:
            chosen_ids, chosen_start = _sequence(tokenizer, row["prompt"], row["chosen"])
            rejected_ids, rejected_start = _sequence(tokenizer, row["prompt"], row["rejected"])
            if max(len(chosen_ids), len(rejected_ids)) > args.max_length:
                raise ValueError("over_max_length")
            chosen_score = _score_response(model, torch, chosen_ids, chosen_start, device)
            rejected_score = _score_response(model, torch, rejected_ids, rejected_start, device)
            margin = float(chosen_score["mean"]) - float(rejected_score["mean"])
            evaluated.append(
                {
                    "pair_id": row["pair_id"],
                    "prompt_id": row["prompt_id"],
                    "split": row["split"],
                    "label_confidence": row["label_confidence"],
                    "label_margin": row["label_margin"],
                    "comparison_type": row["comparison_type"],
                    "chosen_logprob_per_token": chosen_score["mean"],
                    "rejected_logprob_per_token": rejected_score["mean"],
                    "chosen_logprob_sum": chosen_score["sum"],
                    "rejected_logprob_sum": rejected_score["sum"],
                    "chosen_token_count": chosen_score["token_count"],
                    "rejected_token_count": rejected_score["token_count"],
                    "model_margin": margin,
                    "correct": margin > 0.0,
                    "tie": math.isclose(margin, 0.0, abs_tol=1e-12),
                }
            )
        except ValueError as exc:
            skipped.append({"pair_id": row["pair_id"], "skip_reason": str(exc)})
        if index % 10 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)} pairs")

    summary = summarize_details(
        evaluated,
        skipped,
        split=args.split,
        model=args.model,
        max_length=args.max_length,
    )
    summary["precision"] = precision
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.details.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with args.details.open("w", encoding="utf-8") as handle:
        for row in evaluated:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
