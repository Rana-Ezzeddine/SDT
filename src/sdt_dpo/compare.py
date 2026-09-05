"""Compare paired baseline and DPO fixed-pair evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate import wilson_interval


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pair_id = str(row["pair_id"])
            if pair_id in rows:
                raise ValueError(f"Duplicate pair_id {pair_id!r} in {path}")
            rows[pair_id] = row
    return rows


def _percentile(sorted_values: list[float], quantile: float) -> float:
    return sorted_values[int(quantile * (len(sorted_values) - 1))]


def _prompt_macro(rows: list[dict[str, Any]], field: str) -> float:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["prompt_id"])].append(float(row[field]))
    return statistics.fmean(statistics.fmean(values) for values in groups.values())


def _prompt_cluster_interval(
    rows: list[dict[str, Any]],
    field: str,
    *,
    seed: int,
    samples: int,
) -> list[float] | None:
    if not rows or samples <= 0:
        return None
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["prompt_id"])].append(float(row[field]))
    prompt_values = [statistics.fmean(values) for values in groups.values()]
    randomizer = random.Random(seed)
    estimates = sorted(
        statistics.fmean(
            prompt_values[randomizer.randrange(len(prompt_values))]
            for _ in range(len(prompt_values))
        )
        for _ in range(samples)
    )
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def exact_mcnemar_p(base_only: int, dpo_only: int) -> float:
    """Exact two-sided McNemar p-value over discordant pairs."""

    discordant = base_only + dpo_only
    if discordant == 0:
        return 1.0
    smaller = min(base_only, dpo_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * lower_tail)


def analyze(
    base: dict[str, dict[str, Any]],
    dpo: dict[str, dict[str, Any]],
    *,
    beta: float = 0.1,
    seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not base or not dpo:
        raise ValueError(
            f"Evaluation details must be nonempty (baseline={len(base)}, dpo={len(dpo)})"
        )
    if set(base) != set(dpo):
        raise ValueError(
            "Baseline and DPO evaluations must contain exactly the same pair IDs "
            f"(baseline={len(base)}, dpo={len(dpo)}, shared={len(set(base) & set(dpo))})"
        )
    if beta <= 0:
        raise ValueError("beta must be positive")

    required = {
        "prompt_id",
        "correct",
        "model_margin",
        "chosen_logprob_sum",
        "rejected_logprob_sum",
        "chosen_token_count",
        "rejected_token_count",
    }
    details: list[dict[str, Any]] = []
    for pair_id in sorted(base):
        before = base[pair_id]
        after = dpo[pair_id]
        missing = (required - set(before)) | (required - set(after))
        if missing:
            raise ValueError(
                "Evaluation files lack fields required for DPO-relative metrics: "
                f"{sorted(missing)}. Rerun both evaluations with the updated evaluator."
            )
        if (
            int(before["chosen_token_count"]) != int(after["chosen_token_count"])
            or int(before["rejected_token_count"]) != int(after["rejected_token_count"])
        ):
            raise ValueError(f"Tokenization differs between models for pair {pair_id}")

        chosen_log_ratio = float(after["chosen_logprob_sum"]) - float(
            before["chosen_logprob_sum"]
        )
        rejected_log_ratio = float(after["rejected_logprob_sum"]) - float(
            before["rejected_logprob_sum"]
        )
        implicit_margin = beta * (chosen_log_ratio - rejected_log_ratio)
        absolute_delta = float(after["correct"]) - float(before["correct"])
        details.append(
            {
                "pair_id": pair_id,
                "prompt_id": str(before["prompt_id"]),
                "baseline_correct": bool(before["correct"]),
                "dpo_correct": bool(after["correct"]),
                "absolute_accuracy_delta": absolute_delta,
                "baseline_model_margin": float(before["model_margin"]),
                "dpo_model_margin": float(after["model_margin"]),
                "absolute_margin_delta": float(after["model_margin"])
                - float(before["model_margin"]),
                "chosen_policy_logratio": chosen_log_ratio,
                "rejected_policy_logratio": rejected_log_ratio,
                "dpo_implicit_reward_margin": implicit_margin,
                "dpo_implicit_reward_correct": implicit_margin > 0.0,
                "dpo_implicit_reward_tie": math.isclose(
                    implicit_margin, 0.0, abs_tol=1e-12
                ),
                "checkpoint_behavior_changed": (
                    not math.isclose(chosen_log_ratio, 0.0, abs_tol=1e-8)
                    or not math.isclose(rejected_log_ratio, 0.0, abs_tol=1e-8)
                ),
            }
        )

    pair_count = len(details)
    base_accuracy = statistics.fmean(float(row["baseline_correct"]) for row in details)
    dpo_accuracy = statistics.fmean(float(row["dpo_correct"]) for row in details)
    implicit_correct = sum(bool(row["dpo_implicit_reward_correct"]) for row in details)
    base_only = sum(
        bool(row["baseline_correct"]) and not bool(row["dpo_correct"]) for row in details
    )
    dpo_only = sum(
        not bool(row["baseline_correct"]) and bool(row["dpo_correct"]) for row in details
    )
    implicit_margins = [float(row["dpo_implicit_reward_margin"]) for row in details]
    changed_pairs = sum(bool(row["checkpoint_behavior_changed"]) for row in details)

    report = {
        "paired_n": pair_count,
        "unique_prompts": len({row["prompt_id"] for row in details}),
        "beta": beta,
        "primary_dpo_relative_metrics": {
            "implicit_reward_accuracy": implicit_correct / pair_count,
            "implicit_reward_accuracy_wilson_95": wilson_interval(
                implicit_correct, pair_count
            ),
            "implicit_reward_prompt_macro_accuracy": _prompt_macro(
                details, "dpo_implicit_reward_correct"
            ),
            "implicit_reward_prompt_cluster_accuracy_95": _prompt_cluster_interval(
                details,
                "dpo_implicit_reward_correct",
                seed=seed,
                samples=bootstrap_samples,
            ),
            "mean_implicit_reward_margin": statistics.fmean(implicit_margins),
            "median_implicit_reward_margin": statistics.median(implicit_margins),
            "implicit_reward_margin_prompt_cluster_95": _prompt_cluster_interval(
                details,
                "dpo_implicit_reward_margin",
                seed=seed + 1,
                samples=bootstrap_samples,
            ),
            "ties": sum(bool(row["dpo_implicit_reward_tie"]) for row in details),
        },
        "secondary_absolute_likelihood_metrics": {
            "baseline_preference_accuracy": base_accuracy,
            "dpo_preference_accuracy": dpo_accuracy,
            "accuracy_delta_dpo_minus_baseline": dpo_accuracy - base_accuracy,
            "baseline_prompt_macro_accuracy": _prompt_macro(
                details, "baseline_correct"
            ),
            "dpo_prompt_macro_accuracy": _prompt_macro(details, "dpo_correct"),
            "prompt_macro_accuracy_delta": _prompt_macro(
                details, "absolute_accuracy_delta"
            ),
            "prompt_cluster_accuracy_delta_95": _prompt_cluster_interval(
                details,
                "absolute_accuracy_delta",
                seed=seed + 2,
                samples=bootstrap_samples,
            ),
            "mean_model_margin_baseline": statistics.fmean(
                float(row["baseline_model_margin"]) for row in details
            ),
            "mean_model_margin_dpo": statistics.fmean(
                float(row["dpo_model_margin"]) for row in details
            ),
            "mean_model_margin_delta": statistics.fmean(
                float(row["absolute_margin_delta"]) for row in details
            ),
            "discordant_pairs": {
                "baseline_correct_dpo_wrong": base_only,
                "baseline_wrong_dpo_correct": dpo_only,
                "exact_mcnemar_two_sided_p": exact_mcnemar_p(base_only, dpo_only),
            },
        },
        "checkpoint_behavior_check": {
            "pairs_with_changed_completion_logprobs": changed_pairs,
            "all_pairs_changed": changed_pairs == pair_count,
            "mean_abs_chosen_policy_logratio": statistics.fmean(
                abs(float(row["chosen_policy_logratio"])) for row in details
            ),
            "mean_abs_rejected_policy_logratio": statistics.fmean(
                abs(float(row["rejected_policy_logratio"])) for row in details
            ),
        },
        "interpretation": (
            "Use the DPO-relative implicit reward accuracy and margin as the primary check of "
            "the DPO objective. Absolute length-normalized likelihood ranking is retained as a "
            "secondary diagnostic. Prompt-cluster intervals are primary because pairs from one "
            "prompt are correlated. A pilot dataset validates the pipeline and provides preliminary "
            "evidence; final alignment claims require the frozen multi-judge experiment."
        ),
    }
    return report, details


def compare(
    base: dict[str, dict[str, Any]],
    dpo: dict[str, dict[str, Any]],
    *,
    beta: float = 0.1,
    seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    report, _ = analyze(
        base,
        dpo,
        beta=beta,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument("--dpo-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, details = analyze(
        _read_jsonl(args.baseline_details),
        _read_jsonl(args.dpo_details),
        beta=args.beta,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.details_output:
        args.details_output.parent.mkdir(parents=True, exist_ok=True)
        with args.details_output.open("w", encoding="utf-8") as handle:
            for row in details:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
