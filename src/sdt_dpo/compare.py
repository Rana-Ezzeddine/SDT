"""Compare paired baseline and DPO fixed-pair evaluation outputs."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["pair_id"])] = row
    return rows


def _bootstrap_delta(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, seed: int, samples: int
) -> list[float] | None:
    if not pairs or samples <= 0:
        return None
    grouped: dict[str, list[float]] = {}
    for index, (before, after) in enumerate(pairs):
        prompt_id = str(before.get("prompt_id", f"pair-{index}"))
        grouped.setdefault(prompt_id, []).append(
            float(after["correct"]) - float(before["correct"])
        )
    prompt_deltas = [statistics.fmean(values) for values in grouped.values()]
    randomizer = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        deltas.append(
            statistics.fmean(
                prompt_deltas[randomizer.randrange(len(prompt_deltas))]
                for _ in range(len(prompt_deltas))
            )
        )
    deltas.sort()
    lower = deltas[int(0.025 * (samples - 1))]
    upper = deltas[int(0.975 * (samples - 1))]
    return [lower, upper]


def exact_mcnemar_p(base_only: int, dpo_only: int) -> float:
    """Exact two-sided McNemar p-value over discordant pairs."""

    discordant = base_only + dpo_only
    if discordant == 0:
        return 1.0
    smaller = min(base_only, dpo_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / 2**discordant
    return min(1.0, 2.0 * lower_tail)


def compare(
    base: dict[str, dict[str, Any]],
    dpo: dict[str, dict[str, Any]],
    *,
    seed: int = 42,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    common_ids = sorted(set(base) & set(dpo))
    if not common_ids:
        raise ValueError("Baseline and DPO files share no pair IDs")
    paired = [(base[pair_id], dpo[pair_id]) for pair_id in common_ids]
    base_accuracy = statistics.fmean(float(before["correct"]) for before, _ in paired)
    dpo_accuracy = statistics.fmean(float(after["correct"]) for _, after in paired)
    prompt_groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for index, pair in enumerate(paired):
        prompt_id = str(pair[0].get("prompt_id", f"pair-{index}"))
        prompt_groups.setdefault(prompt_id, []).append(pair)
    base_prompt_accuracy = statistics.fmean(
        statistics.fmean(float(before["correct"]) for before, _ in group)
        for group in prompt_groups.values()
    )
    dpo_prompt_accuracy = statistics.fmean(
        statistics.fmean(float(after["correct"]) for _, after in group)
        for group in prompt_groups.values()
    )
    base_only = sum(bool(before["correct"]) and not bool(after["correct"]) for before, after in paired)
    dpo_only = sum(not bool(before["correct"]) and bool(after["correct"]) for before, after in paired)
    return {
        "paired_n": len(paired),
        "unique_prompts": len(prompt_groups),
        "baseline_only_rows": len(set(base) - set(dpo)),
        "dpo_only_rows": len(set(dpo) - set(base)),
        "baseline_preference_accuracy": base_accuracy,
        "dpo_preference_accuracy": dpo_accuracy,
        "accuracy_delta_dpo_minus_baseline": dpo_accuracy - base_accuracy,
        "baseline_prompt_macro_accuracy": base_prompt_accuracy,
        "dpo_prompt_macro_accuracy": dpo_prompt_accuracy,
        "prompt_macro_accuracy_delta": dpo_prompt_accuracy - base_prompt_accuracy,
        "prompt_cluster_bootstrap_95": _bootstrap_delta(
            paired, seed=seed, samples=bootstrap_samples
        ),
        "mean_model_margin_baseline": statistics.fmean(
            float(before["model_margin"]) for before, _ in paired
        ),
        "mean_model_margin_dpo": statistics.fmean(
            float(after["model_margin"]) for _, after in paired
        ),
        "mean_model_margin_delta": statistics.fmean(
            float(after["model_margin"]) - float(before["model_margin"])
            for before, after in paired
        ),
        "discordant_pairs": {
            "baseline_correct_dpo_wrong": base_only,
            "baseline_wrong_dpo_correct": dpo_only,
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(base_only, dpo_only),
        },
        "interpretation": (
            "Positive deltas favor DPO. The primary interval resamples prompts, because multiple "
            "pairs from one prompt are correlated. McNemar is retained as an exploratory pair-level "
            "diagnostic. With this tiny sample, treat results as a pipeline check, not evidence of "
            "general improvement."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-details", type=Path, required=True)
    parser.add_argument("--dpo-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(
        _read_jsonl(args.baseline_details),
        _read_jsonl(args.dpo_details),
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
