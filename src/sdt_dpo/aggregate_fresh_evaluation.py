"""Aggregate a position-controlled, multi-generation, multi-judge evaluation."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .judge_generations import DIMENSIONS


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            prompt_id = str(row["prompt_id"])
            if prompt_id in rows:
                raise ValueError(f"Duplicate prompt_id {prompt_id} in {path}")
            rows[prompt_id] = row
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _bootstrap_interval(values: list[float], *, seed: int, samples: int) -> list[float]:
    if not values:
        raise ValueError("Cannot bootstrap an empty list")
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    low = means[math.floor(0.025 * (samples - 1))]
    high = means[math.ceil(0.975 * (samples - 1))]
    return [low, high]


def aggregate(manifest_path: Path, *, bootstrap_samples: int = 10000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("judgments", [])
    generation_seeds = [int(value) for value in manifest.get("generation_seeds", [])]
    judge_models = [str(value) for value in manifest.get("judge_models", [])]
    if not entries or not generation_seeds or not judge_models:
        raise ValueError("Manifest must define judgments, generation_seeds, and judge_models")

    votes: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    judge_stats: dict[str, Counter[str]] = defaultdict(Counter)
    prompt_text: dict[str, str] = {}
    expected_prompt_ids: set[str] | None = None

    for entry in entries:
        generation_seed = int(entry["generation_seed"])
        judge_model = str(entry["judge_model"])
        forward = _read_jsonl(Path(entry["forward"]))
        reverse = _read_jsonl(Path(entry["reverse"]))
        if set(forward) != set(reverse):
            raise ValueError(f"Forward/reverse IDs differ for {judge_model}, seed {generation_seed}")
        if expected_prompt_ids is None:
            expected_prompt_ids = set(forward)
        elif set(forward) != expected_prompt_ids:
            raise ValueError("Every judge and generation seed must cover the same prompts")

        for prompt_id in sorted(forward):
            first = forward[prompt_id]
            second = reverse[prompt_id]
            prompt = str(first["prompt"])
            if str(second["prompt"]) != prompt:
                raise ValueError(f"Prompt mismatch for {prompt_id}")
            prompt_text.setdefault(prompt_id, prompt)
            if prompt_text[prompt_id] != prompt:
                raise ValueError(f"Prompt changed across judgments for {prompt_id}")

            deterministic_tie = (
                first.get("judgment_type") == "deterministic_identical_response"
                and second.get("judgment_type") == "deterministic_identical_response"
            )
            consistent = first["winner"] == second["winner"]
            if deterministic_tie:
                judge_stats[judge_model]["deterministic_identical"] += 1
            else:
                judge_stats[judge_model]["position_checks"] += 1
                judge_stats[judge_model]["position_consistent"] += int(consistent)

            dimension_deltas: dict[str, float] | None = None
            if consistent and "dpo_scores" in first and "dpo_scores" in second:
                dimension_deltas = {
                    dimension: statistics.fmean(
                        [
                            float(first["dpo_scores"][dimension])
                            - float(first["baseline_scores"][dimension]),
                            float(second["dpo_scores"][dimension])
                            - float(second["baseline_scores"][dimension]),
                        ]
                    )
                    for dimension in DIMENSIONS
                }
            vote = first["winner"] if consistent else "inconsistent"
            judge_stats[judge_model][vote] += 1
            votes[(generation_seed, prompt_id)].append(
                {
                    "judge_model": judge_model,
                    "vote": vote,
                    "dimension_deltas": dimension_deltas,
                }
            )

    assert expected_prompt_ids is not None
    expected_entries = len(generation_seeds) * len(judge_models)
    if len(entries) != expected_entries:
        raise ValueError(
            f"Expected {expected_entries} judgment entries, found {len(entries)}"
        )

    majority = len(judge_models) // 2 + 1
    seed_results: dict[tuple[int, str], dict[str, Any]] = {}
    for key, panel_votes in votes.items():
        if len(panel_votes) != len(judge_models):
            raise ValueError(f"Incomplete judge panel for seed/prompt {key}")
        counts = Counter(item["vote"] for item in panel_votes)
        winner = (
            "dpo"
            if counts["dpo"] >= majority
            else "baseline"
            if counts["baseline"] >= majority
            else "tie"
        )
        seed_results[key] = {
            "winner": winner,
            "vote_counts": dict(counts),
            "judge_votes": panel_votes,
        }

    prompt_rows: list[dict[str, Any]] = []
    prompt_dimension_values: dict[str, list[float]] = {d: [] for d in DIMENSIONS}
    for prompt_id in sorted(expected_prompt_ids):
        outcomes = [seed_results[(seed, prompt_id)]["winner"] for seed in generation_seeds]
        counts = Counter(outcomes)
        generation_majority = len(generation_seeds) // 2 + 1
        winner = (
            "dpo"
            if counts["dpo"] >= generation_majority
            else "baseline"
            if counts["baseline"] >= generation_majority
            else "tie"
        )
        dimension_deltas: dict[str, float | None] = {}
        for dimension in DIMENSIONS:
            values = [
                float(vote["dimension_deltas"][dimension])
                for seed in generation_seeds
                for vote in seed_results[(seed, prompt_id)]["judge_votes"]
                if vote["dimension_deltas"] is not None
            ]
            dimension_deltas[dimension] = statistics.fmean(values) if values else None
            if values:
                prompt_dimension_values[dimension].append(statistics.fmean(values))
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt_text[prompt_id],
                "winner": winner,
                "generation_outcomes": outcomes,
                "generation_outcome_counts": dict(counts),
                "dimension_deltas": dimension_deltas,
            }
        )

    prompt_counts = Counter(row["winner"] for row in prompt_rows)
    scores = [1.0 if row["winner"] == "dpo" else 0.0 if row["winner"] == "baseline" else 0.5 for row in prompt_rows]
    dimension_summary = {
        dimension: {
            "mean_dpo_minus_baseline": statistics.fmean(values) if values else None,
            "prompt_bootstrap_95": _bootstrap_interval(
                values, seed=42 + index, samples=bootstrap_samples
            )
            if values
            else None,
        }
        for index, (dimension, values) in enumerate(prompt_dimension_values.items())
    }
    judge_summary = {}
    for judge_model, counts in judge_stats.items():
        checks = counts["position_checks"]
        judge_summary[judge_model] = {
            **dict(counts),
            "position_consistency_rate": counts["position_consistent"] / checks if checks else None,
        }

    report = {
        "evaluation_unit": "unique_prompt",
        "n_prompts": len(prompt_rows),
        "generation_seeds": generation_seeds,
        "judge_models": judge_models,
        "position_reversal": True,
        "prompt_level_dpo_wins": prompt_counts["dpo"],
        "prompt_level_baseline_wins": prompt_counts["baseline"],
        "prompt_level_ties": prompt_counts["tie"],
        "prompt_macro_tie_adjusted_dpo_score": statistics.fmean(scores),
        "prompt_bootstrap_95": _bootstrap_interval(scores, seed=42, samples=bootstrap_samples),
        "judge_diagnostics": judge_summary,
        "dimension_deltas": dimension_summary,
        "decision_rule": (
            "Strict judge majority per seed, then strict generation-seed majority per prompt; "
            "position-inconsistent judge votes cannot contribute to a win."
        ),
    }
    return report, prompt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    report, rows = aggregate(args.manifest, bootstrap_samples=args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
