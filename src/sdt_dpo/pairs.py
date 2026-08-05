"""Convert the raw SDT evaluation JSON into auditable DPO pairs."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "validation", "test")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def normalize_prompt(text: str) -> str:
    """Normalize exact prompts so formatting-only duplicates share one group."""

    return " ".join(str(text).strip().split()).casefold()


def prompt_group_id(prompt: str) -> str:
    normalized = normalize_prompt(prompt)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def assign_split(
    prompt: str,
    *,
    seed: int = 42,
    train_share: float = 0.80,
    validation_share: float = 0.10,
) -> str:
    """Assign a complete normalized-prompt group to one deterministic split."""

    if not 0 < train_share < 1:
        raise ValueError("train_share must be between 0 and 1")
    if not 0 <= validation_share < 1 or train_share + validation_share >= 1:
        raise ValueError("train_share + validation_share must be below 1")

    key = f"{seed}::{normalize_prompt(prompt)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
    if value < train_share:
        return "train"
    if value < train_share + validation_share:
        return "validation"
    return "test"


def valid_judge_scores(evaluation: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Return valid per-judge means and repeat variability.

    Failed judgments are omitted. They are never converted to numeric zero.
    """

    result: dict[str, dict[str, float]] = {}
    for judgment in evaluation.get("judgments", []) or []:
        if str(judgment.get("status", "")).lower() != "success":
            continue
        judge = str(judgment.get("judge_model", "")).strip()
        if not judge:
            continue

        attempts = [
            score
            for item in (judgment.get("scores", []) or [])
            if (score := _number(item.get("judge_score"))) is not None
        ]
        reported_avg = _number(judgment.get("score_avg"))
        if attempts:
            mean_score = statistics.fmean(attempts)
            repeat_std = statistics.pstdev(attempts) if len(attempts) > 1 else 0.0
        elif reported_avg is not None:
            mean_score = reported_avg
            repeat_std = _number(judgment.get("score_std")) or 0.0
        else:
            continue

        result[judge] = {
            "mean": mean_score,
            "repeat_std": repeat_std,
            "valid_attempts": float(len(attempts)),
        }
    return result


def _confidence_label(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def construct_pair(
    record: dict[str, Any],
    source_a: str,
    source_b: str,
    *,
    split: str,
    min_common_judges: int,
    min_margin: float,
    min_confidence: float,
    score_span: float,
) -> dict[str, Any]:
    prompt = str(record.get("instruction", "")).strip()
    responses = record.get("responses", {}) or {}
    evaluations = record.get("evaluations", {}) or {}
    text_a = str(responses.get(source_a, "")).strip()
    text_b = str(responses.get(source_b, "")).strip()

    judges_a = valid_judge_scores(evaluations.get(source_a, {}) or {})
    judges_b = valid_judge_scores(evaluations.get(source_b, {}) or {})
    common_judges = sorted(set(judges_a) & set(judges_b))
    expected_judges = set()
    for evaluation in evaluations.values():
        for judgment in (evaluation or {}).get("judgments", []) or []:
            judge = str(judgment.get("judge_model", "")).strip()
            if judge:
                expected_judges.add(judge)

    deltas = [judges_a[j]["mean"] - judges_b[j]["mean"] for j in common_judges]
    mean_delta = statistics.fmean(deltas) if deltas else 0.0
    direction = 1 if mean_delta > 0 else -1 if mean_delta < 0 else 0

    if direction > 0:
        chosen_source, rejected_source = source_a, source_b
        chosen_text, rejected_text = text_a, text_b
    elif direction < 0:
        chosen_source, rejected_source = source_b, source_a
        chosen_text, rejected_text = text_b, text_a
    else:
        chosen_source, rejected_source = source_a, source_b
        chosen_text, rejected_text = text_a, text_b

    agreement = (
        sum(
            1
            for delta in deltas
            if not math.isclose(delta, 0.0, abs_tol=1e-12)
            and (delta > 0) == (direction > 0)
        )
        / len(deltas)
        if deltas and direction
        else 0.0
    )
    coverage = len(common_judges) / len(expected_judges) if expected_judges else 0.0
    repeat_stds = [
        value
        for judge in common_judges
        for value in (judges_a[judge]["repeat_std"], judges_b[judge]["repeat_std"])
    ]
    average_repeat_std = statistics.fmean(repeat_stds) if repeat_stds else 0.0
    consistency = max(0.0, 1.0 - average_repeat_std / score_span)
    confidence_score = agreement * coverage * consistency
    confidence_label = _confidence_label(confidence_score)
    label_margin = abs(mean_delta)

    exclusion_reasons: list[str] = []
    if not prompt:
        exclusion_reasons.append("empty_prompt")
    if not text_a or not text_b:
        exclusion_reasons.append("empty_response")
    if text_a == text_b:
        exclusion_reasons.append("identical_responses")
    if len(common_judges) < min_common_judges:
        exclusion_reasons.append("insufficient_common_successful_judges")
    if not direction:
        exclusion_reasons.append("aggregate_tie")
    if label_margin < min_margin:
        exclusion_reasons.append("margin_below_threshold")
    if confidence_score < min_confidence:
        exclusion_reasons.append("confidence_below_threshold")

    preferred = str(record.get("preferred_response", "")).strip()
    preferred_field_agrees: bool | None = None
    if preferred in {source_a, source_b} and direction:
        preferred_field_agrees = preferred == chosen_source

    pair_key = "::".join(
        [str(record.get("id", "")), prompt_group_id(prompt), source_a, source_b]
    )
    pair_id = hashlib.sha256(pair_key.encode("utf-8")).hexdigest()[:20]

    return {
        "pair_id": pair_id,
        "record_id": str(record.get("id", "")),
        "prompt_id": prompt_group_id(prompt),
        "split": split,
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "chosen_source": chosen_source,
        "rejected_source": rejected_source,
        "comparison_type": "__vs__".join(sorted((source_a, source_b))),
        "chosen_score_common_judges": (
            statistics.fmean(
                [
                    (judges_a if chosen_source == source_a else judges_b)[j]["mean"]
                    for j in common_judges
                ]
            )
            if common_judges
            else None
        ),
        "rejected_score_common_judges": (
            statistics.fmean(
                [
                    (judges_b if rejected_source == source_b else judges_a)[j]["mean"]
                    for j in common_judges
                ]
            )
            if common_judges
            else None
        ),
        "label_margin": label_margin,
        "label_confidence": confidence_label,
        "confidence_score": confidence_score,
        "judge_agreement": agreement,
        "judge_coverage": coverage,
        "repeat_consistency": consistency,
        "average_repeat_std": average_repeat_std,
        "common_successful_judges": len(common_judges),
        "expected_judges": len(expected_judges),
        "raw_preferred_response": preferred,
        "preferred_field_agrees": preferred_field_agrees,
        "pair_weight": 1.0,
        "retain": not exclusion_reasons,
        "exclusion_reasons": exclusion_reasons,
    }


def build_pairs(
    records: Iterable[dict[str, Any]],
    *,
    seed: int = 42,
    train_share: float = 0.80,
    validation_share: float = 0.10,
    min_common_judges: int = 2,
    min_margin: float = 0.10,
    min_confidence: float = 0.60,
    score_span: float = 4.0,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for record in records:
        prompt = str(record.get("instruction", ""))
        split = assign_split(
            prompt,
            seed=seed,
            train_share=train_share,
            validation_share=validation_share,
        )
        sources = sorted((record.get("responses", {}) or {}).keys())
        for source_a, source_b in itertools.combinations(sources, 2):
            pairs.append(
                construct_pair(
                    record,
                    source_a,
                    source_b,
                    split=split,
                    min_common_judges=min_common_judges,
                    min_margin=min_margin,
                    min_confidence=min_confidence,
                    score_span=score_span,
                )
            )
    return pairs


def validate_no_prompt_leakage(pairs: Iterable[dict[str, Any]]) -> None:
    prompt_splits: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        prompt_splits[str(pair["prompt_id"])].add(str(pair["split"]))
    leaked = {prompt_id: splits for prompt_id, splits in prompt_splits.items() if len(splits) > 1}
    if leaked:
        preview = dict(list(leaked.items())[:5])
        raise ValueError(f"Prompt leakage detected: {preview}")


def summarize_pairs(pairs: list[dict[str, Any]], record_count: int) -> dict[str, Any]:
    retained = [pair for pair in pairs if pair["retain"]]
    exclusions = Counter(
        reason for pair in pairs for reason in pair.get("exclusion_reasons", [])
    )
    return {
        "records": record_count,
        "possible_pairs": len(pairs),
        "retained_pairs": len(retained),
        "retained_prompts": len({pair["prompt_id"] for pair in retained}),
        "pairs_by_split": dict(Counter(pair["split"] for pair in pairs)),
        "retained_by_split": dict(Counter(pair["split"] for pair in retained)),
        "retained_by_confidence": dict(
            Counter(pair["label_confidence"] for pair in retained)
        ),
        "retained_by_comparison_type": dict(
            Counter(pair["comparison_type"] for pair in retained)
        ),
        "exclusion_reasons": dict(exclusions),
        "preferred_field_agreement": {
            "agrees": sum(pair["preferred_field_agrees"] is True for pair in pairs),
            "disagrees": sum(pair["preferred_field_agrees"] is False for pair in pairs),
            "not_applicable": sum(pair["preferred_field_agrees"] is None for pair in pairs),
        },
        "methodology": {
            "split_unit": "normalized exact prompt",
            "near_duplicate_clustering": False,
            "failed_judgment_handling": "excluded; never converted to zero",
            "label_margin_units": "raw judge-score units",
            "pair_weight": "1.0 for the equal-weight DPO baseline",
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-share", type=float, default=0.80)
    parser.add_argument("--validation-share", type=float, default=0.10)
    parser.add_argument("--min-common-judges", type=int, default=2)
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument(
        "--score-span",
        type=float,
        default=4.0,
        help="Maximum score minus minimum score; 4.0 for a 1-to-5 scale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Input JSON must be a list of SDT records")

    pairs = build_pairs(
        records,
        seed=args.seed,
        train_share=args.train_share,
        validation_share=args.validation_share,
        min_common_judges=args.min_common_judges,
        min_margin=args.min_margin,
        min_confidence=args.min_confidence,
        score_span=args.score_span,
    )
    validate_no_prompt_leakage(pairs)
    write_jsonl(args.output, pairs)

    report = summarize_pairs(pairs, len(records))
    report["configuration"] = {
        "seed": args.seed,
        "train_share": args.train_share,
        "validation_share": args.validation_share,
        "test_share": round(1.0 - args.train_share - args.validation_share, 10),
        "min_common_judges": args.min_common_judges,
        "min_margin": args.min_margin,
        "min_confidence": args.min_confidence,
        "score_span": args.score_span,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
