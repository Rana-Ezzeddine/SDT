from __future__ import annotations

import json
import unittest
from pathlib import Path

from sdt_dpo.pairs import (
    assign_split,
    build_pairs,
    construct_pair,
    summarize_judgment_evidence,
    valid_judge_scores,
    validate_no_prompt_leakage,
)


def judgment(judge: str, scores: list[float], status: str = "success") -> dict:
    return {
        "judge_model": judge,
        "status": status,
        "scores": [{"judge_score": score} for score in scores],
        "score_avg": sum(scores) / len(scores) if scores else 0.0,
        "score_std": 0.25,
    }


def sample_record() -> dict:
    return {
        "id": "r1",
        "instruction": "Explain photosynthesis.",
        "responses": {"a": "better", "b": "worse", "c": "middle"},
        "preferred_response": "a",
        "evaluations": {
            "a": {"judgments": [judgment("j1", [5, 5, 5]), judgment("j2", [4, 4, 4])]},
            "b": {"judgments": [judgment("j1", [2, 2, 2]), judgment("j2", [2, 2, 2])]},
            "c": {"judgments": [judgment("j1", [3, 3, 3]), judgment("j2", [3, 3, 3])]},
        },
    }


class PairPreparationTests(unittest.TestCase):
    def test_failed_judgment_is_missing_not_zero(self) -> None:
        evaluation = {
            "judgments": [
                judgment("valid", [4, 5]),
                judgment("failed", [0], status="failed"),
            ]
        }
        scores = valid_judge_scores(evaluation)
        self.assertEqual(set(scores), {"valid"})
        self.assertEqual(scores["valid"]["mean"], 4.5)

    def test_pair_direction_margin_and_equal_weight(self) -> None:
        pair = construct_pair(
            sample_record(),
            "a",
            "b",
            split="train",
            min_common_judges=2,
            min_margin=0.10,
            min_confidence=0.60,
            score_span=4.0,
        )
        self.assertEqual(pair["chosen"], "better")
        self.assertEqual(pair["rejected"], "worse")
        self.assertAlmostEqual(pair["label_margin"], 2.5)
        self.assertEqual(pair["judge_agreement"], 1.0)
        self.assertEqual(pair["pair_weight"], 1.0)
        self.assertIs(pair["retain"], True)

    def test_supplied_aggregates_are_used_without_recalculating_attempts(self) -> None:
        evaluation = {
            "judgments": [
                {
                    "judge_model": "j1",
                    "status": "success",
                    "scores": [{"judge_score": 1.0}, {"judge_score": 1.0}],
                    "score_avg": 4.25,
                    "score_std": 0.4,
                }
            ]
        }
        scores = valid_judge_scores(evaluation, use_supplied_aggregates=True)
        self.assertEqual(scores["j1"]["mean"], 4.25)
        self.assertEqual(scores["j1"]["repeat_std"], 0.4)
        self.assertEqual(scores["j1"]["valid_attempts"], 2.0)

    def test_single_judge_pilot_retains_pair_without_fake_confidence(self) -> None:
        record = sample_record()
        for evaluation in record["evaluations"].values():
            evaluation["judgments"] = evaluation["judgments"][:1]
        pairs = build_pairs(
            [record],
            min_common_judges=1,
            label_mode="single_judge_pilot",
            use_supplied_aggregates=True,
        )
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(pair["retain"] for pair in pairs))
        self.assertTrue(all(pair["confidence_score"] is None for pair in pairs))
        self.assertTrue(all(pair["judge_agreement"] is None for pair in pairs))
        self.assertTrue(
            all(pair["label_confidence"] == "single_judge_pilot" for pair in pairs)
        )

    def test_single_judge_mode_rejects_failed_zero_placeholder(self) -> None:
        record = sample_record()
        record["evaluations"]["a"]["judgments"] = [
            judgment("j1", [], status="failed")
        ]
        record["evaluations"]["b"]["judgments"] = [judgment("j1", [2, 2, 2])]
        pair = construct_pair(
            record,
            "a",
            "b",
            split="train",
            min_common_judges=1,
            min_margin=0.10,
            min_confidence=0.60,
            score_span=4.0,
            label_mode="single_judge_pilot",
            use_supplied_aggregates=True,
        )
        self.assertFalse(pair["retain"])
        self.assertIn(
            "insufficient_common_successful_judges", pair["exclusion_reasons"]
        )

    def test_judgment_audit_reports_duplicate_attempt_texts(self) -> None:
        record = sample_record()
        record["evaluations"]["a"]["judgments"][0]["scores"] = [
            {"judge_score": 5, "judge_response": "same"},
            {"judge_score": 5, "judge_response": "same"},
            {"judge_score": 4, "judge_response": "different"},
        ]
        report = summarize_judgment_evidence([record])
        self.assertEqual(report["judgment_blocks"], 6)
        self.assertEqual(report["unique_judge_response_texts_per_block"]["2"], 1)

    def test_three_candidates_make_three_pairs(self) -> None:
        pairs = build_pairs([sample_record()])
        self.assertEqual(len(pairs), 3)
        self.assertEqual(len({pair["prompt_id"] for pair in pairs}), 1)
        self.assertEqual(len({pair["split"] for pair in pairs}), 1)

    def test_formatting_duplicate_prompts_share_split(self) -> None:
        self.assertEqual(
            assign_split(" Hello   WORLD ", seed=7), assign_split("hello world", seed=7)
        )

    def test_leakage_validator_rejects_cross_split_prompt(self) -> None:
        rows = [
            {"prompt_id": "same", "split": "train"},
            {"prompt_id": "same", "split": "test"},
        ]
        with self.assertRaisesRegex(ValueError, "Prompt leakage"):
            validate_no_prompt_leakage(rows)

    def test_included_sample_reproduces_audited_counts(self) -> None:
        raw_path = Path(__file__).resolve().parents[1] / "data/raw/sdt_100_llama.json"
        records = json.loads(raw_path.read_text(encoding="utf-8"))
        pairs = build_pairs(records, train_share=0.70, validation_share=0.15, seed=42)
        retained = [pair for pair in pairs if pair["retain"]]
        self.assertEqual(len(records), 100)
        self.assertEqual(len(pairs), 300)
        self.assertEqual(len(retained), 188)
        self.assertEqual(
            {split: sum(pair["split"] == split for pair in retained) for split in ("train", "validation", "test")},
            {"train": 139, "validation": 27, "test": 22},
        )
        validate_no_prompt_leakage(pairs)


if __name__ == "__main__":
    unittest.main()
