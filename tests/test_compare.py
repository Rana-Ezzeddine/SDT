from __future__ import annotations

import unittest

from sdt_dpo.compare import compare, exact_mcnemar_p


def row(
    *,
    prompt_id: str,
    correct: bool,
    margin: float,
    chosen_sum: float,
    rejected_sum: float,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "correct": correct,
        "model_margin": margin,
        "chosen_logprob_sum": chosen_sum,
        "rejected_logprob_sum": rejected_sum,
        "chosen_token_count": 10,
        "rejected_token_count": 8,
    }


class ComparisonTests(unittest.TestCase):
    def test_paired_comparison_includes_dpo_relative_metrics(self) -> None:
        base = {
            "1": row(
                prompt_id="p1",
                correct=False,
                margin=-0.2,
                chosen_sum=-10.0,
                rejected_sum=-8.0,
            ),
            "2": row(
                prompt_id="p1",
                correct=True,
                margin=0.1,
                chosen_sum=-5.0,
                rejected_sum=-6.0,
            ),
            "3": row(
                prompt_id="p2",
                correct=False,
                margin=-0.1,
                chosen_sum=-9.0,
                rejected_sum=-8.0,
            ),
        }
        dpo = {
            "1": row(
                prompt_id="p1",
                correct=True,
                margin=0.3,
                chosen_sum=-8.0,
                rejected_sum=-7.0,
            ),
            "2": row(
                prompt_id="p1",
                correct=True,
                margin=0.2,
                chosen_sum=-4.5,
                rejected_sum=-5.0,
            ),
            "3": row(
                prompt_id="p2",
                correct=False,
                margin=-0.05,
                chosen_sum=-9.0,
                rejected_sum=-8.0,
            ),
        }
        report = compare(base, dpo, beta=0.1, bootstrap_samples=100)
        primary = report["primary_dpo_relative_metrics"]
        secondary = report["secondary_absolute_likelihood_metrics"]
        behavior = report["checkpoint_behavior_check"]

        self.assertEqual(report["paired_n"], 3)
        self.assertAlmostEqual(primary["implicit_reward_accuracy"], 1 / 3)
        self.assertAlmostEqual(secondary["accuracy_delta_dpo_minus_baseline"], 1 / 3)
        self.assertEqual(
            secondary["discordant_pairs"]["baseline_wrong_dpo_correct"], 1
        )
        self.assertEqual(behavior["pairs_with_changed_completion_logprobs"], 2)
        self.assertFalse(behavior["all_pairs_changed"])

    def test_pair_id_sets_must_match_exactly(self) -> None:
        base = {
            "1": row(
                prompt_id="p1",
                correct=True,
                margin=0.1,
                chosen_sum=-1.0,
                rejected_sum=-2.0,
            )
        }
        dpo = {
            "2": row(
                prompt_id="p2",
                correct=False,
                margin=-0.1,
                chosen_sum=-2.0,
                rejected_sum=-1.0,
            )
        }
        with self.assertRaisesRegex(ValueError, "exactly the same pair IDs"):
            compare(base, dpo)

    def test_exact_mcnemar_no_disagreement(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
