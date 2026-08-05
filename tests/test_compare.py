from __future__ import annotations

import unittest

from sdt_dpo.compare import compare, exact_mcnemar_p


def row(correct: bool, margin: float) -> dict:
    return {"correct": correct, "model_margin": margin}


class ComparisonTests(unittest.TestCase):
    def test_paired_comparison(self) -> None:
        base = {"1": row(False, -0.2), "2": row(True, 0.1), "3": row(False, -0.1)}
        dpo = {"1": row(True, 0.3), "2": row(True, 0.2), "3": row(False, -0.05)}
        report = compare(base, dpo, bootstrap_samples=100)
        self.assertEqual(report["paired_n"], 3)
        self.assertAlmostEqual(report["accuracy_delta_dpo_minus_baseline"], 1 / 3)
        self.assertEqual(report["discordant_pairs"]["baseline_wrong_dpo_correct"], 1)
        self.assertEqual(report["discordant_pairs"]["baseline_correct_dpo_wrong"], 0)

    def test_exact_mcnemar_no_disagreement(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
