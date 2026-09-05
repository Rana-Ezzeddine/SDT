from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sdt_dpo.generate import load_prompts
from sdt_dpo.judge_generations import (
    DIMENSIONS,
    _dpo_is_a,
    _extract_json,
    _validate_judgment,
    summarize,
)


class PilotEvaluationTests(unittest.TestCase):
    def test_generation_uses_each_retained_prompt_once(self) -> None:
        rows = [
            {"prompt_id": "p1", "prompt": "one", "split": "test", "retain": True, "label_margin": 0.5},
            {"prompt_id": "p1", "prompt": "one", "split": "test", "retain": True, "label_margin": 0.4},
            {"prompt_id": "p2", "prompt": "two", "split": "train", "retain": True, "label_margin": 0.5},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            prompts = load_prompts(path, split="test", min_margin=0.1)
        self.assertEqual(prompts, [{"prompt_id": "p1", "prompt": "one"}])

    def test_blind_order_is_deterministic(self) -> None:
        self.assertEqual(_dpo_is_a("prompt", 42), _dpo_is_a("prompt", 42))

    def test_judge_json_and_summary(self) -> None:
        scores = {dimension: 4 for dimension in DIMENSIONS}
        parsed = _extract_json(
            "```json\n"
            + json.dumps(
                {
                    "winner": "A",
                    "scores": {"A": scores, "B": scores},
                    "reason": "A is clearer.",
                }
            )
            + "\n```"
        )
        judgment = _validate_judgment(parsed)
        self.assertEqual(judgment["winner"], "A")
        rows = [
            {
                "winner": "dpo",
                "baseline_scores": {dimension: 3 for dimension in DIMENSIONS},
                "dpo_scores": {dimension: 4 for dimension in DIMENSIONS},
            },
            {
                "winner": "tie",
                "baseline_scores": {dimension: 4 for dimension in DIMENSIONS},
                "dpo_scores": {dimension: 4 for dimension in DIMENSIONS},
            },
        ]
        report = summarize(rows)
        self.assertEqual(report["dpo_wins"], 1)
        self.assertEqual(report["ties"], 1)
        self.assertEqual(report["tie_adjusted_dpo_score"], 0.75)
        self.assertEqual(report["mean_dimension_delta_dpo_minus_baseline"]["autonomy"], 0.5)


if __name__ == "__main__":
    unittest.main()
