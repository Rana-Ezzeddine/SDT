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
    _normalized_response,
    _validate_judgment,
    summarize,
)
from sdt_dpo.judge_generations_local import _local_judge_config_fingerprint


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

    def test_identical_response_normalization_only_ignores_whitespace(self) -> None:
        self.assertEqual(_normalized_response("same\n answer"), "same answer")
        self.assertNotEqual(_normalized_response("Same answer"), "same answer")

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
        self.assertEqual(report["api_judged_n"], 2)
        self.assertEqual(report["mean_dimension_delta_dpo_minus_baseline"]["autonomy"], 0.5)

    def test_deterministic_identical_ties_do_not_fabricate_dimension_scores(self) -> None:
        rows = [
            {
                "winner": "dpo",
                "judgment_type": "llm_judge",
                "baseline_scores": {dimension: 3 for dimension in DIMENSIONS},
                "dpo_scores": {dimension: 4 for dimension in DIMENSIONS},
            },
            {
                "winner": "tie",
                "judgment_type": "deterministic_identical_response",
            },
        ]
        report = summarize(rows)
        self.assertEqual(report["n"], 2)
        self.assertEqual(report["api_judged_n"], 1)
        self.assertEqual(report["deterministic_identical_ties"], 1)
        self.assertEqual(report["tie_adjusted_dpo_score"], 0.75)
        self.assertEqual(
            report["mean_dimension_delta_dpo_minus_baseline"]["autonomy"], 1.0
        )

    def test_local_judge_fingerprint_covers_inputs_and_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.jsonl"
            dpo = Path(directory) / "dpo.jsonl"
            baseline.write_text('{"prompt_id":"p","response":"a"}\n')
            dpo.write_text('{"prompt_id":"p","response":"b"}\n')
            first, manifest = _local_judge_config_fingerprint(
                baseline_path=baseline,
                dpo_path=dpo,
                judge_model="judge",
                seed=42,
                max_new_tokens=512,
            )
            second, _ = _local_judge_config_fingerprint(
                baseline_path=baseline,
                dpo_path=dpo,
                judge_model="judge",
                seed=42,
                max_new_tokens=256,
            )
        self.assertNotEqual(first, second)
        self.assertEqual(manifest["backend"], "local_transformers")


if __name__ == "__main__":
    unittest.main()
