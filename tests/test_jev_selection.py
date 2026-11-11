import unittest
import json
import tempfile
from pathlib import Path
from sdt_dpo.generate import load_prompts
from sdt_dpo.selection import eligible_pair
from sdt_dpo.jev_pairs import build, probability
from test_jev_pairs import record


class JevSelectionTests(unittest.TestCase):
    def test_gate_overrides_sdt_ranking_and_margin_filter(self):
        r = record()
        r['evaluations']['sdt-augmented']['content_score_average'] = .79
        r['preferred_response'] = 'sdt'
        rows = build([r])
        gate = [p for p in rows if p['preference_basis'] == 'content_gate']
        self.assertEqual(len(gate), 2)
        self.assertTrue(all(p['chosen_sdt_score'] < p['rejected_sdt_score'] for p in gate))
        self.assertTrue(all(eligible_pair(p, min_sdt_margin=1) for p in gate))

    def test_threshold_is_inclusive(self):
        r = record()
        for e in r['evaluations'].values():
            e['content_score_average'] = .8
        self.assertEqual(sum(p['retain'] for p in build([r])), 3)

    def test_jev_ignores_legacy_confidence_and_margin(self):
        p = build([record()])[0]
        self.assertTrue(eligible_pair(p, min_margin=1))
        self.assertFalse(eligible_pair(p, min_sdt_margin=1))
        p['retain'] = False
        self.assertFalse(eligible_pair(p))

    def test_invalid_probabilities(self):
        for value in (True, None, float('nan'), float('inf'), -.1, 1.1, '0.8'):
            with self.assertRaises(ValueError):
                probability(value)

    def test_normalized_prompt_group_does_not_leak(self):
        a, b = record(), record()
        b.update(id=2, instruction='  EXPLAIN  ')
        rows = build([a, b])
        self.assertEqual(len({r['prompt_id'] for r in rows}), 1)
        self.assertEqual(len({r['split'] for r in rows}), 1)

    def test_legacy_filter_preserved(self):
        row = dict(retain=True, label_margin=.2, label_confidence='high')
        self.assertTrue(eligible_pair(row, min_margin=.1))
        self.assertFalse(eligible_pair(row, min_margin=.3))

    def test_generation_preserves_small_gap_quality_gate(self):
        row = build([record()])[0]
        row.update(split='test', preference_basis='content_gate', sdt_margin=0.0, label_margin=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pairs.jsonl'
            path.write_text(json.dumps(row) + '\n')
            self.assertEqual(len(load_prompts(path, split='test', min_margin=.1)), 1)
