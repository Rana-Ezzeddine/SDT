import copy
import unittest
from sdt_dpo.jev_pairs import build


def record():
    sources = ['baseline', 'sdt', 'sdt-augmented']
    return dict(id=1, model_name='generator', instruction='Explain', input='gravity',
                status='success', preferred_response='sdt-augmented',
                responses={s:s for s in sources}, evaluations={s:dict(
                    content_score_average=.9, sdt_score_average=(i+1)/4)
                    for i,s in enumerate(sources)})


class JevTests(unittest.TestCase):
    def test_three_pairs_and_prompt(self):
        pairs = build([record()])
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(p['retain'] for p in pairs))
        self.assertIn('gravity', pairs[0]['prompt'])
        self.assertEqual(len({p['split'] for p in pairs}), 1)

    def test_one_passing(self):
        r = record()
        for s in ['baseline','sdt']:
            r['evaluations'][s]['content_score_average'] = .7
        pairs = build([r])
        self.assertEqual(sum(p['retain'] for p in pairs), 2)
        self.assertTrue(all(p['chosen_source']=='sdt-augmented' for p in pairs if p['retain']))

    def test_ties_identical_and_failed(self):
        r = record()
        r['evaluations']['sdt']['sdt_score_average'] = .75
        self.assertEqual(sum(p['retain'] for p in build([r])), 2)
        r['responses']['baseline'] = r['responses']['sdt']
        self.assertEqual(sum(p['retain'] for p in build([r])), 1)
        for ev in r['evaluations'].values():
            ev['content_score_average'] = .7
        self.assertFalse(any(p['retain'] for p in build([r])))

    def test_validation_and_identity(self):
        r = record()
        r['preferred_response']='baseline'
        with self.assertRaises(ValueError): build([r])
        r = record()
        with self.assertRaises(ValueError): build([r,r])
        other = copy.deepcopy(r)
        other['model_name']='other'
        pairs = build([r,other])
        self.assertEqual(len({p['pair_id'] for p in pairs}),6)
        self.assertEqual(len({p['split'] for p in pairs}),1)
