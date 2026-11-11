import json
import tempfile
import unittest
from pathlib import Path
from sdt_dpo.run_identity import training_identity, check_run_identity


class RunIdentityTests(unittest.TestCase):
    def test_resume_matches_data_and_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = root / 'pairs.jsonl'
            pairs.write_text('original')
            out = root / 'model'
            out.mkdir()
            config = {'learning_rate': 1e-6, 'resume_from_checkpoint': 'auto'}
            identity = training_identity(config, pairs)
            check_run_identity(out, identity)
            (out / 'training_identity.json').write_text(json.dumps(identity))
            check_run_identity(out, training_identity(config, pairs))
            with self.assertRaises(ValueError):
                check_run_identity(out, training_identity({'learning_rate': 2e-6}, pairs))
            pairs.write_text('changed')
            with self.assertRaises(ValueError):
                check_run_identity(out, training_identity(config, pairs))

    def test_unknown_existing_directory_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            file = out / 'config.json'
            file.write_text('{}')
            with self.assertRaises(ValueError):
                check_run_identity(out, {})
            self.assertTrue(file.exists())
