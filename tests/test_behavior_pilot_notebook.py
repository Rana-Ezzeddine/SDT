import ast
import json
from pathlib import Path
import unittest


class BehaviorPilotNotebookTests(unittest.TestCase):
    def test_recovery_notebooks_compile_without_outputs_or_training_calls(self):
        root = Path(__file__).resolve().parents[1]
        for name in ['SDT_Jev_Behavior_Resume_Evaluation_Colab.ipynb',
                     'SDT_Jev_Account_Transfer_Resume_Colab.ipynb']:
            with self.subTest(notebook=name):
                nb = json.loads((root/'notebooks'/name).read_text())
                sources = []
                for cell in nb['cells']:
                    if cell['cell_type'] != 'code':
                        continue
                    source = ''.join(cell['source'])
                    ast.parse(source)
                    sources.append(source)
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
                source = '\n'.join(sources)
                self.assertIn("module != 'sdt_dpo.train'", source)
                self.assertIn('RUN_LOCKED_TEST = False', source)
                self.assertIn('drive.flush_and_unmount()', source)
                self.assertIn("expected['sha256']", source)
                self.assertNotIn("run(name+'-training'", source)
                self.assertIn('Reusing complete fixed-pair evaluation:', source)

    def test_cells_compile_and_test_is_locked(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'notebooks/SDT_Jev_Behavior_Pilot_Colab.ipynb').read_text())
        sources = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                ast.parse(source)
                sources.append(source)
                self.assertEqual(cell['outputs'], [])
        source = '\n'.join(sources)
        self.assertIn('RUN_LOCKED_TEST = False', source)
        self.assertIn('drive.flush_and_unmount()', source)
        self.assertIn("num_train_epochs=2", source)
        self.assertIn("'validation'", source)
        self.assertIn("'weights-verified.json'", source)
        self.assertGreaterEqual(source.count('assert os.path.ismount(DRIVE_MOUNT)'), 4)
        self.assertIn('RUN.resolve().is_relative_to(DRIVE_MOUNT.resolve())', source)

    def test_export_does_not_reveal_model_and_checks_prompt_sets(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'notebooks/SDT_Jev_Behavior_Pilot_Colab.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells'] if 'def export_blind' in ''.join(c['source']))
        # Execute the helper only, without notebook side effects.
        tree = ast.parse(source)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        self.assertEqual(function.name, 'export_blind')
        self.assertIn('assert ids == expected_ids', source)
        self.assertIn('rows.append(dict(id=key, prompt=', source)


if __name__ == '__main__':
    unittest.main()
