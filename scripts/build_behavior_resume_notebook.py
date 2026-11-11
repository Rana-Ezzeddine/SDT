"""Validate the browser-edited recovery notebook and export a clean copy."""
import ast
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True,
                    help='Downloaded, browser-edited recovery notebook to validate and clean.')
source = parser.parse_args().source
nb = json.loads(source.read_text())
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
for cell in code:
    ast.parse(cell)
assert '20261110T190910Z' in code[0]
assert "module != 'sdt_dpo.train'" in code[1]
assert 'Verified saved model:' in code[2]
assert 'sdt_dpo.train' not in code[2]
assert 'Reusing complete fixed-pair evaluation:' in code[3]
assert 'RUN_LOCKED_TEST = False' in code[5]
for c in nb['cells']:
    if c['cell_type'] == 'code':
        c['outputs'] = []
        c['execution_count'] = None
        c['metadata'] = {}
nb['cells'][0]['source'] = ['# Resume saved Jev pilot — evaluation only\n',
    'Existing run: `20261110T190910Z`. No training and no new run folder.\n',
    'When a GPU is available, run the code cells from the top. Authorize Drive when prompted.\n',
    'Completed evaluations are reused; generation resumes by prompt ID. Test remains locked.\n']
nb['cells'][2]['source'] = ['## 1. Restore existing run and helpers\n']
nb['cells'][4]['source'] = ['## 2. Verify both saved models (no training)\n']
nb['metadata'].pop('widgets', None)
target = Path(__file__).resolve().parents[1]/'notebooks/SDT_Jev_Behavior_Resume_Evaluation_Colab.ipynb'
target.write_text(json.dumps(nb, indent=1)+'\n')
print('Validated', len(code), 'code cells. Saved:', target)
