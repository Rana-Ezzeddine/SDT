"""Build an evaluation-only recovery notebook using a copied/shared backup."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
nb = json.loads((root/'notebooks/SDT_Jev_Behavior_Resume_Evaluation_Colab.ipynb').read_text())
nb['cells'][0]['source'] = ['# Restore saved Jev pilot in an authorized Colab account\n',
    'No training. No locked-test evaluation. GPU access remains subject to Colab limits.\n',
    'First make the private backup folder accessible in this account’s My Drive (a shared-folder shortcut also works). '
    'Set SOURCE_DIR below to that folder. The backup is read-only input; results are copied into a separate run folder.\n']
setup = '''from pathlib import Path
import os, sys, json, subprocess, hashlib, shutil, zipfile
from google.colab import drive
import torch
assert torch.cuda.is_available(), 'A permitted GPU runtime is required. No training is needed.'
DRIVE_MOUNT = Path('/content/pilot-drive')
drive.mount(str(DRIVE_MOUNT))
assert os.path.ismount(DRIVE_MOUNT), 'Drive mount failed.'

# Change ONLY this path if the copied backup/shortcut has another name.
SOURCE_DIR = DRIVE_MOUNT/'MyDrive/SDT_JEV_Backup_20261110T190910Z'
RUN = DRIVE_MOUNT/'MyDrive/SDT_JEV_Behavior_Pilot_v2/20261110T190910Z'
assert SOURCE_DIR.is_dir(), f'Backup folder not found: {SOURCE_DIR}'
assert SOURCE_DIR.resolve() != RUN.resolve(), 'Source backup and working run must differ.'
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
RUN.mkdir(parents=True, exist_ok=True)
def copy_once(source, target):
    if target.exists():
        return  # Never overwrite resumed outputs. Models/inputs are verified below.
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name+'.copying')
    shutil.copy2(source, partial)
    assert digest(partial) == digest(source), f'Copy mismatch: {source}'
    partial.replace(target)

# The separate backup stores each final model directly under its candidate folder.
# Restore the original model paths so recorded generation metadata remains valid.
for name in ['lr5e7', 'lr1e6']:
    source_model = SOURCE_DIR/name
    receipt = source_model/'weights-verified.json'
    assert receipt.is_file(), f'Missing receipt: {receipt}'
    for filename, expected in json.loads(receipt.read_text()).items():
        p = source_model/filename
        assert p.is_file() and p.stat().st_size == expected['bytes'], p
        assert digest(p) == expected['sha256'], f'Source hash mismatch: {p}'
    for p in source_model.iterdir():
        if p.is_file():
            target = RUN/name/p.name if p.name == 'weights-verified.json' else RUN/name/'model'/p.name
            copy_once(p, target)
for p in SOURCE_DIR.iterdir():
    if p.is_file(): copy_once(p, RUN/p.name)
for filename in ['pilot-pairs.jsonl', 'sample-report.json', 'sampling-audit.json', 'jev_behavior_pilot.zip']:
    assert (RUN/filename).is_file(), filename
    assert digest(RUN/filename) == digest(SOURCE_DIR/filename), f'Input mismatch: {filename}'

bundle = RUN/'jev_behavior_pilot.zip'
WORK = Path('/content/jev-behavior-pilot')
with zipfile.ZipFile(bundle) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(WORK)
os.chdir(WORK)
os.environ['PYTHONPATH'] = str(WORK/'src')
sys.path.insert(0, str(WORK/'src'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.5.3', 'trl==1.17.1', 'accelerate==1.15.0', 'datasets==4.8.5', '-e', '.'], check=True)
import yaml
print('Restored existing run:', RUN)
print('Source backup unchanged:', SOURCE_DIR)
'''
nb['cells'][1]['source'] = setup.splitlines(True)
for c in nb['cells']:
    if c['cell_type']=='code':
        ast.parse(''.join(c['source']))
        c['outputs']=[]
        c['execution_count']=None
        c['metadata']={}
assert 'sdt_dpo.train' not in ''.join(nb['cells'][5]['source'])
assert 'RUN_LOCKED_TEST = False' in ''.join(nb['cells'][11]['source'])
out = root/'notebooks/SDT_Jev_Account_Transfer_Resume_Colab.ipynb'
out.write_text(json.dumps(nb, indent=1)+'\n')
print(out)
