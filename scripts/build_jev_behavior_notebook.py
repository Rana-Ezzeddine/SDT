"""Create a clean pilot with separate development and locked-test stages."""
import json
from pathlib import Path

def main():
    cells = []
    def md(s):
        cells.append(dict(cell_type='markdown', metadata={}, source=s.splitlines(True)))
    def code(s):
        cells.append(dict(cell_type='code', metadata={}, source=s.splitlines(True), outputs=[], execution_count=None))
    md('''# Jev behavioral pilot · Qwen 0.5B full DPO
2,000 train / 200 validation / 300 test prompts. New run, no old checkpoint reuse.
Length-bounded to 1,024 tokens; exact prompt grouping, not semantic deduplication.
This notebook prepares blinded JEV inputs; it does not invent JEV scores or claim behavioral success from DPO metrics.
Reserve at least 30 GB of Drive storage for both runs and resumable checkpoints. Run one candidate at a time if needed.
''')
    code('''from pathlib import Path
import os, sys, json, subprocess, hashlib, shutil, zipfile
from datetime import datetime, timezone
import torch
assert torch.cuda.is_available(), "Connect a GPU runtime."
print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
from google.colab import files, drive
uploaded = files.upload()  # jev_behavior_pilot.zip
assert len(uploaded) == 1
bundle = Path('/content/jev_behavior_pilot.zip')
bundle.write_bytes(next(iter(uploaded.values())))
WORK = Path('/content/jev-behavior-pilot')
with zipfile.ZipFile(bundle) as z:
    assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
    z.extractall(WORK)
os.chdir(WORK)
os.environ['PYTHONPATH'] = str(WORK/'src')
sys.path.insert(0, str(WORK/'src'))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==5.5.3', 'trl==1.17.1', 'accelerate==1.15.0', 'datasets==4.8.5', '-e', '.'], check=True)
import yaml
DRIVE_MOUNT = Path('/content/pilot-drive')
drive.mount(str(DRIVE_MOUNT))
assert os.path.ismount(DRIVE_MOUNT) and (DRIVE_MOUNT/'MyDrive').is_dir(), 'Drive is not mounted. Stop.'
''')
    md('## 1. Create or explicitly resume one pilot folder')
    code('''assert os.path.ismount(DRIVE_MOUNT) and (DRIVE_MOUNT/'MyDrive').is_dir(), 'Drive is not mounted. Stop.'
RESUME_RUN = ''  # paste an existing exact Drive path only to resume this experiment
RUN = Path(RESUME_RUN) if RESUME_RUN else DRIVE_MOUNT/'MyDrive/SDT_JEV_Behavior_Pilot_v2' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
assert RUN.resolve().is_relative_to(DRIVE_MOUNT.resolve()), 'Run must be on mounted Drive.'
RUN.mkdir(parents=True, exist_ok=bool(RESUME_RUN))
for source in [bundle, WORK/'pilot-pairs.jsonl', WORK/'sample-report.json', WORK/'sampling-audit.json']:
    target = RUN/source.name
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest(), 'Run inputs changed'
    else:
        shutil.copy2(source, target)
PAIRS = RUN/'pilot-pairs.jsonl'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
print('SAVE THIS PATH:', RUN)
print((RUN/'sample-report.json').read_text())
def run(name, module, *args):
    assert os.path.ismount(DRIVE_MOUNT), 'Drive disconnected. Stop.'
    with (RUN/(name+'.log')).open('a') as log:
        p = subprocess.Popen([sys.executable, '-u', '-m', module, *map(str,args)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end=''); log.write(line); log.flush()
        if p.wait() != 0:
            raise RuntimeError(f'{name} failed; its log is saved. Do not continue.')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
''')
    md('## 2. Train two candidates; each selects its best epoch using validation only\nBoth start from the original model. Test results must not guide selection. Interrupted training resumes from an available checkpoint; completed models must pass weight hashes before reuse.')
    code('''EXPERIMENTS = [('lr5e7', 5e-7), ('lr1e6', 1e-6)]
for name, lr in EXPERIMENTS:
    model = RUN/name/'model'
    cfg = yaml.safe_load((WORK/'configs/jev_phi4_dpo.yaml').read_text())
    cfg.update(model_id=MODEL, pairs_file=str(PAIRS), output_dir=str(model), learning_rate=lr,
        num_train_epochs=2, max_length=1024, eval_strategy='epoch', save_strategy='epoch',
        logging_steps=10, save_total_limit=1, metric_for_best_model='eval_rewards/accuracies')
    config = RUN/(name+'.yaml')
    if config.exists(): assert yaml.safe_load(config.read_text()) == cfg
    else: config.write_text(yaml.safe_dump(cfg))
    receipt = RUN/name/'weights-verified.json'
    if not receipt.exists():
        run(name+'-training', 'sdt_dpo.train', '--config', config)
        weights = list(model.glob('*.safetensors'))
        assert weights and (model/'run_manifest.json').exists()
        receipt.write_text(json.dumps({p.name:dict(bytes=p.stat().st_size, sha256=digest(p)) for p in weights}, indent=2))
    for filename, expected in json.loads(receipt.read_text()).items():
        assert (model/filename).stat().st_size == expected['bytes']
        assert digest(model/filename) == expected['sha256']
    print(name, json.loads((model/'validation_metrics.json').read_text()))
''')
    md('## 3. Validation comparison and generated answers\nFixed pairs diagnose preference learning. Generated answers are needed to assess actual SDT behavior. All models use identical decoding; no SDT system-prompt advantage.')
    code('''for name, model in [('baseline',MODEL)] + [(n,str(RUN/n/'model')) for n,_ in EXPERIMENTS]:
    run(name+'-validation', 'sdt_dpo.evaluate', '--pairs', PAIRS, '--model', model,
        '--split','validation','--max-length','1024','--output',RUN/(name+'-validation.json'),
        '--details',RUN/(name+'-validation-pairs.jsonl'))
    run(name+'-validation-generation', 'sdt_dpo.generate', '--pairs',PAIRS,'--model',model,
        '--split','validation','--max-new-tokens','512','--temperature','0',
        '--output',RUN/(name+'-validation-generations.jsonl'))
    if name != 'baseline':
        run(name+'-validation-comparison','sdt_dpo.compare','--baseline-details',RUN/'baseline-validation-pairs.jsonl',
            '--dpo-details',RUN/(name+'-validation-pairs.jsonl'),'--output',RUN/(name+'-validation-comparison.json'))
''')
    md('## 4. Prepare anonymous responses for JEV\nSend only the blinded file to the evaluator; keep the mapping private. Agree with Ghadi on the exact JEV prompt/schema and quality rule. Choose a candidate using validation SDT and quality results, not test results. Neither longer answers nor higher DPO implicit accuracy alone establishes success.')
    code('''import random
def export_blind(split, names):
    rows, mapping, expected_ids = [], {}, None
    for name in names:
        source = RUN/(name+'-'+split+'-generations.jsonl')
        records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        ids = {r['prompt_id'] for r in records}
        assert len(ids) == len(records)
        if expected_ids is None: expected_ids = ids
        else: assert ids == expected_ids, 'Generation sets differ'
        for r in records:
            key = hashlib.sha256((split+name+r['prompt_id']).encode()).hexdigest()[:24]
            # The repository generation schema stores response text in response.
            text = r['response']
            assert text.strip()
            rows.append(dict(id=key, prompt=r['prompt'], response=text))
            mapping[key] = dict(model=name, prompt_id=r['prompt_id'])
    random.Random(42).shuffle(rows)
    (RUN/(split+'-jev-blinded.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\\n' for r in rows))
    (RUN/(split+'-jev-private-map.json')).write_text(json.dumps(mapping,indent=2))
    print('Prepared',len(rows),'responses for',split)
export_blind('validation', ['baseline']+[n for n,_ in EXPERIMENTS])
''')
    md('## 5. Freeze selection, then unlock the test once\nLeave disabled until validation scoring is reviewed. Set the reason before opening test results. Do not retune on this test. A quality drop may justify selecting neither candidate.')
    code('''RUN_LOCKED_TEST = False
SELECTED_NAME = ''
SELECTION_REASON = ''
if RUN_LOCKED_TEST:
    assert SELECTED_NAME in dict(EXPERIMENTS) and SELECTION_REASON.strip()
    selected = dict(name=SELECTED_NAME, reason=SELECTION_REASON)
    lock = RUN/'selected-configuration.json'
    if lock.exists(): assert json.loads(lock.read_text()) == selected, 'Test selection is already frozen'
    else: lock.write_text(json.dumps(selected,indent=2))
    for name, model in [('baseline',MODEL),(SELECTED_NAME,str(RUN/SELECTED_NAME/'model'))]:
        run(name+'-test','sdt_dpo.evaluate','--pairs',PAIRS,'--model',model,'--split','test',
            '--max-length','1024','--output',RUN/(name+'-test.json'),'--details',RUN/(name+'-test-pairs.jsonl'))
        run(name+'-test-generation','sdt_dpo.generate','--pairs',PAIRS,'--model',model,'--split','test',
            '--max-new-tokens','512','--temperature','0','--output',RUN/(name+'-test-generations.jsonl'))
    run('test-comparison','sdt_dpo.compare','--baseline-details',RUN/'baseline-test-pairs.jsonl',
        '--dpo-details',RUN/(SELECTED_NAME+'-test-pairs.jsonl'),'--output',RUN/'test-comparison.json')
    export_blind('test',['baseline',SELECTED_NAME])
else:
    print('Locked test not opened. Finish validation JEV scoring first.')
''')
    md('## 6. Verify Drive persistence before disconnecting')
    code('''drive.flush_and_unmount()
drive.mount(str(DRIVE_MOUNT))
assert os.path.ismount(DRIVE_MOUNT), 'Drive remount failed.'
for name,_ in EXPERIMENTS:
    model = RUN/name/'model'
    receipt = RUN/name/'weights-verified.json'
    assert receipt.exists(), name
    for filename, expected in json.loads(receipt.read_text()).items():
        assert (model/filename).stat().st_size == expected['bytes']
        assert digest(model/filename) == expected['sha256']
print('PASS: saved model weights verified after Drive remount.',RUN)
''')
    out = Path(__file__).resolve().parents[1]/'notebooks/SDT_Jev_Behavior_Pilot_Colab.ipynb'
    out.write_text(json.dumps(dict(cells=cells,metadata=dict(accelerator='GPU', kernelspec=dict(name='python3',display_name='Python 3')), nbformat=4,nbformat_minor=5),indent=1)+'\n')
    print(out)

if __name__ == '__main__': main()
