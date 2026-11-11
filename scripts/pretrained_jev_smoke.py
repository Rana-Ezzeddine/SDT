"""Small pretrained-model integration experiment on real Jev records, not a benchmark."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import yaml
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from sdt_dpo.jev_pairs import build
from sdt_dpo.pairs import write_jsonl
from sdt_dpo.train import _within_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    (repo / 'outputs').mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='pretrained-jev-', dir=repo / 'outputs'))
    print('Artifacts:', root, flush=True)
    model_id = 'HuggingFaceTB/SmolLM2-135M-Instruct'
    base = snapshot_download(model_id, allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt'])
    tokenizer = AutoTokenizer.from_pretrained(base)
    selected, sources, counts = [], {}, Counter()
    # Preserve production hash split assignments; sample each generator within each split.
    for path in args.input:
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        records = json.loads(raw)
        random.Random(42).shuffle(records)
        quota = dict(train=4, validation=2, test=2)
        for record in records:
            rows = [p for p in build([record]) if p['retain']]
            if not rows:
                continue
            split = rows[0]['split']
            if quota[split] == 0 or not all(_within_length(tokenizer, p, 256) for p in rows):
                continue
            quota[split] -= 1
            selected.extend(rows)
            counts[f'{record["model_name"]}:{split}:prompts'] += 1
            if not any(quota.values()):
                break
        assert not any(quota.values()), f'Insufficient eligible short records: {quota}'
    pair_file = root / 'sample-pairs.jsonl'
    write_jsonl(pair_file, selected)
    audit = dict(model_id=model_id, model_snapshot=base, inputs_sha256=sources,
        selection='seed-42 shuffle, all retained responses <=256 tokens; generator-balanced prompt quotas',
        prompt_counts=dict(counts), pairs_by_split=dict(Counter(p['split'] for p in selected)),
        seed=42, scope='pipeline check; short-example-biased sample, not final research test')
    (root / 'sample-report.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit, indent=2), flush=True)
    cfg = yaml.safe_load((repo / 'configs/jev_phi4_dpo.yaml').read_text())
    trained = root / 'trained'
    cfg.update(model_id=base, pairs_file=str(pair_file), output_dir=str(trained),
        learning_rate=1e-6, gradient_accumulation_steps=4, max_length=256,
        eval_strategy='epoch', save_strategy='epoch', logging_steps=1, warmup_ratio=0.0)
    config = root / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, PYTHONPATH=str(repo/'src'), TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='2', HF_HUB_OFFLINE='1')
    def run(name, module, *arguments):
        print('Running:', name, flush=True)
        with (root / f'{name}.log').open('w') as log:
            process = subprocess.run([sys.executable, '-m', module, *map(str, arguments)],
                                     env=env, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            print((root/f'{name}.log').read_text()[-8000:])
            raise RuntimeError(f'{name} failed; logs preserved at {root}')
        print('PASS:', name, flush=True)
    run('train', 'sdt_dpo.train', '--config', config)
    run('verify', 'sdt_dpo.verify_checkpoint', '--baseline-model', base,
        '--trained-model', trained, '--output', root/'checkpoint-change.json')
    for name, model in [('baseline', base), ('dpo', trained)]:
        run(f'evaluate-{name}', 'sdt_dpo.evaluate', '--pairs', pair_file, '--model', model,
            '--max-length', 256, '--output', root/f'{name}-summary.json',
            '--details', root/f'{name}-pairs.jsonl')
        run(f'generate-{name}', 'sdt_dpo.generate', '--pairs', pair_file, '--model', model,
            '--max-prompts', 2, '--max-new-tokens', 64, '--output', root/f'{name}-generations.jsonl')
    run('compare', 'sdt_dpo.compare', '--baseline-details', root/'baseline-pairs.jsonl',
        '--dpo-details', root/'dpo-pairs.jsonl', '--output', root/'comparison.json',
        '--bootstrap-samples', 1000)
    weights = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in trained.glob('*.safetensors')}
    assert weights
    (root/'completion.json').write_text(json.dumps(dict(status='PASS', weights_sha256=weights,
        fresh_process_reload='verified by evaluation and generation', research_conclusion=False), indent=2)+'\n')
    print('Completed:', root, flush=True)


if __name__ == '__main__':
    main()
