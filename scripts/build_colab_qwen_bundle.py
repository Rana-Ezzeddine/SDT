"""Package current source and a deterministic real-data sample for Colab."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random
import zipfile

from transformers import AutoTokenizer
from sdt_dpo.jev_pairs import build
from sdt_dpo.train import _within_length


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True,
                        help='JEV JSON dataset; repeat for each generator.')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = repo / 'outputs/colab-qwen-smoke'
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    selected, hashes, counts = [], {}, Counter()
    for path in args.input:
        name = path.name
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        records = json.loads(raw)
        random.Random(42).shuffle(records)
        quota = {'train':32, 'validation':8, 'test':8}
        for record in records:
            rows = [r for r in build([record]) if r['retain']]
            if not rows:
                continue
            split = rows[0]['split']
            if not quota[split] or not all(_within_length(tokenizer, r, 512) for r in rows):
                continue
            quota[split] -= 1
            selected.extend(rows)
            counts[f'{name}:{split}:prompts'] += 1
            if not any(quota.values()):
                break
        assert not any(quota.values())
    report = dict(inputs_sha256=hashes, prompt_counts=dict(counts),
        pairs_by_split=dict(Counter(r['split'] for r in selected)),
        selection='seed42; balanced generators; production prompt hash splits; all retained responses <=512 tokens',
        purpose='development smoke test, not a final benchmark')
    bundle = out / 'sdt_colab_qwen_bundle.zip'
    with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as z:
        for pattern in ['src/sdt_dpo/*.py', 'tests/*.py', 'configs/*.yaml']:
            for path in repo.glob(pattern):
                z.write(path, str(path.relative_to(repo)))
        for name in ['pyproject.toml', 'README.md', 'scripts/run_sample_pipeline.sh',
                     'notebooks/SDT_Full_DPO_Colab.ipynb', 'notebooks/SDT_1500_DPO_Pilot_Colab.ipynb',
                     'notebooks/SDT_1500_DPO_Pilot_V2_Training_Colab.ipynb',
                     'notebooks/SDT_1500_DPO_Pilot_V2_Evaluation_Colab.ipynb']:
            z.write(repo/name, name)
        z.writestr('sample-pairs.jsonl', ''.join(json.dumps(p, ensure_ascii=False)+'\n' for p in selected))
        z.writestr('sample-report.json', json.dumps(report, indent=2)+'\n')
    print(bundle)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
