"""Build a development pilot bundle, preserving production prompt splits.

This is length-bounded, not an unbiased sample of the full dataset.
No model evaluation results are consulted during selection.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import zipfile

from transformers import AutoTokenizer
from sdt_dpo.jev_pairs import build
from sdt_dpo.train import _within_length
from sdt_dpo.pairs import validate_no_prompt_leakage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--exclude-bundle', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.exclude_bundle) as z:
        excluded = {r['prompt_id'] for r in map(json.loads, z.read('sample-pairs.jsonl').decode().splitlines())}
    groups = defaultdict(list)
    hashes = {}
    for path in args.input:
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        for row in build(json.loads(raw)):
            if row['retain'] and row['prompt_id'] not in excluded:
                groups[row['prompt_id']].append(row)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    selected, audit = [], []
    # Shuffle prompt groups, never individual pairs. All generators of a prompt stay together.
    ids = sorted(groups)
    random.Random(20261110).shuffle(ids)
    quotas = {'train': 2000, 'validation': 200, 'test': 300}
    counts = Counter()
    for pid in ids:
        rows = groups[pid]
        split = rows[0]['split']
        assert {r['split'] for r in rows} == {split}
        if counts[split] >= quotas[split]:
            continue
        fits = all(_within_length(tokenizer, r, 1024) for r in rows)
        audit.append({'prompt_id': pid, 'split': split, 'all_pairs_fit_1024': fits})
        if fits:
            counts[split] += 1
            selected.extend(rows)
        if dict(counts) == quotas:
            break
    assert dict(counts) == quotas, counts
    validate_no_prompt_leakage(selected)
    report = dict(inputs_sha256=hashes, selection_seed=20261110,
        prompts_by_split=dict(counts), pairs_by_split=dict(Counter(r['split'] for r in selected)),
        pairs_by_generator=dict(Counter(r['generator_model'] for r in selected)),
        pairs_by_basis=dict(Counter(r['preference_basis'] for r in selected)),
        prior_smoke_prompt_ids_excluded=sorted(excluded),
        length_filter_rejections=dict(Counter(r['split'] for r in audit if not r['all_pairs_fit_1024'])),
        limitations=['Length-bounded to 1024; no claims of full-length representativeness.',
                      'Normalized exact prompt grouping only; semantic near-duplicate audit still required.',
                      'Selection is prompt-random, not topic-stratified; generator distribution is reported.'])
    repo = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(args.output / 'jev_behavior_pilot.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for pattern in ['src/sdt_dpo/*.py', 'configs/*.yaml']:
            for path in repo.glob(pattern):
                z.write(path, str(path.relative_to(repo)))
        for name in ['pyproject.toml', 'README.md']:
            z.write(repo/name, name)
        z.writestr('pilot-pairs.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected))
        z.writestr('sample-report.json', json.dumps(report, indent=2))
        z.writestr('sampling-audit.json', json.dumps(audit))
    (args.output/'sample-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'prior_smoke_prompt_ids_excluded'}, indent=2))


if __name__ == '__main__':
    main()
