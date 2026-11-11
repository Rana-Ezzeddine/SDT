"""Prepare quality-gated Jev pairs without changing the legacy pilot parser."""
import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from .pairs import assign_split, prompt_group_id, validate_no_prompt_leakage, write_jsonl


def probability(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid Jev probability: {value!r}")
    return float(value)


def build(records, threshold=0.8, seed=42):
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between zero and one")
    pairs, seen = [], set()
    for r in records:
        identity = (r['model_name'], str(r['id']))
        if identity in seen:
            raise ValueError(f"Duplicate record identity: {identity}")
        seen.add(identity)
        instruction = r['instruction'].strip()
        extra = r.get('input', '').strip()
        prompt = instruction + (f"\n\nInput:\n{extra}" if extra else '')
        if not instruction or r.get('status') != 'success':
            raise ValueError(f"Invalid record: {identity}")
        responses, ev = r['responses'], r['evaluations']
        sources = ['baseline', 'sdt', 'sdt-augmented']
        scores = {s: probability(ev[s]['sdt_score_average']) for s in sources}
        quality = {s: probability(ev[s]['content_score_average']) for s in sources}
        passing = [s for s in sources if quality[s] >= threshold]
        if passing:
            best = max(scores[s] for s in passing)
            # Stored labels can choose any of the tied maxima; do not invent tie labels.
            if r['preferred_response'] not in passing or scores[r['preferred_response']] != best:
                raise ValueError(f"Stored preference disagrees with gate/SDT rule: {identity}")
        for a, b in itertools.combinations(sources, 2):
            pa, pb = a in passing, b in passing
            reasons = []
            basis = 'content_gate' if pa != pb else 'sdt_score'
            if not pa and not pb:
                reasons.append('both_failed_content')
            if not responses[a].strip() or not responses[b].strip():
                reasons.append('empty_response')
            if responses[a].strip() == responses[b].strip():
                reasons.append('identical_responses')
            if pa == pb and scores[a] == scores[b]:
                reasons.append('sdt_tie')
            chosen, rejected = (a, b) if (pa if pa != pb else scores[a] > scores[b]) else (b, a)
            key = json.dumps([*identity, prompt, a, b], ensure_ascii=False)
            pairs.append(dict(
                pair_id=hashlib.sha256(key.encode()).hexdigest()[:20],
                record_id=str(r['id']), generator_model=r['model_name'],
                prompt_id=prompt_group_id(prompt), split=assign_split(prompt, seed=seed),
                prompt=prompt, chosen=responses[chosen], rejected=responses[rejected],
                chosen_source=chosen, rejected_source=rejected,
                comparison_type='__vs__'.join(sorted((a,b))),
                label_mode='jev', label_confidence='jev_single_judge',
                confidence_score=None, pair_weight=1.0,
                preference_basis=basis, sdt_margin=abs(scores[a]-scores[b]),
                label_margin=abs(scores[a]-scores[b]),
                chosen_sdt_score=scores[chosen], rejected_sdt_score=scores[rejected],
                chosen_content_score=quality[chosen], rejected_content_score=quality[rejected],
                content_threshold=threshold, stored_preferred_response=r['preferred_response'],
                retain=not reasons, exclusion_reasons=reasons))
    validate_no_prompt_leakage(pairs)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--content-threshold', type=float, default=.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    records = []
    hashes = {}
    for path in args.input:
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        records.extend(json.loads(raw))
    pairs = build(records, args.content_threshold, args.seed)
    kept = [p for p in pairs if p['retain']]
    report = dict(records=len(records), possible_pairs=len(pairs), retained_pairs=len(kept),
                  inputs_sha256=hashes, seed=args.seed, content_threshold=args.content_threshold,
                  split_policy='80/10/10 deterministic normalized full prompt; no near-duplicate clustering',
                  retained_by_split=dict(Counter(p['split'] for p in kept)),
                  retained_by_source=dict(Counter(p['generator_model'] for p in kept)),
                  retained_by_basis=dict(Counter(p['preference_basis'] for p in kept)),
                  exclusions=dict(Counter(reason for p in pairs for reason in p['exclusion_reasons'])))
    write_jsonl(args.output, pairs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
