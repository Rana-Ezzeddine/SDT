"""Offline, real DPO integration test. Synthetic data/results are NOT research evidence.

Run with the project's installed Python environment. No pretrained download required.
Creates a fresh outputs/smoke-jev-* folder, preserves every log and checkpoint.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
from sdt_dpo.jev_pairs import build
from sdt_dpo.pairs import write_jsonl


def main():
    repo = Path(__file__).resolve().parents[1]
    (repo / 'outputs').mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='smoke-jev-', dir=repo / 'outputs'))
    print(f'Artifacts: {root}', flush=True)
    (root / 'environment-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1', PYTHONPATH=str(repo / 'src'))

    def run(name, module, *args):
        with (root / f'{name}.log').open('w') as log:
            result = subprocess.run([sys.executable, '-m', module, *map(str, args)],
                                    cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print((root / f'{name}.log').read_text()[-12000:])
            raise RuntimeError(f'{name} failed; inspect {root / (name + ".log")}')
        print(f'PASS: {name}', flush=True)

    torch.manual_seed(42)
    vocab = {word: i for i, word in enumerate(
        ['[UNK]', '[PAD]', '[EOS]', '<user>', '<assistant>',
         'Explain', 'topic', 'helpful', 'clear', 'poor', 'answer'] + [str(i) for i in range(100)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token='[UNK]'))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]',
                                       pad_token='[PAD]', eos_token='[EOS]')
    tokenizer.chat_template = "{% for m in messages %}{{ '<' + m['role'] + '> ' + m['content'] + ' ' }}{% if m['role'] == 'assistant' %}{{ eos_token + ' ' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    base = root / 'baseline'
    tokenizer.save_pretrained(base)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=32,
        intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, max_position_embeddings=128,
        eos_token_id=2, pad_token_id=1, attention_dropout=0.0))
    model.save_pretrained(base)
    records = [dict(id=i, model_name='synthetic', instruction=f'Explain topic {i}', input='',
        status='success', preferred_response='sdt-augmented',
        responses={'baseline':'poor answer', 'sdt':'clear answer', 'sdt-augmented':'helpful clear answer'},
        evaluations={s:dict(content_score_average=.9, sdt_score_average=score)
            for s, score in [('baseline', .2), ('sdt', .5), ('sdt-augmented', .8)]}) for i in range(100)]
    pairs = build(records)
    selected = []
    for split, count in [('train', 8), ('validation', 4), ('test', 4)]:
        selected.extend([p for p in pairs if p['split'] == split][:count])
    pair_file = root / 'pairs.jsonl'
    # This training row must be excluded, not silently truncated by TRL.
    overlong = dict(next(p for p in selected if p['split'] == 'train'))
    overlong.update(pair_id='overlength-canary', chosen='helpful ' * 100)
    selected.append(overlong)
    write_jsonl(pair_file, selected)
    cfg = yaml.safe_load((repo / 'configs/jev_phi4_dpo.yaml').read_text())
    cfg.update(model_id=str(base), pairs_file=str(pair_file), output_dir=str(root / 'trained'),
        use_cpu=True, learning_rate=.001, max_steps=4, gradient_accumulation_steps=1,
        max_length=64, eval_steps=2, save_steps=2, logging_steps=1, warmup_ratio=0.0)
    config_file = root / 'config.yaml'
    config_file.write_text(yaml.safe_dump(cfg))
    run('training', 'sdt_dpo.train', '--config', config_file)
    trained = root / 'trained'
    manifest = json.loads((trained / 'run_manifest.json').read_text())
    assert manifest['dropped_for_length']['train'] == 1
    assert manifest['eligible_pairs']['train'] == 8
    history = json.loads((trained / 'trainer_log_history.json').read_text())
    losses = [float(row['loss']) for row in history if 'loss' in row]
    assert len(losses) == 4 and all(math.isfinite(v) for v in losses)
    run('weight-change', 'sdt_dpo.verify_checkpoint', '--baseline-model', base,
        '--trained-model', trained, '--output', root / 'weight-change.json')

    # Resume from a completed intermediate checkpoint into a separate directory.
    resumed = root / 'resumed'
    resumed.mkdir()
    shutil.copytree(trained / 'checkpoint-2', resumed / 'checkpoint-2')
    shutil.copy2(trained / 'training_identity.json', resumed / 'training_identity.json')
    cfg['output_dir'] = str(resumed)
    config_file.write_text(yaml.safe_dump(cfg))
    run('resume', 'sdt_dpo.train', '--config', config_file)
    resumed_state = json.loads((resumed / 'checkpoint-4/trainer_state.json').read_text())
    assert resumed_state['global_step'] == 4
    # Compare last-step weights too: selected best can be the earlier checkpoint.
    last_original = AutoModelForCausalLM.from_pretrained(trained / 'checkpoint-4')
    last_resumed = AutoModelForCausalLM.from_pretrained(resumed / 'checkpoint-4')
    resume_last_diff = max((v - last_resumed.state_dict()[k]).abs().max().item()
                          for k, v in last_original.state_dict().items())
    assert resume_last_diff < 1e-6, f'Resumed final step differs: {resume_last_diff}'
    original = AutoModelForCausalLM.from_pretrained(trained).eval()
    restored = AutoModelForCausalLM.from_pretrained(resumed).eval()
    max_diff = max((v - restored.state_dict()[k]).abs().max().item()
                   for k, v in original.state_dict().items())
    assert max_diff < 1e-6, f'Resumed weights differ: {max_diff}'

    # Independent durable-copy check, including hashes and real inference reload.
    backup = root / 'verified-backup'
    shutil.copytree(trained, backup, ignore=shutil.ignore_patterns('checkpoint-*'))
    hashes = {}
    for f in trained.glob('*.safetensors'):
        hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
        assert hashes[f.name] == hashlib.sha256((backup / f.name).read_bytes()).hexdigest()
    assert hashes
    reloaded = AutoModelForCausalLM.from_pretrained(backup).eval()
    inputs = AutoTokenizer.from_pretrained(backup).apply_chat_template(
        [{'role':'user', 'content':'Explain topic 1'}], add_generation_prompt=True,
        return_tensors='pt', return_dict=True)
    with torch.inference_mode():
        expected = original.generate(**inputs, max_new_tokens=4, do_sample=False)
        actual = reloaded.generate(**inputs, max_new_tokens=4, do_sample=False)
    assert torch.equal(expected, actual), 'Backup inference differs'
    for name, path in [('baseline', base), ('dpo', trained), ('backup', backup)]:
        run(f'eval-{name}', 'sdt_dpo.evaluate', '--pairs', pair_file, '--model', path,
            '--max-length', 64, '--output', root / f'{name}-summary.json',
            '--details', root / f'{name}-pairs.jsonl')
    assert (root / 'dpo-pairs.jsonl').read_bytes() == (root / 'backup-pairs.jsonl').read_bytes()
    run('comparison', 'sdt_dpo.compare', '--baseline-details', root / 'baseline-pairs.jsonl',
        '--dpo-details', root / 'dpo-pairs.jsonl', '--output', root / 'comparison.json',
        '--bootstrap-samples', 100)
    for name, path in [('baseline', base), ('dpo', backup)]:
        run(f'generate-{name}', 'sdt_dpo.generate', '--pairs', pair_file, '--model', path,
            '--output', root / f'{name}-generations.jsonl', '--max-new-tokens', 4)
    generated = [[json.loads(line) for line in (root / f'{name}-generations.jsonl').read_text().splitlines()]
                 for name in ('baseline', 'dpo')]
    assert {r['prompt_id'] for r in generated[0]} == {r['prompt_id'] for r in generated[1]}
    summary = dict(status='PASS', synthetic_only=True, optimizer_steps=4,
        overlength_training_pair_excluded=True,
        finite_training_losses=losses, paired_generation_prompts=len(generated[0]),
        resume_from_step=2, resume_max_weight_difference=max_diff,
        resume_last_step_max_weight_difference=resume_last_diff,
        backup_weight_sha256=hashes, backup_generation_exact_match=True,
        backup_evaluation_exact_match=True, artifacts=str(root))
    (root / 'smoke-report.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
