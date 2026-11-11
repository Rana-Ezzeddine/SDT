"""Prevent accidental reuse of a training directory for a different experiment."""
import hashlib
import json


def training_identity(config, pairs_path):
    digest = hashlib.sha256()
    with pairs_path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    settings = {k: v for k, v in config.items()
                if k not in {'output_dir', 'pairs_file', 'resume_from_checkpoint'}}
    return {'pairs_sha256': digest.hexdigest(), 'settings': settings}


def check_run_identity(output_dir, identity):
    path = output_dir / 'training_identity.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('Training data/configuration changed: use a new output_dir, not this checkpoint.')
    elif any(output_dir.iterdir()):
        raise ValueError('Nonempty output_dir has no training identity. Use a new directory; existing files are preserved.')
