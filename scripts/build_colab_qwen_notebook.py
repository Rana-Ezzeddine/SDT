"""Build the concise Colab runner for the packaged Jev sample and current code."""
import json
from pathlib import Path


def main():
    cells = []
    def md(s):
        cells.append(dict(cell_type='markdown', metadata={}, source=s.splitlines(True)))
    def code(s):
        cells.append(dict(cell_type='code', metadata={}, source=s.splitlines(True), execution_count=None, outputs=[]))
    md('# Qwen 0.5B · full DPO · real Jev sample\nDevelopment test, not a final benchmark. T4 runtime. Separate prompts: 64 train / 16 validation / 16 test. Short examples (≤512 tokens). No hyperparameter search.')
    code('''import torch, subprocess, sys
assert torch.cuda.is_available(), "A GPU runtime must connect before proceeding."
print(torch.cuda.get_device_name(0))
subprocess.run(["nvidia-smi"], check=True)
''')
    md('## 1. Upload the prepared code + data bundle')
    code('''from google.colab import files
from pathlib import Path
import zipfile, os, json
uploaded = files.upload()  # select sdt_colab_qwen_bundle.zip
bundle = Path("/content/sdt_colab_qwen_bundle.zip")
assert len(uploaded) == 1, "Upload only the prepared bundle."
bundle.write_bytes(next(iter(uploaded.values())))
WORK = Path("/content/sdt-qwen-test")
assert not WORK.exists(), "Use a fresh runtime to avoid mixing experiments."
with zipfile.ZipFile(bundle) as z:
    assert all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in z.namelist())
    z.extractall(WORK)
os.chdir(WORK)
os.environ["PYTHONPATH"] = str(WORK / "src")
sys.path.insert(0, str(WORK / "src"))
print((WORK / "sample-report.json").read_text())
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.5.3", "trl==1.17.1", "accelerate==1.15.0", "datasets==4.8.5", "-e", "."], check=True)
subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"], check=True)
''')
    md('## 2. Mount Drive and create a NEW run folder\nApprove the Google authorization yourself. Do not train until this succeeds.')
    code('''from google.colab import drive
from datetime import datetime, timezone
import shutil, yaml, hashlib
drive.mount("/content/drive")
RUN = Path("/content/drive/MyDrive/SDT_JEV_Colab_v1") / datetime.now(timezone.utc).strftime("qwen05-full-%Y%m%dT%H%M%SZ")
RUN.mkdir(parents=True, exist_ok=False)
shutil.copy2(bundle, RUN / bundle.name)
shutil.copy2(WORK / "sample-report.json", RUN / "sample-report.json")
shutil.copy2(WORK / "sample-pairs.jsonl", RUN / "sample-pairs.jsonl")
PAIRS = RUN / "sample-pairs.jsonl"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TRAINED = RUN / "trained"
cfg = yaml.safe_load((WORK / "configs/jev_phi4_dpo.yaml").read_text())
cfg.update(model_id=MODEL, pairs_file=str(PAIRS), output_dir=str(TRAINED), max_length=512,
           gradient_accumulation_steps=8, eval_strategy="epoch", save_strategy="epoch", logging_steps=1)
CONFIG = RUN / "config.yaml"
CONFIG.write_text(yaml.safe_dump(cfg))
print("Keep this run path:", RUN)
def run(name, module, *args):
    with (RUN / (name + ".log")).open("a") as log:
        p = subprocess.Popen([sys.executable, "-u", "-m", module, *map(str, args)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        assert p.wait() == 0, f"{name} failed; inspect its saved log."
''')
    md('## 3. Train once; save weights and resumable checkpoint to Drive')
    code('''run("training", "sdt_dpo.train", "--config", CONFIG)
assert (TRAINED / "model.safetensors").exists()
run("checkpoint-change", "sdt_dpo.verify_checkpoint", "--baseline-model", MODEL,
    "--trained-model", TRAINED, "--output", RUN / "checkpoint-change.json")
print((TRAINED / "validation_metrics.json").read_text())
''')
    md('## 4. Frozen test: same stored pairs for baseline and trained model')
    code('''for name, model in [("baseline", MODEL), ("dpo", TRAINED)]:
    run("evaluate-" + name, "sdt_dpo.evaluate", "--pairs", PAIRS, "--model", model,
        "--max-length", "512", "--output", RUN / (name + "-summary.json"),
        "--details", RUN / (name + "-pairs.jsonl"))
run("comparison", "sdt_dpo.compare", "--baseline-details", RUN / "baseline-pairs.jsonl",
    "--dpo-details", RUN / "dpo-pairs.jsonl", "--output", RUN / "comparison.json")
print((RUN / "comparison.json").read_text())
''')
    md('## 5. Verify generation from saved weights and inventory\nThese two responses per model are an inference check, not a judged SDT evaluation.')
    code('''for name, model in [("baseline", MODEL), ("dpo", TRAINED)]:
    run("generate-" + name, "sdt_dpo.generate", "--pairs", PAIRS, "--model", model,
        "--max-prompts", "2", "--max-new-tokens", "64", "--output", RUN / (name + "-generations.jsonl"))
def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
weights = list(TRAINED.glob("*.safetensors"))
assert weights
inventory = {str(p.relative_to(RUN)): {"bytes":p.stat().st_size, "sha256":sha256(p)} for p in weights}
(RUN / "completion.json").write_text(json.dumps({"status":"PASS", "weights":inventory, "reload":"separate evaluation/generation processes"}, indent=2))
print("Saved run:", RUN)
print(json.dumps(inventory, indent=2))
''')
    md('## 6. Flush Drive and verify the weights again after remounting\nKeep the runtime connected until this check finishes. Approve remount authorization if requested.')
    code('''drive.flush_and_unmount()
drive.mount("/content/drive")
for relative, expected in inventory.items():
    path = RUN / relative
    assert path.exists() and path.stat().st_size == expected["bytes"], relative
    assert sha256(path) == expected["sha256"], relative
print("Drive remount verification passed:", RUN)
''')
    out = Path(__file__).resolve().parents[1] / 'notebooks/SDT_Jev_Qwen_Full_DPO_Colab.ipynb'
    out.write_text(json.dumps(dict(cells=cells, metadata=dict(accelerator='GPU',
        kernelspec=dict(display_name='Python 3', name='python3'), colab=dict(provenance=[])),
        nbformat=4, nbformat_minor=5), indent=1)+'\n')
    print(out)


if __name__ == '__main__':
    main()
