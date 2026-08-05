from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FullDPOConfigurationTests(unittest.TestCase):
    def test_peft_is_not_a_project_dependency(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = project["project"]["dependencies"]
        self.assertFalse(any(item.lower().startswith("peft") for item in dependencies))

    def test_configuration_selects_full_parameter_training(self) -> None:
        config = (ROOT / "configs/full.yaml").read_text(encoding="utf-8")
        self.assertIn("training_method: full_parameter_dpo", config)
        self.assertIn("warmup_steps: 2", config)
        self.assertNotIn("warmup_ratio", config)
        self.assertNotIn("lora:", config.lower())

    def test_trainer_contains_no_lora_or_peft_configuration(self) -> None:
        trainer = (ROOT / "src/sdt_dpo/train.py").read_text(encoding="utf-8")
        self.assertNotIn("LoraConfig", trainer)
        self.assertNotIn("peft_config", trainer)
        self.assertIn("trainable_parameters != total_parameters", trainer)
        self.assertIn(
            'trainer.evaluate(metric_key_prefix="validation")',
            trainer,
        )

    def test_workflow_loads_the_complete_checkpoint(self) -> None:
        script = (ROOT / "scripts/run_sample_pipeline.sh").read_text(encoding="utf-8")
        self.assertIn("--model outputs/dpo-full", script)
        self.assertNotIn("--adapter", script)

    def test_colab_notebook_is_valid_and_has_no_saved_outputs(self) -> None:
        notebook_path = ROOT / "notebooks/SDT_Full_DPO_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 20)
        self.assertTrue(
            all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
        )


if __name__ == "__main__":
    unittest.main()
