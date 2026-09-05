from __future__ import annotations

import json
import tomllib
import unittest
from pathlib import Path

from sdt_dpo.train import _resolve_warmup_steps


ROOT = Path(__file__).resolve().parents[1]


class FullDPOConfigurationTests(unittest.TestCase):
    def test_peft_is_not_a_project_dependency(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = project["project"]["dependencies"]
        self.assertFalse(any(item.lower().startswith("peft") for item in dependencies))
        self.assertEqual(
            project["project"]["scripts"]["sdt-verify-checkpoint"],
            "sdt_dpo.verify_checkpoint:main",
        )

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

    def test_overfit_canary_is_explicitly_separate(self) -> None:
        config = (ROOT / "configs/sanity_overfit.yaml").read_text(encoding="utf-8")
        self.assertIn("max_train_samples: 8", config)
        self.assertIn("sanity_use_train_as_validation: true", config)
        self.assertIn("output_dir: outputs/dpo-overfit-sanity", config)

    def test_1500_pilot_is_explicitly_single_judge_and_full_parameter(self) -> None:
        config = (ROOT / "configs/pilot_1500_lfm.yaml").read_text(encoding="utf-8")
        self.assertIn("model_id: LiquidAI/LFM2.5-1.2B-Instruct", config)
        self.assertIn("training_method: full_parameter_dpo", config)
        self.assertIn("label_mode: single_judge_pilot", config)
        self.assertIn("filter_by_confidence: false", config)
        self.assertIn("warmup_ratio: 0.03", config)

    def test_warmup_ratio_is_resolved_without_passing_it_to_dpo_config(self) -> None:
        config = {
            "warmup_ratio": 0.03,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
        }
        self.assertEqual(_resolve_warmup_steps(config, 3051, world_size=1), 12)
        trainer = (ROOT / "src/sdt_dpo/train.py").read_text(encoding="utf-8")
        dpo_config_call = trainer.split("training_args = DPOConfig(", 1)[1].split(
            "\n    )", 1
        )[0]
        self.assertNotIn("warmup_ratio=", dpo_config_call)

    def test_explicit_warmup_steps_take_precedence(self) -> None:
        config = {"warmup_steps": 5, "warmup_ratio": 0.50}
        self.assertEqual(_resolve_warmup_steps(config, 3051, world_size=1), 5)

    def test_pilot_notebook_is_concise_and_has_no_saved_outputs(self) -> None:
        notebook_path = ROOT / "notebooks/SDT_1500_DPO_Pilot_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertLessEqual(len(notebook["cells"]), 24)
        self.assertTrue(
            all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
        )

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
