"""Verify that a full-DPO checkpoint changed a representative model parameter."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any


DEFAULT_PARAMETER = "auto"


def _load_parameter(
    model_path: str,
    parameter_name: str,
    *,
    torch: Any,
    auto_model: Any,
    transformers_version: str,
) -> tuple[Any, dict[str, Any]]:
    dtype_key = "dtype" if int(transformers_version.split(".", 1)[0]) >= 5 else "torch_dtype"
    model = auto_model.from_pretrained(
        model_path,
        **{dtype_key: torch.float32},
        low_cpu_mem_usage=True,
    )
    parameters = dict(model.named_parameters())
    if parameter_name == "auto":
        parameter_name = next(
            (
                name
                for name, parameter in parameters.items()
                if parameter.is_floating_point() and parameter.ndim >= 2
            ),
            "",
        )
    if parameter_name not in parameters:
        preview = sorted(parameters)[:20]
        raise ValueError(
            f"Parameter {parameter_name!r} was not found in {model_path}. "
            f"First available names: {preview}"
        )
    parameter = parameters[parameter_name].detach().cpu().float().clone()
    metadata = {
        "model": model_path,
        "parameter": parameter_name,
        "shape": list(parameter.shape),
        "numel": parameter.numel(),
    }
    del parameters
    del model
    gc.collect()
    return parameter, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--trained-model", required=True)
    parser.add_argument("--parameter", default=DEFAULT_PARAMETER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-identical",
        action="store_true",
        help="Write the report without failing when the parameter is identical.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        __version__ as transformers_version,
    )

    baseline, baseline_metadata = _load_parameter(
        args.baseline_model,
        args.parameter,
        torch=torch,
        auto_model=AutoModelForCausalLM,
        transformers_version=transformers_version,
    )
    trained, trained_metadata = _load_parameter(
        args.trained_model,
        str(baseline_metadata["parameter"]),
        torch=torch,
        auto_model=AutoModelForCausalLM,
        transformers_version=transformers_version,
    )
    if baseline.shape != trained.shape:
        raise ValueError(
            f"Parameter shapes differ: baseline={tuple(baseline.shape)}, "
            f"trained={tuple(trained.shape)}"
        )

    difference = trained - baseline
    absolute = difference.abs()
    baseline_norm = float(torch.linalg.vector_norm(baseline).item())
    difference_norm = float(torch.linalg.vector_norm(difference).item())
    report = {
        "baseline": baseline_metadata,
        "trained": trained_metadata,
        "exactly_equal": bool(torch.equal(baseline, trained)),
        "changed_elements": int(torch.count_nonzero(difference).item()),
        "mean_absolute_change": float(absolute.mean().item()),
        "max_absolute_change": float(absolute.max().item()),
        "l2_change": difference_norm,
        "relative_l2_change": (
            difference_norm / baseline_norm if baseline_norm > 0 else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["exactly_equal"] and not args.allow_identical:
        raise SystemExit(
            "Checkpoint verification failed: the representative parameter is identical"
        )


if __name__ == "__main__":
    main()
