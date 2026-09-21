"""Blindly compare baseline and DPO generations with a local Hugging Face model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from jinja2.exceptions import TemplateError
from transformers import AutoModelForCausalLM, AutoTokenizer

from .judge_generations import (
    JUDGE_PROMPT_VERSION,
    _dpo_is_a,
    _extract_json,
    _judge_prompt,
    _normalized_response,
    _read_generations,
    _sha256,
    _validate_judgment,
    summarize,
)


def _local_judge_config_fingerprint(
    *,
    baseline_path: Path,
    dpo_path: Path,
    judge_model: str,
    seed: int,
    max_new_tokens: int,
    reverse_order: bool,
) -> tuple[str, dict[str, Any]]:
    manifest = {
        "backend": "local_transformers",
        "prompt_version": JUDGE_PROMPT_VERSION,
        "baseline_sha256": _sha256(baseline_path),
        "dpo_sha256": _sha256(dpo_path),
        "judge_model": judge_model,
        "order_randomization_seed": seed,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "reverse_order": reverse_order,
    }
    encoded = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), manifest


def _generate_judgment(
    *,
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    previous_response: str = "",
    previous_error: str = "",
) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful, impartial evaluator. Return valid JSON only.",
        },
        {"role": "user", "content": prompt},
    ]
    if previous_response:
        messages.extend(
            [
                {"role": "assistant", "content": previous_response},
                {
                    "role": "user",
                    "content": (
                        "That response was invalid because: "
                        f"{previous_error}. Return only one valid JSON object matching "
                        "the requested schema; do not use Markdown fences."
                    ),
                },
            ]
        )
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (ValueError, TypeError, TemplateError):
        # Some instruct templates (notably older Mistral templates) do not
        # accept a separate system role. Preserve the instruction by folding it
        # into the first user message instead of dropping it.
        fallback = [
            {
                "role": "user",
                "content": (
                    "You are a careful, impartial evaluator. Return valid JSON only.\n\n"
                    + prompt
                ),
            }
        ]
        if previous_response:
            fallback.extend(messages[2:])
        rendered = tokenizer.apply_chat_template(
            fallback, tokenize=False, add_generation_prompt=True
        )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            eos_token_id=tokenizer.eos_token_id,
        )
    completion = generated[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--dpo", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--reverse-order",
        action="store_true",
        help="Invert the deterministic A/B assignment to measure position consistency.",
    )
    parser.add_argument(
        "--failures", type=Path, help="Optional JSONL log of failed parsing attempts."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if args.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")

    baseline = _read_generations(args.baseline)
    dpo = _read_generations(args.dpo)
    if not baseline or set(baseline) != set(dpo):
        raise ValueError(
            "Baseline and DPO generation files must contain exactly the same nonempty prompt IDs"
        )

    config_fingerprint, judge_manifest = _local_judge_config_fingerprint(
        baseline_path=args.baseline,
        dpo_path=args.dpo,
        judge_model=args.judge_model,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        reverse_order=args.reverse_order,
    )

    completed: dict[str, dict[str, Any]] = {}
    if args.details.exists():
        with args.details.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("judge_config_fingerprint") != config_fingerprint:
                    raise ValueError(
                        "Existing judgment details use a different local judge "
                        "configuration or generation input. Use a new details file."
                    )
                completed[str(row["prompt_id"])] = row

    remaining_nonidentical = any(
        prompt_id not in completed
        and _normalized_response(baseline[prompt_id]["response"])
        != _normalized_response(dpo[prompt_id]["response"])
        for prompt_id in baseline
    )
    model = tokenizer = None
    if remaining_nonidentical:
        tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.judge_model,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

    args.details.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with args.details.open(mode, encoding="utf-8") as handle:
        for index, prompt_id in enumerate(sorted(baseline), start=1):
            if prompt_id in completed:
                continue
            before = baseline[prompt_id]
            after = dpo[prompt_id]
            if str(before["prompt"]) != str(after["prompt"]):
                raise ValueError(f"Prompt text differs for prompt_id {prompt_id}")
            if _normalized_response(before["response"]) == _normalized_response(
                after["response"]
            ):
                row = {
                    "prompt_id": prompt_id,
                    "prompt": before["prompt"],
                    "winner": "tie",
                    "judgment_type": "deterministic_identical_response",
                    "reason": "Baseline and DPO responses are identical after whitespace normalization.",
                    "judge_model": None,
                    "judge_config_fingerprint": config_fingerprint,
                    "prompt_version": JUDGE_PROMPT_VERSION,
                    "reverse_order": args.reverse_order,
                }
                completed[prompt_id] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"Recorded deterministic tie {index}/{len(baseline)} prompts")
                continue

            assert model is not None and tokenizer is not None
            dpo_is_a = _dpo_is_a(prompt_id, args.seed) ^ args.reverse_order
            response_a = after["response"] if dpo_is_a else before["response"]
            response_b = before["response"] if dpo_is_a else after["response"]
            judge_prompt = _judge_prompt(str(before["prompt"]), response_a, response_b)
            raw = ""
            previous_error = ""
            judged: dict[str, Any] | None = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    raw = _generate_judgment(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=judge_prompt,
                        max_new_tokens=args.max_new_tokens,
                        previous_response=raw,
                        previous_error=previous_error,
                    )
                    judged = _validate_judgment(_extract_json(raw))
                    break
                except Exception as error:
                    previous_error = f"{type(error).__name__}: {error}"
                    failure = {
                        "prompt_id": prompt_id,
                        "attempt": attempt,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "raw_judge_response": raw,
                        "judge_config_fingerprint": config_fingerprint,
                    }
                    if args.failures:
                        args.failures.parent.mkdir(parents=True, exist_ok=True)
                        with args.failures.open("a", encoding="utf-8") as failure_handle:
                            failure_handle.write(
                                json.dumps(failure, ensure_ascii=False) + "\n"
                            )
                    if attempt == args.max_retries:
                        raise RuntimeError(
                            f"Local judge failed for prompt {prompt_id} after "
                            f"{args.max_retries} attempts"
                        ) from error
            assert judged is not None
            winner = (
                "tie"
                if judged["winner"] == "tie"
                else "dpo"
                if (judged["winner"] == "A") == dpo_is_a
                else "baseline"
            )
            row = {
                "prompt_id": prompt_id,
                "prompt": before["prompt"],
                "winner": winner,
                "dpo_was_response": "A" if dpo_is_a else "B",
                "baseline_scores": judged["scores"]["B" if dpo_is_a else "A"],
                "dpo_scores": judged["scores"]["A" if dpo_is_a else "B"],
                "reason": judged["reason"],
                "judgment_type": "local_llm_judge",
                "judge_model": args.judge_model,
                "judge_config_fingerprint": config_fingerprint,
                "prompt_version": JUDGE_PROMPT_VERSION,
                "reverse_order": args.reverse_order,
                "raw_judge_response": raw,
            }
            completed[prompt_id] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"Judged {index}/{len(baseline)} prompts")

    report = summarize([completed[prompt_id] for prompt_id in sorted(completed)])
    report["locally_judged_n"] = report.pop("api_judged_n")
    report["judge_backend"] = "local_transformers"
    report["judge_model"] = args.judge_model
    report["order_randomization_seed"] = args.seed
    report["reverse_order"] = args.reverse_order
    report["judge_config_fingerprint"] = config_fingerprint
    report["judge_manifest"] = judge_manifest
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
