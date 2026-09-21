"""Generate fresh responses for one model on unique prompts from a locked split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .evaluate import _device_and_dtype


def load_prompts(
    pairs_path: Path,
    *,
    split: str,
    min_margin: float,
) -> list[dict[str, str]]:
    prompts: dict[str, str] = {}
    with pairs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("retain") or row.get("split") != split:
                continue
            if float(row.get("label_margin", 0.0)) < min_margin:
                continue
            prompt_id = str(row["prompt_id"])
            prompt = str(row["prompt"])
            if prompt_id in prompts and prompts[prompt_id] != prompt:
                raise ValueError(f"Prompt text differs within prompt_id {prompt_id}")
            prompts[prompt_id] = prompt
    if not prompts:
        raise ValueError("No retained prompts matched the requested split and margin")
    return [
        {"prompt_id": prompt_id, "prompt": prompts[prompt_id]}
        for prompt_id in sorted(prompts)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--min-margin", type=float, default=0.10)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-input-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature < 0:
        raise ValueError("temperature must be nonnegative")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

    prompts = load_prompts(args.pairs, split=args.split, min_margin=args.min_margin)
    if args.max_prompts is not None:
        prompts = prompts[: max(0, args.max_prompts)]
    if not prompts:
        raise ValueError("max_prompts removed every prompt")

    expected_generation = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    completed: dict[str, dict[str, Any]] = {}
    if args.output.exists():
        with args.output.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                prompt_id = str(row["prompt_id"])
                if prompt_id in completed:
                    raise ValueError(f"Duplicate prompt_id {prompt_id} in {args.output}")
                if str(row.get("model")) != args.model:
                    raise ValueError(
                        "Existing generation output used a different model; use a new output file"
                    )
                recorded = row.get("generation", {})
                for key, value in expected_generation.items():
                    recorded_value = (
                        recorded.get(key, 1.0) if key == "top_p" else recorded.get(key)
                    )
                    if recorded_value != value:
                        raise ValueError(
                            "Existing generation output used different decoding settings; "
                            "use a new output file"
                        )
                completed[prompt_id] = row

    prompt_lookup = {item["prompt_id"]: item["prompt"] for item in prompts}
    unexpected = set(completed) - set(prompt_lookup)
    if unexpected:
        raise ValueError(
            f"Existing generation output contains unexpected prompt IDs: {sorted(unexpected)[:5]}"
        )
    for prompt_id, row in completed.items():
        if str(row.get("prompt")) != prompt_lookup[prompt_id]:
            raise ValueError(f"Prompt text changed for prompt_id {prompt_id}")
    if len(completed) == len(prompts):
        print(f"Generation output is already complete: {len(completed)}/{len(prompts)} prompts")
        return

    torch.manual_seed(args.seed)
    device, dtype, precision = _device_and_dtype(torch)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype_key = "dtype" if int(__version__.split(".", 1)[0]) >= 5 else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(args.model, **{dtype_key: dtype})
    model.to(device)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        for index, item in enumerate(prompts, start=1):
            if item["prompt_id"] in completed:
                continue
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": item["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer(
                rendered,
                add_special_tokens=False,
                return_tensors="pt",
            )
            input_length = int(encoded["input_ids"].shape[-1])
            if input_length > args.max_input_length:
                raise ValueError(
                    f"Prompt {item['prompt_id']} has {input_length} tokens, above "
                    f"max_input_length={args.max_input_length}"
                )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_seed = int.from_bytes(
                hashlib.sha256(
                    f"{args.seed}::{item['prompt_id']}".encode("utf-8")
                ).digest()[:8],
                "big",
            ) % (2**31)
            torch.manual_seed(prompt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(prompt_seed)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if args.temperature > 0:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
            with torch.inference_mode():
                generated = model.generate(**encoded, **generation_kwargs)
            completion_ids = generated[0, input_length:]
            response = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            row = {
                **item,
                "model": args.model,
                "response": response,
                "input_tokens": input_length,
                "generated_tokens": int(completion_ids.numel()),
                "generation": {
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "seed": args.seed,
                    "prompt_seed": prompt_seed,
                    "precision": precision,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(prompts):
                print(f"Generated {index}/{len(prompts)} prompts")


if __name__ == "__main__":
    main()
