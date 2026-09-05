"""Generate fresh responses for one model on unique prompts from a locked split."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.temperature < 0:
        raise ValueError("temperature must be nonnegative")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

    prompts = load_prompts(args.pairs, split=args.split, min_margin=args.min_margin)
    if args.max_prompts is not None:
        prompts = prompts[: max(0, args.max_prompts)]
    if not prompts:
        raise ValueError("max_prompts removed every prompt")

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
    with args.output.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(prompts, start=1):
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
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if args.temperature > 0:
                generation_kwargs["temperature"] = args.temperature
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
                    "seed": args.seed,
                    "precision": precision,
                },
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 10 == 0 or index == len(prompts):
                print(f"Generated {index}/{len(prompts)} prompts")


if __name__ == "__main__":
    main()
