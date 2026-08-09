from __future__ import annotations

import unittest

from sdt_dpo.evaluate import _sequence


class CharacterTokenizer:
    """Minimal fast tokenizer that exposes one token and offset per character."""

    is_fast = True

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize:
            raise AssertionError("The evaluator should render before tokenizing")
        user = messages[0]["content"]
        assistant = messages[1]["content"]
        return f"<user>{user}</user><assistant>{assistant}</assistant>"

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        self.last_rendered = text
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class ResponseBoundaryTests(unittest.TestCase):
    def test_response_boundary_does_not_require_generation_prefix_match(self) -> None:
        tokenizer = CharacterTokenizer()
        response = "same phrase; same phrase"
        token_ids, response_start = _sequence(
            tokenizer,
            prompt="The prompt also says same phrase.",
            response=response,
        )
        rendered = "".join(chr(token_id) for token_id in token_ids)
        self.assertEqual(rendered, tokenizer.last_rendered)
        self.assertTrue(rendered[response_start:].startswith(response))
        self.assertEqual(rendered[response_start - 1], ">")

    def test_empty_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty_response"):
            _sequence(CharacterTokenizer(), prompt="prompt", response="")


if __name__ == "__main__":
    unittest.main()
