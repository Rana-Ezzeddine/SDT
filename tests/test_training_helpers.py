import unittest
from sdt_dpo.train import _within_length, _resolve_warmup_steps, _precision


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        ids = list(range(len(messages[-1]['content'])))
        return ids if kwargs.get('return_dict') is False else {'input_ids': ids, 'attention_mask': ids}


class TrainingHelperTests(unittest.TestCase):
    def test_t4_amp_keeps_fp32_master_parameters(self):
        from types import SimpleNamespace
        torch = SimpleNamespace(float32='fp32', float16='fp16', bfloat16='bf16',
            cuda=SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (7,5),
                                 is_bf16_supported=lambda: True))
        dtype, bf16, fp16, name = _precision(torch)
        self.assertEqual(dtype, 'fp32')
        self.assertFalse(bf16)
        self.assertTrue(fp16)

    def test_lengths_count_tokens_not_mapping_keys(self):
        row = dict(prompt='p', chosen='x'*20, rejected='y')
        self.assertFalse(_within_length(Tokenizer(), row, 10))
        self.assertTrue(_within_length(Tokenizer(), row, 20))
        row.update(chosen='x', rejected='y'*20)
        self.assertFalse(_within_length(Tokenizer(), row, 10))

    def test_max_steps_controls_warmup(self):
        self.assertEqual(_resolve_warmup_steps({'warmup_ratio': .1, 'max_steps': 10}, 1000), 1)
