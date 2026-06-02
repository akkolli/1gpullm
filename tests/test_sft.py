import unittest

from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel

from one_gpu_lm.sft import IGNORE_INDEX, encode_chat_messages


class ChatSFTTest(unittest.TestCase):
    def test_chat_labels_only_train_assistant_tokens(self):
        tokenizer = toy_tokenizer()
        x, y = encode_chat_messages(
            tokenizer,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            seq_len=8,
        )

        trained_targets = [int(token) for token in y.tolist() if token != IGNORE_INDEX]
        self.assertEqual(trained_targets, [tokenizer.token_to_id("answer"), 0])
        self.assertEqual(tuple(x.shape), (8,))
        self.assertEqual(tuple(y.shape), (8,))


def toy_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(
        WordLevel(
            {
                "<|endoftext|>": 0,
                "<|user|>": 1,
                "<|assistant|>": 2,
                "<|system|>": 3,
                "[UNK]": 4,
                "question": 5,
                "answer": 6,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return tokenizer


if __name__ == "__main__":
    unittest.main()
