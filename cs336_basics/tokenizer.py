import regex as re
from collections import defaultdict
from typing import Iterable, Generator


def get_stats(ids: list[bytes]):
    count = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        count[pair] += 1
    return count


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab

        self.merges = {merge: i for i, merge in enumerate(merges)}
        self.special_tokens = special_tokens or []

        for special_token in self.special_tokens:
            if special_token.encode("utf-8") not in self.vocab.values():
                self.vocab[len(self.vocab)] = special_token.encode("utf-8")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        self.general_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        if self.special_tokens:
            self.special_token_pattern = re.compile(r"|".join(re.escape(token) for token in sorted_special_tokens))
        else:
            self.special_token_pattern = None

    @classmethod
    def from_file(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = {int(line.split()[0]): bytes(line.split()[1]) for line in f.readlines()}
        with open(merges_filepath, "rb") as f:
            merges = [tuple(line.split()) for line in f.readlines()]
        return cls(vocab, merges, special_tokens)

    def pre_tokenize(self, text: str) -> Generator[str, None, None]:
        #  process special tokens
        last_end = 0

        # Sort special tokens by length in descending order to match longer tokens first
        if self.special_tokens:
            for match in self.special_token_pattern.finditer(text):
                if match.start() > last_end:
                    non_special_chunk = text[last_end : match.start()]
                    for sub_match in self.general_pattern.finditer(non_special_chunk):
                        yield sub_match.group(0)
                yield match.group(0)
                last_end = match.end()
            if last_end < len(text):
                remaining_text = text[last_end:]
                for sub_match in self.general_pattern.finditer(remaining_text):
                    yield sub_match.group(0)
        else:
            for sub_match in self.general_pattern.finditer(text):
                yield sub_match.group(0)

    def encode(self, text: str) -> list[int]:
        final_token_ids = []

        for chunk_str in self.pre_tokenize(text):
            chunk_bytes = chunk_str.encode("utf-8")

            if chunk_str in self.special_tokens:
                final_token_ids.append(self.inverse_vocab[chunk_bytes])
                continue

            if chunk_bytes in self.inverse_vocab:
                final_token_ids.append(self.inverse_vocab[chunk_bytes])
                continue
            
            current_bytes = [bytes([x]) for x in chunk_bytes]

            while len(current_bytes) >= 2:
                stats = get_stats(current_bytes)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                new_bytes = pair[0] + pair[1]
                i = 0
                while i < len(current_bytes) - 1:
                    if current_bytes[i] == pair[0] and current_bytes[i + 1] == pair[1]:
                        current_bytes = current_bytes[:i] + [new_bytes] + current_bytes[i + 2 :]
                        i += 2
                    else:
                        i += 1

            final_token_ids.extend([self.inverse_vocab[x] for x in current_bytes])

        return final_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[token] for token in ids]).decode("utf-8", errors="replace")
