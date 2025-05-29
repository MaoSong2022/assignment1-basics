import regex as re
from collections import defaultdict
from typing import Iterable


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

    @classmethod
    def from_file(
        cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        with open(vocab_filepath, "rb") as f:
            vocab = {int(line.split()[0]): bytes(line.split()[1]) for line in f.readlines()}
        with open(merges_filepath, "rb") as f:
            merges = [tuple(line.split()) for line in f.readlines()]
        return cls(vocab, merges, special_tokens)

    def pre_tokenize(self, text: str) -> list[str]:
        #  process special tokens
        chunks = []
        last_end = 0

        # Sort special tokens by length in descending order to match longer tokens first
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            special_token_pattern = re.compile(r"|".join(re.escape(token) for token in sorted_special_tokens))
            for match in special_token_pattern.finditer(text):
                if match.start() > last_end:
                    chunks.append(text[last_end : match.start()])
                chunks.append(match.group(0))
                last_end = match.end()
            if last_end < len(text):
                chunks.append(text[last_end:])
        else:
            chunks = [text]

        result = []
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for chunk in chunks:
            if chunk in self.special_tokens:
                result.append(chunk)
            else:
                pre_tokenized_chunks = re.findall(pattern, chunk)
                result.extend(pre_tokenized_chunks)
        return result

    def encode(self, text: str) -> list[int]:
        pre_tokenized_chunks = self.pre_tokenize(text)
        sequence_bytes = []
        for chunk in pre_tokenized_chunks:
            chunk_bytes = chunk.encode("utf-8")
            if chunk in self.special_tokens:
                sequence_bytes.append(chunk_bytes)
                continue

            if chunk_bytes in self.inverse_vocab:
                sequence_bytes.append(chunk_bytes)
                continue
            split = [bytes([x]) for x in chunk_bytes]

            while len(split) >= 2:
                stats = get_stats(split)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                new_bytes = pair[0] + pair[1]
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [new_bytes] + split[i + 2 :]
                        i += 2
                    else:
                        i += 1

            sequence_bytes.extend(split)

        return [self.inverse_vocab[x] for x in sequence_bytes]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[token] for token in ids]).decode("utf-8", errors="replace")
