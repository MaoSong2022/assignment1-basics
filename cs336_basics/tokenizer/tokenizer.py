import json
import regex as re
from collections.abc import Iterable, Iterator


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}  # dict[bytes, int]
        self.merges = merges
        self.merge_rank = {merge: i for i, merge in enumerate(merges)}  # dict[tuple[bytes, bytes], int]

        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens.sort(key=lambda x: len(x), reverse=True)  # sort from longest to shortest

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath) as f:
            vocab = json.load(f)

        merges = []
        with open(merges_filepath) as f:
            for line in f:
                line = line.strip()
                parts = line.split()
                merges.append((parts[0], parts[1]))

        return cls(vocab, merges, special_tokens)

    def pre_tokenize(self, text: str) -> list[bytes]:
        GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens:
            pattern = f"({'|'.join(re.escape(tok) for tok in self.special_tokens)})"
            chunks = re.split(pattern, text)
            words = []
            for chunk in chunks:
                # empty
                if not chunk:
                    continue
                # special token
                if chunk in self.special_tokens:
                    words.append(chunk)
                else:  # normal text
                    words.extend([match.group(0).encode("utf-8") for match in re.finditer(GPT2_PAT, chunk)])
        else:  # normal text
            words = [match.group(0).encode("utf-8") for match in re.finditer(GPT2_PAT, text)]

        return words

    def encode(self, text: str) -> list[int]:
        words = self.pre_tokenize(text)
        all_token_ids = []

        for word in words:
            # special token
            if isinstance(word, str) and word in self.special_tokens:
                all_token_ids.append(self.inverse_vocab[word.encode("utf-8")])
                continue

            token_ids = [self.inverse_vocab[bytes([b])] for b in word]
            while len(token_ids) >= 2:
                pairs = list(zip(token_ids, token_ids[1:]))

                # find the merge pair
                min_rank = float("inf")
                min_pair = None
                for pair in pairs:
                    # 转换为bytes对，匹配merge_rank的key
                    pair_bytes = (self.vocab[pair[0]], self.vocab[pair[1]])
                    rank = self.merge_rank.get(pair_bytes, float("inf"))
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair
                if min_pair is None or min_rank == float("inf"):
                    break

                new_token_id = self.inverse_vocab[self.vocab[min_pair[0]] + self.vocab[min_pair[1]]]

                # merge pair
                i = 0
                new_token_ids = []
                while i < len(token_ids):
                    if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == min_pair:
                        new_token_ids.append(new_token_id)
                        i += 2
                    else:
                        new_token_ids.append(token_ids[i])
                        i += 1

                # update word
                token_ids = new_token_ids

            all_token_ids.extend(token_ids)

        return all_token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[token_id] for token_id in ids]).decode("utf-8", errors="replace")
