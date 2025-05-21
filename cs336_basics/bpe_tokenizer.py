import regex as re
from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size: int, special_tokens: list[str]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.vocab = {}  # dict[int, bytes]
        self.inverse_vocab = {}  # dict[bytes, int]
        self.merges = []  # list[tuple[bytes, bytes]]

    def pre_tokenize(self, file_path: str) -> list[list[str]]:
        with open(file_path) as f:
            data = f.read()

        chunks = re.split(r"<\|endoftext\|>", data)
        pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokenized_chunks = [re.findall(pattern, chunk) for chunk in chunks]
        return pre_tokenized_chunks

    def get_stats(
        self, bytes_tuple_counter: defaultdict[tuple[bytes, bytes], int]
    ) -> defaultdict[tuple[bytes, bytes], int]:
        counter = defaultdict(int)
        for bytes_tuple, count in bytes_tuple_counter.items():
            for i in range(len(bytes_tuple) - 1):
                counter[(bytes_tuple[i], bytes_tuple[i + 1])] += count

        return counter

    def merge_pairs(
        self, bytes_tuple_counter: defaultdict[tuple[bytes, bytes], int], merge_pair: tuple[bytes, bytes]
    ) -> defaultdict[tuple[bytes, bytes], int]:
        counter = defaultdict(int)
        for bytes_tuple, count in bytes_tuple_counter.items():
            bytes_list = []
            i = 0
            while i < len(bytes_tuple):
                if i < len(bytes_tuple) - 1:
                    pair = (bytes_tuple[i], bytes_tuple[i + 1])
                    if pair == merge_pair:
                        bytes_list.append(merge_pair[0] + merge_pair[1])
                        i += 2
                        continue

                bytes_list.append(bytes_tuple[i])
                i += 1
            counter[tuple(bytes_list)] += count

        return counter

    def train(self, file_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # add special tokens
        self.vocab = {i: token.encode("utf-8") for i, token in enumerate(self.special_tokens)}
        self.inverse_vocab = {token.encode("utf-8"): i for i, token in enumerate(self.special_tokens)}

        # add initial 256 characters
        for i in range(256):
            self.vocab[len(self.inverse_vocab)] = bytes([i])
            self.inverse_vocab[bytes([i])] = len(self.vocab)

        pre_tokenized_chunks = self.pre_tokenize(file_path)
        word_counts = defaultdict(int)
        for chunk in pre_tokenized_chunks:
            for word in chunk:
                word_counts[word] += 1

        bytes_tuple_counter = defaultdict(int)
        for word, count in word_counts.items():
            bytes_tuple = tuple(bytes([c]) for c in word.encode("utf-8"))
            bytes_tuple_counter[bytes_tuple] += count

        while len(self.vocab) < self.vocab_size:
            new_token_id = len(self.vocab)
            stats = self.get_stats(bytes_tuple_counter)
            # find the best merge pair
            best_score = (0, bytes([0]), bytes([0]))
            for pair, count in stats.items():
                current_score = (count, pair[0], pair[1])
                if current_score > best_score:
                    best_score = current_score

            freq, merge_pair = best_score[0], (best_score[1], best_score[2])

            bytes_tuple_counter = self.merge_pairs(bytes_tuple_counter, merge_pair)
            new_bytes = merge_pair[0] + merge_pair[1]
            self.vocab[new_token_id] = new_bytes
            self.inverse_vocab[new_bytes] = new_token_id
            self.merges.append(merge_pair)

        return self.vocab, self.merges
