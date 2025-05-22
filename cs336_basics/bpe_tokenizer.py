import regex as re
from collections import defaultdict


def pre_tokenize(file_path: str) -> list[list[str]]:
    with open(file_path) as f:
        data = f.read()

    chunks = re.split(r"<\|endoftext\|>", data)
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_chunks = [re.findall(pattern, chunk) for chunk in chunks]
    return pre_tokenized_chunks


def get_stats(splits: dict[bytes, list[bytes]], word_freqs: dict[bytes, int]) -> defaultdict[tuple[bytes, bytes], int]:
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue

        for i in range(len(split) - 1):
            pair_freqs[(split[i], split[i + 1])] += freq

    return pair_freqs


def get_merge_pair(pair_freqs: defaultdict[tuple[bytes, bytes], int]) -> tuple[tuple[bytes, bytes], int]:
    best_pair = None
    max_freq = -1
    for pair, freq in pair_freqs.items():
        if freq > max_freq or (freq == max_freq and pair > best_pair):
            max_freq = freq
            best_pair = pair

    return best_pair, max_freq


def merge_pairs(splits: dict[bytes, list[bytes]], merge_pair: tuple[bytes, bytes]) -> dict[bytes, list[bytes]]:
    new_subword = merge_pair[0] + merge_pair[1]
    for word in splits:
        split = splits[word]
        if len(split) == 1:
            continue

        if new_subword not in word:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == merge_pair[0] and split[i + 1] == merge_pair[1]:
                split = split[:i] + [new_subword] + split[i + 2 :]
            else:
                i += 1

        splits[word] = split

    return splits


def train_bpe(
    vocab_size: int, special_tokens: list[str], file_path: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Initialize vocabulary and merges
    vocab = {}  # dict[int, str]
    merges = []  # list[tuple[bytes, bytes]]

    # add special tokens
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}

    # add initial 256 characters
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    pre_tokenized_chunks = pre_tokenize(file_path)
    # print(pre_tokenized_chunks)

    word_freqs = defaultdict(int)
    for chunk in pre_tokenized_chunks:
        for word in chunk:
            word_freqs[word.encode("utf-8")] += 1

    splits = {word: [bytes([i]) for i in word] for word in word_freqs}

    while len(vocab) < vocab_size:
        pair_freqs = get_stats(splits, word_freqs)
        # find the best merge pair
        merge_pair, freq = get_merge_pair(pair_freqs)
        splits = merge_pairs(splits, merge_pair)
        new_token_id = len(vocab)
        vocab[new_token_id] = merge_pair[0] + merge_pair[1]
        merges.append((merge_pair[0], merge_pair[1]))

    return vocab, merges
