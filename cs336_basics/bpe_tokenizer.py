from asyncio import new_event_loop
import regex as re
from collections import defaultdict


def pre_tokenize(file_path: str) -> list[list[str]]:
    with open(file_path) as f:
        data = f.read()

    chunks = re.split(r"<\|endoftext\|>", data)
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_chunks = [re.findall(pattern, chunk) for chunk in chunks]
    return pre_tokenized_chunks


def get_stats(
    splits: dict[bytes, list[bytes]], word_freqs: dict[bytes, int]
) -> tuple[defaultdict[tuple[bytes, bytes], int], defaultdict[tuple[bytes, bytes], set[bytes]]]:
    pair_freqs = defaultdict(int)
    pair_to_word = defaultdict(set)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue

        for i in range(len(split) - 1):
            pair_freqs[(split[i], split[i + 1])] += freq
            pair_to_word[(split[i], split[i + 1])].add(word)
    return pair_freqs, pair_to_word


def get_merge_pair(pair_freqs: defaultdict[tuple[bytes, bytes], int]) -> tuple[tuple[bytes, bytes], int]:
    best_pair = None
    max_freq = -1
    for pair, freq in pair_freqs.items():
        if freq > max_freq or (freq == max_freq and pair > best_pair):
            max_freq = freq
            best_pair = pair

    return best_pair, max_freq


def merge_pairs(
    splits: dict[bytes, list[bytes]],
    merge_pair: tuple[bytes, bytes],
    pair_freqs: defaultdict[tuple[bytes, bytes], int],
    pair_to_words: defaultdict[tuple[bytes, bytes], set[bytes]],
    word_freqs: dict[bytes, int],
) -> dict[bytes, list[bytes]]:
    token1, token2 = merge_pair
    new_token = token1 + token2
    words_to_update = list(pair_to_words[merge_pair])

    if merge_pair in pair_freqs:
        del pair_freqs[merge_pair]
    if merge_pair in pair_to_words:
        del pair_to_words[merge_pair]

    for word in words_to_update:
        split = splits[word]
        freq_of_word = word_freqs[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            # Check if the current position and the next form the merge_pair
            if split[i] == token1 and split[i + 1] == token2:
                # Found an instance of the merge_pair

                # 1. Decrement frequencies of old surrounding pairs
                # (previous_token, token1)
                if i > 0:
                    prev_token = split[i - 1]
                    old_left_pair = (prev_token, token1)
                    pair_freqs[old_left_pair] -= freq_of_word
                    new_left_pair = (prev_token, new_token)
                    pair_freqs[new_left_pair] += freq_of_word
                    pair_to_words[new_left_pair].add(word)

                if i < len(split) - 2:
                    next_token = split[i + 2]
                    old_right_pair = (token2, next_token)
                    pair_freqs[old_right_pair] -= freq_of_word
                    new_right_pair = (new_token, next_token)
                    pair_freqs[new_right_pair] += freq_of_word
                    pair_to_words[new_right_pair].add(word)

                split = split[:i] + [new_token] + split[i + 2 :]
            else:
                i += 1

        # Update the splits dictionary for this word
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

    pair_freqs, pair_to_words = get_stats(splits, word_freqs)

    while len(vocab) < vocab_size:
        # find the best merge pair
        merge_pair, freq = get_merge_pair(pair_freqs)
        splits = merge_pairs(splits, merge_pair, pair_freqs, pair_to_words, word_freqs)
        new_token_id = len(vocab)
        vocab[new_token_id] = merge_pair[0] + merge_pair[1]
        merges.append((merge_pair[0], merge_pair[1]))

    return vocab, merges
