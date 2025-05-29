"""Byte Pair Encoding (BPE) tokenizer implementation.

This module implements a BPE tokenizer that learns subword units from text data.
It includes functions for pre-tokenization, pair frequency counting, merging tokens,
and training the BPE vocabulary.
"""

import regex as re
from collections import defaultdict


def pre_tokenize(file_path: str) -> list[list[str]]:
    """Pre-tokenize text into chunks of words and subwords.

    Args:
        file_path: Path to the text file to tokenize.

    Returns:
        List of lists containing pre-tokenized strings. The outer list represents text chunks
        separated by <|endoftext|> tokens, while inner lists contain the pre-tokenized strings.
    """
    with open(file_path) as f:
        data = f.read()

    # Split text into chunks at <|endoftext|> tokens
    chunks = re.split(r"<\|endoftext\|>", data)

    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_chunks = [re.findall(pattern, chunk) for chunk in chunks]
    return pre_tokenized_chunks


def get_stats(
    splits: dict[bytes, list[bytes]], word_freqs: dict[bytes, int]
) -> tuple[defaultdict[tuple[bytes, bytes], int], defaultdict[tuple[bytes, bytes], set[bytes]]]:
    """Calculate statistics for adjacent token pairs in the vocabulary.

    Args:
        splits: Dictionary mapping words to their current tokenization.
        word_freqs: Dictionary mapping words to their frequencies in the corpus.

    Returns:
        A tuple containing:
        - pair_freqs: Dictionary mapping token pairs to their frequencies
        - pair_to_word: Dictionary mapping token pairs to the set of words containing them
    """
    pair_freqs = defaultdict(int)
    pair_to_word = defaultdict(set)

    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue

        # Count frequencies of adjacent pairs and track which words contain each pair
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
            pair_to_word[pair].add(word)

    return pair_freqs, pair_to_word


def get_merge_pair(pair_freqs: defaultdict[tuple[bytes, bytes], int]) -> tuple[tuple[bytes, bytes], int]:
    """Find the most frequent pair of adjacent tokens to merge.

    Args:
        pair_freqs: Dictionary mapping token pairs to their frequencies.

    Returns:
        A tuple containing:
        - The best pair to merge (as a tuple of two byte sequences)
        - The frequency of that pair

    Note:
        If multiple pairs have the same frequency, the lexicographically larger pair is chosen.
    """
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
    """Merge all occurrences of the selected token pair and update statistics.

    Args:
        splits: Dictionary mapping words to their current tokenization.
        merge_pair: The pair of tokens to merge.
        pair_freqs: Dictionary mapping token pairs to their frequencies.
        pair_to_words: Dictionary mapping token pairs to words containing them.
        word_freqs: Dictionary mapping words to their frequencies.

    Returns:
        Updated splits dictionary with the merged tokens.
    """
    token1, token2 = merge_pair
    new_token = token1 + token2
    words_to_update = list(pair_to_words[merge_pair])

    # Remove the merged pair from tracking dictionaries
    if merge_pair in pair_freqs:
        del pair_freqs[merge_pair]
    if merge_pair in pair_to_words:
        del pair_to_words[merge_pair]

    # Process each word containing the merge pair
    for word in words_to_update:
        split = splits[word]
        freq_of_word = word_freqs[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            # Check if current position contains the merge pair
            if split[i] == token1 and split[i + 1] == token2:
                # Update frequencies for pairs involving tokens before the merge
                if i > 0:
                    prev_token = split[i - 1]
                    old_left_pair = (prev_token, token1)
                    pair_freqs[old_left_pair] -= freq_of_word
                    new_left_pair = (prev_token, new_token)
                    pair_freqs[new_left_pair] += freq_of_word
                    pair_to_words[new_left_pair].add(word)

                # Update frequencies for pairs involving tokens after the merge
                if i < len(split) - 2:
                    next_token = split[i + 2]
                    old_right_pair = (token2, next_token)
                    pair_freqs[old_right_pair] -= freq_of_word
                    new_right_pair = (new_token, next_token)
                    pair_freqs[new_right_pair] += freq_of_word
                    pair_to_words[new_right_pair].add(word)

                # Replace the pair with the merged token
                split = split[:i] + [new_token] + split[i + 2 :]
            else:
                i += 1

        # Update the tokenization for this word
        splits[word] = split

    return splits


def train_bpe(
    vocab_size: int, special_tokens: list[str], file_path: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on the input text.

    Args:
        vocab_size: Target vocabulary size.
        special_tokens: List of special tokens to include in vocabulary.
        file_path: Path to training text file.

    Returns:
        A tuple containing:
        - vocab: Dictionary mapping token IDs to byte sequences
        - merges: List of merge operations (as tuples of byte sequences)
    """
    # Initialize vocabulary with special tokens and base characters
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}
    merges = []

    # Add all possible bytes (0-255) to the vocabulary
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    # Pre-tokenize input text and count word frequencies
    pre_tokenized_chunks = pre_tokenize(file_path)
    word_freqs = defaultdict(int)
    for chunk in pre_tokenized_chunks:
        for word in chunk:
            word_freqs[word.encode("utf-8")] += 1

    # Initialize each word as sequence of bytes
    splits = {word: [bytes([i]) for i in word] for word in word_freqs}

    # Get initial statistics
    pair_freqs, pair_to_words = get_stats(splits, word_freqs)

    # Main training loop: merge pairs until desired vocabulary size is reached
    while len(vocab) < vocab_size:
        merge_pair, freq = get_merge_pair(pair_freqs)
        splits = merge_pairs(splits, merge_pair, pair_freqs, pair_to_words, word_freqs)

        # Add merged token to vocabulary
        new_token_id = len(vocab)
        vocab[new_token_id] = merge_pair[0] + merge_pair[1]
        merges.append((merge_pair[0], merge_pair[1]))

    return vocab, merges
