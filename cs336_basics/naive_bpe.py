import regex as re
from collections import defaultdict


def pre_tokenize(file_path: str) -> list[list[str]]:
    """Pre-tokenize text from a file into chunks of tokens.

    Args:
        file_path: Path to the input text file.

    Returns:
        List of lists containing pre-tokenized text chunks. Each inner list contains
        tokens from one chunk of text split on <|endoftext|> markers.
    """
    with open(file_path) as f:
        data = f.read()

    # Split text into chunks at special token
    chunks = re.split(r"<\|endoftext\|>", data)

    # Pattern matches contractions, words, numbers, symbols and whitespace
    pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokenized_chunks = [re.findall(pattern, chunk) for chunk in chunks]
    return pre_tokenized_chunks


def get_stats(bytes_tuple_counter: defaultdict[tuple[bytes, bytes], int]) -> defaultdict[tuple[bytes, bytes], int]:
    """Calculate frequencies of adjacent byte pairs in the vocabulary.

    Args:
        bytes_tuple_counter: Counter mapping byte tuples to their frequencies.

    Returns:
        Counter mapping byte pairs (bigrams) to their frequencies across all tuples.
    """
    counter = defaultdict(int)
    # Count frequencies of adjacent byte pairs
    for bytes_tuple, count in bytes_tuple_counter.items():
        for i in range(len(bytes_tuple) - 1):
            counter[(bytes_tuple[i], bytes_tuple[i + 1])] += count

    return counter


def merge_pairs(
    bytes_tuple_counter: defaultdict[tuple[bytes, bytes], int], merge_pair: tuple[bytes, bytes]
) -> defaultdict[tuple[bytes, bytes], int]:
    """Merge all occurrences of the specified byte pair in the vocabulary.

    Args:
        bytes_tuple_counter: Counter mapping byte tuples to their frequencies.
        merge_pair: The pair of bytes to merge into a single token.

    Returns:
        Updated counter with the specified pair merged wherever it occurs.
    """
    counter = defaultdict(int)
    for bytes_tuple, count in bytes_tuple_counter.items():
        bytes_list = []
        i = 0
        # Iterate through bytes, merging pairs where found
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


def train(
    file_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer on text data.

    Args:
        file_path: Path to the training text file.
        vocab_size: Target size for the vocabulary.
        special_tokens: List of special tokens to include in vocabulary.

    Returns:
        Tuple containing:
        - Dictionary mapping token IDs to byte sequences
        - List of merge rules as (bytes, bytes) pairs
    """
    # Initialize vocabulary with special tokens
    vocab = {i: token.encode("utf-8") for i, token in enumerate(special_tokens)}

    # Add initial byte-level tokens (0-255)
    for i in range(256):
        vocab[len(vocab)] = bytes([i])

    # Pre-tokenize input text
    pre_tokenized_chunks = pre_tokenize(file_path)

    # Count word frequencies
    word_counts = defaultdict(int)
    for chunk in pre_tokenized_chunks:
        for word in chunk:
            word_counts[word] += 1

    # Convert words to byte tuples and count frequencies
    bytes_tuple_counter = defaultdict(int)
    for word, count in word_counts.items():
        bytes_tuple = tuple(bytes([c]) for c in word.encode("utf-8"))
        bytes_tuple_counter[bytes_tuple] += count

    merges = []
    # Learn merge rules until target vocab size is reached
    while len(vocab) < vocab_size:
        new_token_id = len(vocab)
        stats = get_stats(bytes_tuple_counter)

        # Find the most frequent byte pair
        best_score = (0, bytes([0]), bytes([0]))
        for pair, count in stats.items():
            current_score = (count, pair[0], pair[1])
            if current_score > best_score:
                best_score = current_score

        freq, merge_pair = best_score[0], (best_score[1], best_score[2])

        # Apply the merge and update vocabulary
        bytes_tuple_counter = merge_pairs(bytes_tuple_counter, merge_pair)
        new_bytes = merge_pair[0] + merge_pair[1]
        vocab[new_token_id] = new_bytes
        merges.append(merge_pair)

    return vocab, merges
