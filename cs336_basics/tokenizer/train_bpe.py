import multiprocessing
import os
import regex as re
from typing import BinaryIO
from collections import Counter


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    chunk_size = max(1, file_size // desired_num_chunks)
    bounds = [i * chunk_size for i in range(desired_num_chunks + 1)]
    bounds[-1] = file_size
    mini = 4096  # 4k scan step (bigger save syscall)
    for bi in range(1, len(bounds) - 1):
        pos = bounds[bi]
        file.seek(pos)
        while True:
            buf = file.read(mini)
            if not buf:
                bounds[bi] = file_size
                break
            found = buf.find(split_special_token)
            if found != -1:
                bounds[bi] = pos + found
                break
            pos += len(buf)
    return sorted(set(bounds))


def process_chunk(args: tuple[str, int, int, list[str]]) -> Counter[tuple[int]]:
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")
    # remove special tokens
    pattern = "|".join(re.escape(token) for token in special_tokens)
    documents = re.split(pattern, chunk)

    # 2. split into words with GPT2 regex
    word_bytes = Counter()
    for doc in documents:
        tokens = [match.group(0).encode("utf-8") for match in re.finditer(GPT2_PAT, doc)]
        word_bytes.update([tuple(token) for token in tokens])
    return word_bytes


def compute_pair_counts(word_bytes_freqs: dict[tuple[int], int]) -> Counter[tuple[int, int]]:
    counts = Counter()
    for word_bytes, freq in word_bytes_freqs.items():
        for pair in zip(word_bytes, word_bytes[1:]):
            counts[pair] += freq
    return counts


def merge_pair(token_ids: tuple[int], pair: tuple[int, int], new_id: int) -> tuple[int]:
    new_token_ids = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
            new_token_ids.append(new_id)
            i += 2
        else:
            new_token_ids.append(token_ids[i])
            i += 1
    return tuple(new_token_ids)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocab and merge
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")

    # pre-tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries, boundaries[1:])]
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)

    # get word frequencies in bytes form
    word_bytes_freqs = Counter()  # dict[tuple[int], int]
    for chunk_result in chunk_results:
        word_bytes_freqs.update(chunk_result)

    vocab_index = len(vocab)
    while vocab_index < vocab_size:
        counts = compute_pair_counts(word_bytes_freqs)  # dict[tuple[int, int], int]
        max_pair = max(counts, key=lambda pair: (counts[pair], vocab[pair[0]], vocab[pair[1]]))

        # add to vocab
        vocab[vocab_index] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

        # update word_bytes
        new_word_bytes_freqs = Counter()
        for word_byte, freq in word_bytes_freqs.items():
            new_word_byte = merge_pair(word_byte, max_pair, vocab_index)
            new_word_bytes_freqs[new_word_byte] += freq
        word_bytes_freqs = new_word_bytes_freqs

        # update vocab index
        vocab_index += 1

    return vocab, merges
