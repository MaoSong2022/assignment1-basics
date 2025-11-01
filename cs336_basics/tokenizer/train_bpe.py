from collections import Counter
import multiprocessing
import regex as re
import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def split_into_chunks(input_path: str, delimiter: bytes, special_tokens: list[str]) -> list[bytes]:
    chunks = []
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    # from longest to shortest
    special_token_bytes.sort(key=lambda x: len(x), reverse=True)
    splitting_patten = b"|".join(re.escape(special_token) for special_token in special_token_bytes)
    compiled_pattern = re.compile(splitting_patten)
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, delimiter)

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)

            split_parts = compiled_pattern.split(chunk_bytes)
            for part in split_parts:
                if not part:
                    continue
                chunks.append(part)

    return chunks


def split_into_words(chunk: bytes) -> Counter[bytes]:
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""".encode("utf-8")
    words = re.findall(GPT2_PAT, chunk)
    return Counter(words)


def pre_tokenize(chunks: list[bytes]) -> dict[bytes, int]:
    word_freqs = Counter()

    # TODO: use multiprocessing to optimize the process
    with multiprocessing.Pool(processes=10) as p:
        results = p.map(split_into_words, chunks)

    for result in results:
        word_freqs += result

    return dict(word_freqs)


def count_pair_freqs(word_freqs: dict[tuple[int], int]) -> dict[tuple[int, int], int]:
    pair_freqs = Counter()
    for word, freq in word_freqs.items():
        pair_freqs.update({x: freq for x in zip(word, word[1:])})

    return dict(pair_freqs)


def merge_pairs(
    word_freqs: dict[list[int], int], token_pair: tuple[int, int], new_token_id: int
) -> dict[list[int], int]:
    new_pair_freqs = {}

    for token, freq in word_freqs.items():
        i = 0
        new_token = []
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i + 1]) == token_pair:
                new_token.append(new_token_id)
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        new_pair_freqs[tuple(new_token)] = freq

    return new_pair_freqs


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # split doc into chunks based on special token
    chunks = split_into_chunks(
        input_path=input_path, delimiter="<|endoftext|>".encode("utf-8"), special_tokens=special_tokens
    )

    print(f"len of chunks: {len(chunks)}")

    # compute word frequencies in bytes
    word_freqs = pre_tokenize(chunks)

    # compute word frequencies in int
    word_freqs = {tuple(word): freq for word, freq in word_freqs.items()}

    # initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}

    # add special tokens
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")

    merges = []

    token_index = len(vocab)
    while token_index < vocab_size:
        # compute frequency of pairs
        pair_freqs = count_pair_freqs(word_freqs)

        # select new token
        token_pair = max(pair_freqs, key=lambda k: (pair_freqs[k], vocab[k[0]], vocab[k[1]]))
        merges.append((vocab[token_pair[0]], vocab[token_pair[1]]))

        print(token_index, f"({vocab[token_pair[0]]}, {vocab[token_pair[1]]}), freqs: {pair_freqs[token_pair]}")
        vocab[token_index] = vocab[token_pair[0]] + vocab[token_pair[1]]

        # merge pairs
        word_freqs = merge_pairs(word_freqs, token_pair, token_index)
        token_index += 1


    return vocab, merges
