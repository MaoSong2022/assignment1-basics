from argparse import ArgumentParser
import multiprocessing
import os
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict

from loguru import logger
import tqdm


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


def compute_pair_counts(
    word_bytes_freqs: dict[tuple[int], int],
) -> tuple[Counter[tuple[int, int]], defaultdict[tuple[int, int], set[tuple[int]]]]:
    counts = Counter()
    pair_to_words = defaultdict(set)
    for word_bytes, freq in word_bytes_freqs.items():
        for pair in zip(word_bytes, word_bytes[1:]):
            counts[pair] += freq
            pair_to_words[pair].add(word_bytes)
    return counts, pair_to_words


def merge_pair(
    max_pair: tuple[int, int],
    new_token_id: int,
    counts: Counter[tuple[int, int]],
    pair_to_words: defaultdict[tuple[int, int], set[tuple[int]]],
    word_bytes_freqs: Counter[tuple[int]],
) -> Counter[tuple[int, int]]:
    old_words = list(pair_to_words[max_pair])
    for word in old_words:
        # remove old words
        old_freq = word_bytes_freqs[word]
        del word_bytes_freqs[word]
        for pair in zip(word, word[1:]):
            pair_to_words[pair].discard(word)
            counts[pair] -= old_freq
            if counts[pair] <= 0:
                del counts[pair]
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

        # create new word
        i = 0
        new_token_ids = []
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == max_pair:
                new_token_ids.append(new_token_id)
                i += 2
            else:
                new_token_ids.append(word[i])
                i += 1
        new_word = tuple(new_token_ids)

        # update related information
        word_bytes_freqs[new_word] = old_freq
        for pair in zip(new_token_ids, new_token_ids[1:]):
            counts[pair] += old_freq
            pair_to_words[pair].add(new_word)

    return counts


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # initialize vocab and merge
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    # pre-tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 1000, b"<|endoftext|>") # use large desire_num_chunks for large dataset

    chunk_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries, boundaries[1:])]
    total_chunks = len(chunk_args)
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_results = []
        for result in tqdm.tqdm(pool.imap(process_chunk, chunk_args), total=total_chunks, desc="Processing chunks"):
            chunk_results.append(result)

    logger.info("complete pre-tokenizing")

    # get word frequencies in bytes form
    word_bytes_freqs = Counter()  # dict[tuple[int], int]
    for chunk_result in chunk_results:
        word_bytes_freqs.update(chunk_result)

    counts, pair_to_words = compute_pair_counts(word_bytes_freqs)  # dict[tuple[int, int], int]

    vocab_index = len(vocab)
    while vocab_index < vocab_size:
        max_pair = max(counts, key=lambda pair: (counts[pair], vocab[pair[0]], vocab[pair[1]]))

        # add to vocab
        vocab[vocab_index] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
        logger.info(
            f"vocab index: {vocab_index}, new_token: {vocab[max_pair[0]] + vocab[max_pair[1]]}, count: {counts[max_pair]}"
        )

        # update word_bytes
        counts = merge_pair(max_pair, vocab_index, counts, pair_to_words, word_bytes_freqs)

        # update vocab index
        vocab_index += 1

    return vocab, merges


def main():
    # python cs336_basics/tokenizer/train_bpe.py -i /root/autodl-tmp/data/TinyStoriesV2-GPT4-train.txt -v 10000 -o TinyStories
    # python cs336_basics/tokenizer/train_bpe.py -i /root/autodl-tmp/data/owt_train.txt -v 32000 -o owt
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path")
    parser.add_argument("-v", "--vocab_size", type=int)
    parser.add_argument("-o", "--output")
    args = parser.parse_args()
    logger.add(f"{args.output}_output.log", mode="w")

    vocab, merges = train_bpe(args.input_path, args.vocab_size, special_tokens=["<|endoftext|>"], num_processes=10)
    logger.info("complete training")
    import json

    serializable_vocab = {k: v.decode("utf-8", errors="ignore") for k, v in vocab.items()}
    with open(f"{args.output}_vocab.json", "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, indent=4, ensure_ascii=False)

    serializable_merges = [(a.decode("utf-8", errors="ignore"), b.decode("utf-8", errors="ignore")) for a, b in merges]
    with open(f"{args.output}_merges.txt", "w", encoding="utf-8") as f:
        for merge in serializable_merges:
            f.write(f"{merge[0]} {merge[1]}\n")
    logger.info("complete")


if __name__ == "__main__":
    main()
