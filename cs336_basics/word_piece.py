# reference: https://huggingface.co/learn/llm-course/chapter6/6?fw=pt

from collections import defaultdict
from transformers import AutoTokenizer


def compute_pair_scores(splits: dict[str, list[str]], word_freqs: dict[str, int]) -> dict[tuple[str, str], float]:
    """
    Compute scores for each adjacent pair of tokens based on their frequencies.

    Args:
        splits: Dictionary mapping words to their current tokenization
        word_freqs: Dictionary mapping words to their frequencies in the corpus

    Returns:
        Dictionary mapping token pairs to their scores
    """
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)

    # Calculate frequencies of individual tokens and adjacent pairs
    for word, freq in word_freqs.items():
        split = splits[word]

        # Handle single token case
        if len(split) == 1:
            letter_freqs[split[0]] += freq
            continue

        # Process pairs of adjacent tokens
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    # Calculate score for each pair
    scores = {pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]]) for pair, freq in pair_freqs.items()}
    return scores


def merge_pair(a: str, b: str, splits: dict[str, list[str]], word_freqs: dict[str, int]) -> dict[str, list[str]]:
    """
    Merge occurrences of the token pair (a,b) in all tokenized words.

    Args:
        a: First token of the pair to merge
        b: Second token of the pair to merge
        splits: Dictionary mapping words to their current tokenization
        word_freqs: Dictionary mapping words to their frequencies

    Returns:
        Updated splits dictionary with merged tokens
    """
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                # Merge tokens, handling special "##" prefix
                merge = a + b[2:] if b.startswith("##") else a + b
                split = split[:i] + [merge] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


def main() -> list[str]:
    vocab_size = 70
    # Read corpus
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Count word frequencies
    word_freqs = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    # Build initial alphabet
    alphabet = []
    for word in word_freqs.keys():
        if word[0] not in alphabet:
            alphabet.append(word[0])
        for letter in word[1:]:
            token = f"##{letter}"
            if token not in alphabet:
                alphabet.append(token)

    alphabet.sort()

    # Initialize vocabulary with special tokens and alphabet
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + alphabet.copy()

    # Initialize splits dictionary with character-level tokenization
    splits = {word: [c if i == 0 else f"##{c}" for i, c in enumerate(word)] for word in word_freqs.keys()}

    # Iteratively merge best pairs until desired vocab size is reached
    while len(vocab) < vocab_size:
        scores = compute_pair_scores(splits, word_freqs)

        # Find pair with highest score
        best_pair, max_score = "", None
        for pair, score in scores.items():
            if max_score is None or max_score < score:
                best_pair = pair
                max_score = score

        # Merge best pair in all words
        splits = merge_pair(*best_pair, splits)

        # Add merged token to vocabulary
        new_token = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith("##") else best_pair[0] + best_pair[1]
        vocab.append(new_token)

    return vocab
