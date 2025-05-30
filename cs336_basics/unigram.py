# reference: https://huggingface.co/learn/llm-course/chapter6/7?fw=pt

from collections import defaultdict
from math import log
from typing import Optional
from transformers import AutoTokenizer
import copy


def encode_word(word: str, model: dict[str, float]) -> tuple[list[str], Optional[float]]:
    """Encode a word into subword tokens using the unigram model.

    Args:
        word: The word to encode
        model: dictionary mapping tokens to their negative log probabilities

    Returns:
        A tuple containing:
        - list of tokens that the word was split into
        - Total score (negative log probability) of the tokenization, or None if word cannot be tokenized
    """
    # Initialize dynamic programming table with base case
    best_segmentations = [{"start": 0, "score": 1}] + [{"start": None, "score": None} for _ in range(len(word))]

    # Fill the dynamic programming table
    for start_idx in range(len(word)):
        best_score_at_start = best_segmentations[start_idx]["score"]
        if best_score_at_start is None:
            continue

        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token not in model:
                continue

            score = model[token] + best_score_at_start
            curr_best = best_segmentations[end_idx]
            if curr_best["score"] is None or curr_best["score"] > score:
                best_segmentations[end_idx] = {"start": start_idx, "score": score}

    # Check if word could be tokenized
    segmentation = best_segmentations[-1]
    if segmentation["score"] is None:
        return ["<unk>"], None

    # Reconstruct the tokens from the dynamic programming table
    tokens = []
    score = segmentation["score"]
    start = segmentation["start"]
    end = len(word)

    while start != 0:
        tokens.insert(0, word[start:end])
        next_start = best_segmentations[start]["start"]
        end = start
        start = next_start
    tokens.insert(0, word[start:end])

    return tokens, score


def compute_loss(model: dict[str, float], word_freqs: dict[str, int]) -> float:
    """Compute the total loss (negative log likelihood) of the corpus under the model.

    Args:
        model: dictionary mapping tokens to their negative log probabilities
        word_freqs: dictionary mapping words to their frequencies in the corpus

    Returns:
        Total loss across all words
    """
    loss = 0
    for word, freq in word_freqs.items():
        _, word_loss = encode_word(word, model)
        if word_loss is not None:
            loss += freq * word_loss
    return loss


def compute_scores(model: dict[str, float], word_freqs: dict[str, int]) -> dict[str, float]:
    """Compute improvement scores for each token in the model.

    The score represents how much the loss would increase if the token were removed.

    Args:
        model: dictionary mapping tokens to their negative log probabilities
        word_freqs: dictionary mapping words to their frequencies in the corpus

    Returns:
        dictionary mapping tokens to their improvement scores
    """
    scores = {}
    model_loss = compute_loss(model, word_freqs)

    for token, score in model.items():
        # Always keep single character tokens
        if len(token) == 1:
            continue

        # Compute loss without this token
        model_without_token = copy.deepcopy(model)
        model_without_token.pop(token)
        scores[token] = compute_loss(model_without_token, word_freqs) - model_loss

    return scores


def main():
    # Example corpus for training
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]

    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")

    # Hyperparameters
    initial_vocab_size = 300
    percent_to_remove = 0.1  # Remove 10% of tokens in each iteration
    vocab_size = 100  # Target vocabulary size

    # Count word frequencies
    word_freqs = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        words = [word for word, _ in words_with_offsets]
        for word in words:
            word_freqs[word] += 1

    # Count character and subword frequencies
    char_freqs = defaultdict(int)
    subwords_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        # Count single characters
        for i in range(len(word)):
            char_freqs[word[i]] += freq
            # Count subwords of length >= 2
            for j in range(i + 2, len(word) + 1):
                subwords_freqs[word[i:j]] += freq

    # Build initial vocabulary with most frequent subwords
    sorted_subwords = sorted(subwords_freqs.items(), key=lambda x: x[1], reverse=True)
    token_freqs = dict(list(char_freqs.items()) + sorted_subwords[: initial_vocab_size - len(char_freqs)])

    # Convert frequencies to negative log probabilities
    total_sum = sum(token_freqs.values())
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

    # Iteratively prune vocabulary until target size is reached
    while len(model) > vocab_size:
        # Compute improvement scores and sort tokens
        scores = compute_scores(model, word_freqs)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1])

        # Remove lowest scoring tokens
        num_to_remove = int(len(model) * percent_to_remove)
        for i in range(num_to_remove):
            token_freqs.pop(sorted_scores[i][0])

        # Update model probabilities
        total_sum = sum(token_freqs.values())
        model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}


if __name__ == "__main__":
    main()
