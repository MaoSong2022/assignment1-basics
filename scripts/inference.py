import torch
import argparse

from tokenizers import Tokenizer
from loguru import logger

import cs336_basics.model.model as transformer_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Transformer Text Generation")
    parser.add_argument("--prompt", type=str, nargs="+", default=["Once upon a time"])
    parser.add_argument("--checkpoint", type=str, default="checkpoints/baseline/best_llm_model.pt")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.8)
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Using device: {device}")

    tokenizer = Tokenizer.from_file("tokenizer/TinyStories/tokenizer.json")
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    encodings = tokenizer.encode_batch(args.prompt)
    token_ids = torch.tensor([e.ids for e in encodings], dtype=torch.long).to(device)

    print(f"input prompt: {args.prompt}")

    model_cfg = {
        "vocab_size": 10000,
        "num_layers": 4,
        "d_model": 512,
        "num_heads": 16,
        "d_ff": 1344,
        "max_seq_len": 256,
        "theta": 10000
    }

    model = transformer_model.Model(**model_cfg).to(device)

    state_dict = torch.load('checkpoints/baseline/best_llm_model.pt', weights_only=True)
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()

    logger.info("Generating...")
    with torch.no_grad():
        output_token_ids = model.generate(
            token_ids, 
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=50,
            top_p=0.9,
            eos_token_id=256 # <|endoftext|>
        )

    results = tokenizer.decode_batch(output_token_ids.cpu().tolist())
    
    for i, res in enumerate(results):
        print(f"\n{'='*20} Result {i+1} {'='*20}\n{res.strip()}")


if __name__ == "__main__":
    main()