from argparse import ArgumentParser
import numpy as np
import tqdm

from tokenizers import Tokenizer

from loguru import logger

logger.add("processing.log")


def main():
    parser = ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--input_filepath", type=str)
    args = parser.parse_args()

    chunk_size = 100000

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    output_filepath = args.input_filepath.replace(".txt", ".npy")

    all_token_ids = []
    total_tokens = 0
    chunk_count = 0

    # 3. 分块读取并Tokenize
    with open(args.input_filepath, encoding="utf-8") as f:
        for line in f:
            logger.info(f"len(all_token_ids)={len(all_token_ids)}, total tokens: {total_tokens}, chunk_count: {chunk_count}")
            line = line.strip()
            if not line:
                continue

            # Tokenize当前行
            encoding = tokenizer.encode(line, add_special_tokens=False)
            token_ids = np.array(encoding.ids, dtype=np.int32)
            all_token_ids.append(token_ids)
            total_tokens += len(token_ids)

            if len(all_token_ids) >= chunk_size:
                chunk = np.concatenate(all_token_ids)
                # 首次写入：覆盖；后续：追加（需借助np.memmap）
                if chunk_count == 0:
                    np.save(output_filepath, chunk)
                else:
                    # 内存映射追加
                    existing = np.load(output_filepath, mmap_mode="r+")
                    new_arr = np.concatenate([existing, chunk])
                    np.save(output_filepath, new_arr)
                    del existing  # 释放内存映射
                all_token_ids = []
                chunk_count += 1

    if all_token_ids:
        final_chunk = np.concatenate(all_token_ids)
        if chunk_count == 0:
            np.save(output_filepath, final_chunk)
        else:
            existing = np.load(output_filepath, mmap_mode="r+")
            new_arr = np.concatenate([existing, final_chunk])
            np.save(output_filepath, new_arr)
            del existing


if __name__ == "__main__":
    main()


# python ./scripts/preprocess_dataset.py --tokenizer_path /root/assignment1-basics/hf_tokenizer/tinystories/tokenizer.json --input_filepath /root/autodl-tmp/data/TinyStoriesV2-GPT4-train.txt
# python ./scripts/preprocess_dataset.py --tokenizer_path /root/assignment1-basics/hf_tokenizer/tinystories/tokenizer.json --input_filepath /root/autodl-tmp/data/TinyStoriesV2-GPT4-valid.txt
# python ./scripts/preprocess_dataset.py --tokenizer_path /root/assignment1-basics/hf_tokenizer/OWT-32k/tokenizer.json --input_filepath /root/autodl-tmp/data/owt_train.txt
# python ./scripts/preprocess_dataset.py --tokenizer_path /root/assignment1-basics/hf_tokenizer/OWT-32k/tokenizer.json --input_filepath /root/autodl-tmp/data/owt_valid.txt