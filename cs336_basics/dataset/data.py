import numpy as np
import numpy.typing as npt

import torch
from torch.utils.data import Dataset


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    random_start_indices = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))

    block_indices = random_start_indices[:, None] + np.arange(context_length + 1)

    token_ids = torch.from_numpy(dataset[block_indices].astype(np.int64))

    inputs = token_ids[:, :-1].to(device)
    targets = token_ids[:, 1:].to(device)

    return inputs, targets


class MemmapDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        seq_len: int,
        stride: int = None,
        shuffle: bool = False,
        pad_id: int = 0,
        dtype=np.uint16,
    ):
        self.file_path = file_path
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.pad_id = pad_id

        self.memmap_arr = np.lib.format.open_memmap(file_path, mode="r", dtype=dtype)

        self.total_tokens = len(self.memmap_arr)

        self.num_samples = (self.total_tokens - self.seq_len - 1) // self.stride + 1

        print(f"Dataset Loaded: {self.num_samples} samples, {self.total_tokens} total tokens.")

        if shuffle:
            self.sample_indices = np.random.permutation(self.num_samples)
        else:
            self.sample_indices = np.arange(self.num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = self.sample_indices[idx] * self.stride
        chunk = self.memmap_arr[start : start + self.seq_len + 1].astype(np.int64)

        if len(chunk) < self.seq_len + 1:
            padding = np.full((self.seq_len + 1) - len(chunk), self.pad_id, dtype=np.int64)
            chunk = np.concatenate([chunk, padding])

        full_tensor = torch.from_numpy(chunk)
        inputs, targets = full_tensor[:-1], full_tensor[1:]

        return inputs, targets
