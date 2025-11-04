import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    random_start_indices = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))

    block_indices = random_start_indices[:, None] + np.arange(context_length + 1)

    token_ids = torch.from_numpy(dataset[block_indices].astype(np.int64))

    inputs = token_ids[:, :-1].to(device)
    targets = token_ids[:, 1:].to(device)

    return inputs, targets
