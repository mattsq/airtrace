"""Custom samplers for AirTrace dataloaders."""

from __future__ import annotations

import math
import secrets
from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler


class BlockShuffleSampler(Sampler[int]):
    """Shuffle dataset indices in manageable blocks.

    PyTorch's ``RandomSampler`` generates a full ``torch.randperm(len(dataset))``
    tensor every time a new epoch begins, which can introduce a noticeable stall
    for very large datasets. This sampler avoids that by randomly shuffling
    blocks of indices and only permuting the indices within a single block at a
    time. The first block permutation is generated immediately, so the
    DataLoader can start yielding batches sooner while still presenting batches
    in a randomized order.
    """

    def __init__(
        self,
        data_source: Sized,
        block_size: int = 65536,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        self.data_source = data_source
        self.block_size = block_size
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(secrets.randbits(63))
        self._generator = generator

    def __len__(self) -> int:
        return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        dataset_size = len(self.data_source)
        if dataset_size == 0:
            return iter(())

        block_size = min(self.block_size, dataset_size)
        num_blocks = math.ceil(dataset_size / block_size)
        block_order = torch.randperm(num_blocks, generator=self._generator).tolist()

        for block_idx in block_order:
            start = block_idx * block_size
            end = min(start + block_size, dataset_size)
            block_length = end - start
            local_perm = torch.randperm(block_length, generator=self._generator).tolist()
            for offset in local_perm:
                yield start + offset
