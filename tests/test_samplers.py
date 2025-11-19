import pytest
import torch

from airtrace.data.samplers import BlockShuffleSampler


def test_block_shuffle_sampler_emits_full_permutation() -> None:
    dataset = list(range(1025))
    generator = torch.Generator().manual_seed(0)
    sampler = BlockShuffleSampler(dataset, block_size=256, generator=generator)

    indices = list(iter(sampler))

    assert len(indices) == len(dataset)
    assert sorted(indices) == list(range(len(dataset)))


def test_block_shuffle_sampler_changes_order_each_epoch() -> None:
    dataset = list(range(2048))
    sampler = BlockShuffleSampler(dataset, block_size=512)

    first_epoch = list(iter(sampler))
    second_epoch = list(iter(sampler))

    assert first_epoch != second_epoch
    assert sorted(first_epoch) == sorted(second_epoch) == list(range(len(dataset)))


def test_block_shuffle_sampler_validates_block_size() -> None:
    dataset = list(range(10))
    with pytest.raises(ValueError):
        BlockShuffleSampler(dataset, block_size=0)
