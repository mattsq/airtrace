import torch
import torch.nn as nn

from airtrace.export.end_to_end_model import SingleBatchInputWrapper


class DummyModel(nn.Module):
    def forward(self, x: torch.Tensor, context=None):
        self.last_input_shape = x.shape
        return x * 2


def test_single_batch_input_wrapper_adds_and_removes_batch_dim():
    model = DummyModel()
    wrapper = SingleBatchInputWrapper(model)

    unbatched = torch.ones(5, 3)
    output = wrapper(unbatched)

    assert model.last_input_shape == (1, 5, 3)
    assert output.shape == (5, 3)
    torch.testing.assert_close(output, torch.full((5, 3), 2.0))


def test_single_batch_input_wrapper_preserves_existing_batch():
    model = DummyModel()
    wrapper = SingleBatchInputWrapper(model)

    batched = torch.ones(2, 4, 1)
    output = wrapper(batched)

    assert model.last_input_shape == (2, 4, 1)
    assert output.shape == (2, 4, 1)
    torch.testing.assert_close(output, torch.full((2, 4, 1), 2.0))
