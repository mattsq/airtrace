import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from airtrace.models.base import ARBaseModel


class LazyLinearWrapper(nn.Module):
    """Module with a lazy parameter to verify counting skips uninitialized tensors."""

    def __init__(self):
        super().__init__()
        self.linear = nn.LazyLinear(2)


class DummyARModel(ARBaseModel):
    def __init__(self):
        super().__init__(input_dim=3, output_dim=2)
        self.lazy = LazyLinearWrapper()
        self.regular = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor, context=None, **kwargs):
        preds = self.regular(x)
        return {"preds": preds}


def test_get_num_params_skips_uninitialized_parameters():
    model = DummyARModel()

    # Ensure lazy parameter remains uninitialized
    assert any(isinstance(p, UninitializedParameter) for p in model.lazy.parameters())

    # Counting parameters should not attempt to materialize the lazy tensor
    total_params = model.get_num_params()
    expected_regular_params = sum(p.numel() for p in model.regular.parameters())
    assert total_params == expected_regular_params


def test_reset_parameters_delegates_to_submodules():
    class ResettableModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.reset_called = False

        def reset_parameters(self):  # type: ignore[override]
            self.reset_called = True

    model = DummyARModel()
    model.extra = ResettableModule()  # type: ignore[attr-defined]

    model.reset_parameters()
    assert model.extra.reset_called  # type: ignore[attr-defined]

