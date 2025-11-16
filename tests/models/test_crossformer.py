import pytest
import torch

from airtrace.models import CrossformerModel


@pytest.fixture
def batch() -> torch.Tensor:
    """Create a dummy batch for testing Crossformer."""
    B, T_in, D_in = 4, 32, 6
    return torch.randn(B, T_in, D_in)


def test_crossformer_forward(batch: torch.Tensor) -> None:
    """Crossformer should return predictions with the configured horizon."""
    model = CrossformerModel(
        input_dim=6,
        output_dim=3,
        seg_len=8,
        seg_stride=4,
        dim_seg_size=3,
        d_model=96,
        nhead=4,
        temporal_depth=2,
        spatial_depth=2,
        pred_len=2,
    )

    output = model(batch)

    assert "preds" in output
    assert output["preds"].shape == (4, 2, 3)
    assert output["extras"]["num_patches"].item() > 0
    assert output["extras"]["num_groups"].item() == model.num_groups


def test_crossformer_handles_padding() -> None:
    """Model should pad short sequences and uneven dimensions gracefully."""
    x = torch.randn(2, 10, 5)  # shorter than seg_len and odd feature count
    model = CrossformerModel(input_dim=5, output_dim=2, seg_len=12, seg_stride=4, dim_seg_size=4)

    output = model(x)

    assert output["preds"].shape == (2, 1, 2)
    assert output["extras"]["num_patches"].item() >= 1
    assert output["extras"]["num_groups"].item() == model.num_groups


def test_crossformer_gradient_flow(batch: torch.Tensor) -> None:
    """Gradients should propagate through Crossformer components."""
    model = CrossformerModel(input_dim=6, output_dim=4, seg_len=8, seg_stride=4, d_model=64, nhead=4)
    batch.requires_grad = True

    preds = model(batch)["preds"]
    loss = preds.mean()
    loss.backward()

    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
