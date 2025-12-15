"""Test for OneStepTask shape mismatch with multi-step models.

This test reproduces and validates the fix for MSE broadcasting errors
when using multi-step models (like DLinear) with OneStepTask.
"""
import pytest
import torch
import warnings

from airtrace.tasks.one_step import OneStepTask


class MultiStepModel(torch.nn.Module):
    """Mock model that outputs pred_len=7 steps (like DLinear)."""

    def forward(self, x, meta=None, **kwargs):
        B, T_in, D = x.shape
        # Simulate DLinear/NLinear behavior: output full pred_len
        preds = torch.randn(B, 7, D)  # pred_len=7
        return {"preds": preds}


class PredictableMultiStepModel(torch.nn.Module):
    """Model that outputs different values per timestep."""

    def forward(self, x, meta=None, **kwargs):
        B, T_in, D = x.shape
        # Create predictions with distinct values per timestep
        # Values: [0, 1, 2, 3, 4, 5, 6] along timestep dimension
        preds = torch.arange(7).view(1, 7, 1).repeat(B, 1, D).float()
        return {"preds": preds}  # [B, 7, D]


def test_one_step_task_handles_multi_step_output():
    """Test that OneStepTask correctly handles multi-step model outputs.

    This test verifies that when a model outputs multiple timesteps (e.g., [B, 7, D]),
    OneStepTask slices to the first timestep [:, :1, :] to match targets [B, 1, D].

    On main branch (without fix): This may trigger a UserWarning about shape mismatch
    On fix branch (with fix): This should pass without warnings
    """
    batch = {
        "x": torch.zeros(64, 256, 30),  # [B=64, T_in=256, D=30]
        "y": torch.randn(64, 7, 30),     # [B=64, T_out=7, D=30]
    }

    task = OneStepTask({"loss": "mse", "metrics": ["rmse"]})

    # With fix: should not trigger warnings
    # Without fix: may trigger UserWarning about different sizes
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = task.training_step(batch, MultiStepModel())

        # Check if any warnings about target size were raised
        target_size_warnings = [
            warning for warning in w
            if "target size" in str(warning.message).lower()
            and "different" in str(warning.message).lower()
        ]

        # With the fix, there should be no warnings
        assert len(target_size_warnings) == 0, \
            f"OneStepTask should not trigger shape warnings. Got: {target_size_warnings}"

    # Verify result structure
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].shape == ()  # Scalar loss
    assert "rmse" in result


def test_one_step_task_slices_to_single_timestep():
    """Verify fix: OneStepTask slices multi-step predictions to [:, :1, :].

    This test ensures that when a model outputs multiple timesteps with different
    values, OneStepTask uses only the first timestep for loss computation.
    """
    batch = {
        "x": torch.zeros(2, 10, 3),
        "y": torch.ones(2, 7, 3),  # Targets are all 1.0
    }

    task = OneStepTask({"loss": "mse", "metrics": []})
    result = task.training_step(batch, PredictableMultiStepModel())

    # Model outputs timesteps with values [0, 1, 2, 3, 4, 5, 6]
    # With fix: should use only first timestep (value=0)
    # Loss = (0 - 1)^2 = 1.0
    expected_loss = 1.0

    # Without fix: would average across all timesteps
    # Mean of [0,1,2,3,4,5,6] = 3.0
    # Loss would be approximately (3.0 - 1)^2 = 4.0

    assert torch.isclose(result["loss"], torch.tensor(expected_loss), atol=1e-5), \
        f"OneStepTask should slice predictions to first timestep only. " \
        f"Expected loss ~{expected_loss}, got {result['loss'].item()}"


def test_one_step_task_validation_step_handles_multi_step():
    """Test that validation_step also correctly handles multi-step outputs."""
    batch = {
        "x": torch.randn(16, 60, 10),
        "y": torch.randn(16, 7, 10),
    }

    task = OneStepTask({"loss": "mse", "metrics": ["mae", "rmse"]})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = task.validation_step(batch, MultiStepModel())

        # Should not trigger shape warnings
        target_size_warnings = [
            warning for warning in w
            if "target size" in str(warning.message).lower()
        ]
        assert len(target_size_warnings) == 0

    # Verify result structure
    assert isinstance(result["loss"], torch.Tensor)
    assert "mae" in result
    assert "rmse" in result


def test_one_step_task_with_single_step_model():
    """Test that OneStepTask still works correctly with single-step models.

    This ensures the fix doesn't break compatibility with models that
    already output [B, 1, D].
    """

    class SingleStepModel(torch.nn.Module):
        """Mock model that outputs only one timestep."""

        def forward(self, x, meta=None, **kwargs):
            B, T_in, D = x.shape
            preds = torch.zeros(B, 1, D)  # Single timestep output
            return {"preds": preds}

    batch = {
        "x": torch.randn(8, 50, 5),
        "y": torch.ones(8, 1, 5),
    }

    task = OneStepTask({"loss": "mse", "metrics": []})
    result = task.training_step(batch, SingleStepModel())

    # Should work perfectly
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].shape == ()


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running test_one_step_task_handles_multi_step_output...")
    test_one_step_task_handles_multi_step_output()
    print("✓ Passed")

    print("Running test_one_step_task_slices_to_single_timestep...")
    test_one_step_task_slices_to_single_timestep()
    print("✓ Passed")

    print("Running test_one_step_task_validation_step_handles_multi_step...")
    test_one_step_task_validation_step_handles_multi_step()
    print("✓ Passed")

    print("Running test_one_step_task_with_single_step_model...")
    test_one_step_task_with_single_step_model()
    print("✓ Passed")

    print("\nAll tests passed!")
