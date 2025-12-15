"""Integration tests for DLinear/NLinear with OneStepTask.

This test suite validates that decomposition models (DLinear, NLinear)
work correctly with OneStepTask across different pred_len configurations.
"""
import pytest
import torch
import warnings

from airtrace.models.dlinear import DLinearModel, NLinearModel
from airtrace.tasks.one_step import OneStepTask


@pytest.mark.parametrize("pred_len", [1, 7, 16, 32])
def test_dlinear_with_one_step_task_various_pred_lens(pred_len):
    """Test DLinear with OneStepTask across different pred_len values.

    This ensures that OneStepTask correctly handles DLinear outputs
    regardless of the configured pred_len parameter.
    """
    # Model configured with pred_len (as done via Hydra config)
    model = DLinearModel(
        input_dim=5,
        output_dim=5,
        seq_len=60,
        pred_len=pred_len,
        kernel_size=25
    )

    task = OneStepTask({"loss": "mse", "metrics": ["rmse"]})

    batch = {
        "x": torch.randn(8, 60, 5),
        "y": torch.randn(8, pred_len, 5),
    }

    # Should work without warnings regardless of pred_len
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = task.training_step(batch, model)

        # Check for shape-related warnings
        shape_warnings = [
            warning for warning in w
            if "target size" in str(warning.message).lower()
            or "shape" in str(warning.message).lower()
        ]
        assert len(shape_warnings) == 0, \
            f"No shape warnings expected with pred_len={pred_len}. Got: {shape_warnings}"

    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].shape == ()
    assert "rmse" in result

    # Verify validation also works
    val_result = task.validation_step(batch, model)
    assert isinstance(val_result["loss"], torch.Tensor)


@pytest.mark.parametrize("pred_len", [1, 16, 32])
def test_nlinear_with_one_step_task(pred_len):
    """Test NLinear with OneStepTask across different pred_len values."""
    model = NLinearModel(
        input_dim=10,
        output_dim=10,
        seq_len=128,
        pred_len=pred_len,
        center_data=True
    )

    task = OneStepTask({"loss": "mse", "metrics": ["mae"]})

    batch = {
        "x": torch.randn(4, 128, 10),
        "y": torch.randn(4, pred_len, 10),
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = task.training_step(batch, model)

        shape_warnings = [
            warning for warning in w
            if "target size" in str(warning.message).lower()
        ]
        assert len(shape_warnings) == 0

    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].shape == ()
    assert "mae" in result


def test_dlinear_output_shape_verification():
    """Verify DLinear always outputs [B, pred_len, D] regardless of task."""
    model = DLinearModel(
        input_dim=3,
        output_dim=3,
        seq_len=50,
        pred_len=10,
        kernel_size=5
    )

    x = torch.randn(2, 50, 3)
    output = model(x)

    # Model should output full pred_len
    assert output["preds"].shape == (2, 10, 3), \
        "DLinear should always output [B, pred_len, D]"


def test_nlinear_output_shape_verification():
    """Verify NLinear always outputs [B, pred_len, D] regardless of task."""
    model = NLinearModel(
        input_dim=5,
        output_dim=5,
        seq_len=64,
        pred_len=16,
        center_data=False
    )

    x = torch.randn(4, 64, 5)
    output = model(x)

    # Model should output full pred_len
    assert output["preds"].shape == (4, 16, 5), \
        "NLinear should always output [B, pred_len, D]"


def test_dlinear_decomposition_with_one_step():
    """Test that DLinear's decomposition components are preserved in extras."""
    model = DLinearModel(
        input_dim=8,
        output_dim=8,
        seq_len=100,
        pred_len=20,
        kernel_size=15
    )

    task = OneStepTask({"loss": "mse", "metrics": []})

    batch = {
        "x": torch.randn(2, 100, 8),
        "y": torch.randn(2, 20, 8),
    }

    # Get model output directly
    model_output = model(batch["x"])

    # Verify extras contain decomposition
    assert "extras" in model_output
    assert "seasonal_component" in model_output["extras"]
    assert "trend_component" in model_output["extras"]

    # Verify task still works
    result = task.training_step(batch, model)
    assert isinstance(result["loss"], torch.Tensor)


def test_dlinear_different_input_output_dims():
    """Test DLinear with different input and output dimensions."""
    # This is a common scenario where we predict a subset of sensors
    model = DLinearModel(
        input_dim=10,  # 10 input features
        output_dim=3,  # Predict only 3 target sensors
        seq_len=128,
        pred_len=32,
        kernel_size=25
    )

    task = OneStepTask({"loss": "mse", "metrics": ["rmse", "mae"]})

    batch = {
        "x": torch.randn(16, 128, 10),  # All 10 input features
        "y": torch.randn(16, 32, 3),    # Only 3 target sensors
    }

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = task.training_step(batch, model)

        # No warnings expected
        assert len([warning for warning in w if "target" in str(warning.message).lower()]) == 0

    assert isinstance(result["loss"], torch.Tensor)
    assert "rmse" in result
    assert "mae" in result


def test_nlinear_different_input_output_dims():
    """Test NLinear with different input and output dimensions."""
    model = NLinearModel(
        input_dim=15,  # 15 input features
        output_dim=5,  # Predict only 5 target sensors
        seq_len=256,
        pred_len=16,
        center_data=True
    )

    task = OneStepTask({"loss": "mse", "metrics": []})

    batch = {
        "x": torch.randn(8, 256, 15),  # All 15 input features
        "y": torch.randn(8, 16, 5),    # Only 5 target sensors
    }

    result = task.training_step(batch, model)
    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].shape == ()


@pytest.mark.parametrize("model_class,model_name", [
    (DLinearModel, "DLinear"),
    (NLinearModel, "NLinear"),
])
def test_model_with_one_step_task_batch_sizes(model_class, model_name):
    """Test that both models work with various batch sizes."""
    model = model_class(
        input_dim=4,
        output_dim=4,
        seq_len=64,
        pred_len=8,
    )

    task = OneStepTask({"loss": "mse", "metrics": []})

    for batch_size in [1, 4, 16, 64]:
        batch = {
            "x": torch.randn(batch_size, 64, 4),
            "y": torch.randn(batch_size, 8, 4),
        }

        result = task.training_step(batch, model)
        assert isinstance(result["loss"], torch.Tensor), \
            f"{model_name} failed with batch_size={batch_size}"
        assert result["loss"].shape == ()


def test_dlinear_gradients_flow():
    """Test that gradients flow correctly through DLinear with OneStepTask."""
    model = DLinearModel(
        input_dim=5,
        output_dim=5,
        seq_len=50,
        pred_len=10,
        kernel_size=7
    )

    # Ensure model parameters require grad
    for param in model.parameters():
        param.requires_grad = True

    task = OneStepTask({"loss": "mse", "metrics": []})

    batch = {
        "x": torch.randn(4, 50, 5, requires_grad=True),
        "y": torch.randn(4, 10, 5),
    }

    result = task.training_step(batch, model)
    loss = result["loss"]

    # Backprop
    loss.backward()

    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "Gradients should flow through the model"


def test_nlinear_centering_with_one_step():
    """Test NLinear with center_data=True and False."""
    for center_data in [True, False]:
        model = NLinearModel(
            input_dim=6,
            output_dim=6,
            seq_len=80,
            pred_len=12,
            center_data=center_data
        )

        task = OneStepTask({"loss": "mse", "metrics": []})

        batch = {
            "x": torch.randn(8, 80, 6),
            "y": torch.randn(8, 12, 6),
        }

        result = task.training_step(batch, model)
        assert isinstance(result["loss"], torch.Tensor), \
            f"NLinear failed with center_data={center_data}"


if __name__ == "__main__":
    # Run tests manually for debugging
    print("Running integration tests for DLinear/NLinear with OneStepTask...")

    print("\n1. Testing DLinear with various pred_lens...")
    for pl in [1, 7, 16, 32]:
        test_dlinear_with_one_step_task_various_pred_lens(pl)
        print(f"   ✓ pred_len={pl}")

    print("\n2. Testing NLinear with various pred_lens...")
    for pl in [1, 16, 32]:
        test_nlinear_with_one_step_task(pl)
        print(f"   ✓ pred_len={pl}")

    print("\n3. Testing output shape verification...")
    test_dlinear_output_shape_verification()
    print("   ✓ DLinear output shape")
    test_nlinear_output_shape_verification()
    print("   ✓ NLinear output shape")

    print("\n4. Testing DLinear decomposition...")
    test_dlinear_decomposition_with_one_step()
    print("   ✓ Decomposition preserved")

    print("\n5. Testing different input/output dimensions...")
    test_dlinear_different_input_output_dims()
    print("   ✓ DLinear with dim mismatch")
    test_nlinear_different_input_output_dims()
    print("   ✓ NLinear with dim mismatch")

    print("\n6. Testing various batch sizes...")
    test_model_with_one_step_task_batch_sizes(DLinearModel, "DLinear")
    test_model_with_one_step_task_batch_sizes(NLinearModel, "NLinear")
    print("   ✓ All batch sizes")

    print("\n7. Testing gradient flow...")
    test_dlinear_gradients_flow()
    print("   ✓ Gradients flow correctly")

    print("\n8. Testing NLinear centering...")
    test_nlinear_centering_with_one_step()
    print("   ✓ Both centering modes")

    print("\n✅ All integration tests passed!")
