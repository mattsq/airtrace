"""
Visualize how prediction errors accumulate during autoregressive rollout.

Shows MAE and RMSE growth over rollout steps.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from airtrace.data.datamodule import SensorDataModule
from airtrace.models.registry import build_model
from airtrace.transforms.registry import build_transforms


def autoregressive_rollout_with_errors(
    model,
    x_initial,
    y_ground_truth,
    transform_pipeline
):
    """
    Roll out predictions and track errors at each step.

    Returns:
        errors_per_step: [num_steps, D] array of absolute errors
        predictions: [num_steps, D] array of predictions
    """
    model.eval()
    num_steps = len(y_ground_truth)

    predictions = []
    errors_per_step = []

    x_current = x_initial.clone()

    zscore_transform = transform_pipeline.transforms[0]
    diff_transform = transform_pipeline.transforms[1]

    for step in range(num_steps):
        # Make prediction
        with torch.no_grad():
            pred_dict = model(x_current)
            pred = pred_dict['preds']  # [1, 1, D]

        pred_np = pred[0, 0].cpu().numpy()

        # Inverse transform
        pred_for_inv = pred_np.reshape(1, -1)
        pred_after_diff, _ = diff_transform.inverse(pred_for_inv, None)
        pred_flat = pred_after_diff.reshape(-1, pred_after_diff.shape[-1])
        pred_orig = zscore_transform.scaler_y.inverse_transform(pred_flat)[0]

        predictions.append(pred_orig.copy())

        # Calculate error vs ground truth at this step
        error = np.abs(pred_orig - y_ground_truth[step])
        errors_per_step.append(error)

        # Update context window
        x_current = torch.cat([x_current[:, 1:, :], pred], dim=1)

    return np.array(errors_per_step), np.array(predictions)


def visualize_error_accumulation(
    checkpoint_path: str,
    sample_idx: int = 0,
    num_rollout_steps: int = 32,
    save_path: str = None,
    show: bool = True
):
    """Visualize error accumulation during rollout."""
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"

    cfg = OmegaConf.load(config_path)
    transform_pipeline = build_transforms(cfg.transforms.pipeline)

    datamodule = SensorDataModule(
        data_config=cfg.data,
        transforms=transform_pipeline,
        batch_size=1,
        num_workers=0,
        shuffle=None
    )
    datamodule.setup()

    model = build_model(
        config=cfg.model,
        input_dim=datamodule.in_dim,
        output_dim=datamodule.out_dim
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = datamodule.test_dataloader()

    for i, batch in enumerate(test_loader):
        if i == sample_idx:
            break

    x = batch["x"]
    y = batch["y"]

    num_rollout_steps = min(num_rollout_steps, y.shape[1])

    # Get ground truth in original scale
    zscore_transform = transform_pipeline.transforms[0]
    diff_transform = transform_pipeline.transforms[1]

    y_np = y[0].cpu().numpy()
    y_after_diff, _ = diff_transform.inverse(y_np, None)
    y_flat = y_after_diff.reshape(-1, y_after_diff.shape[-1])
    y_orig = zscore_transform.scaler_y.inverse_transform(y_flat).reshape(y_after_diff.shape)
    y_orig = y_orig[:num_rollout_steps]

    print(f"Rolling out {num_rollout_steps} steps...")
    errors_per_step, predictions = autoregressive_rollout_with_errors(
        model, x, y_orig, transform_pipeline
    )

    sensor_names = cfg.data.window.target_sensors

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: MAE over time for each sensor
    steps = np.arange(1, num_rollout_steps + 1)
    for i, sensor in enumerate(sensor_names):
        ax1.plot(steps, errors_per_step[:, i], label=sensor, marker='o', markersize=3, alpha=0.7)

    ax1.set_xlabel('Rollout Step', fontsize=12)
    ax1.set_ylabel('Absolute Error', fontsize=12)
    ax1.set_title('Error Accumulation per Sensor', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # Log scale to see all sensors

    # Plot 2: Average normalized error across all sensors
    # Normalize each sensor's error by its ground truth range
    normalized_errors = []
    for i, sensor in enumerate(sensor_names):
        sensor_range = np.max(y_orig[:, i]) - np.min(y_orig[:, i])
        if sensor_range > 0:
            norm_error = errors_per_step[:, i] / sensor_range * 100  # Percentage of range
            normalized_errors.append(norm_error)

    normalized_errors = np.array(normalized_errors)
    mean_norm_error = np.mean(normalized_errors, axis=0)
    std_norm_error = np.std(normalized_errors, axis=0)

    ax2.plot(steps, mean_norm_error, 'b-', linewidth=2, label='Mean')
    ax2.fill_between(steps, mean_norm_error - std_norm_error, mean_norm_error + std_norm_error,
                      alpha=0.3, label='±1 Std Dev')
    ax2.set_xlabel('Rollout Step', fontsize=12)
    ax2.set_ylabel('Normalized Error (% of sensor range)', fontsize=12)
    ax2.set_title('Average Error Growth Across Sensors', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add text with final stats
    final_mean = mean_norm_error[-1]
    ax2.text(0.05, 0.95, f'Final Error: {final_mean:.1f}% of range',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle(
        f'Error Accumulation | Sample: {sample_idx} | Model: {cfg.model.name}',
        fontsize=16,
        fontweight='bold'
    )

    plt.tight_layout()

    if save_path:
        print(f"Saving figure to: {save_path}")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize error accumulation')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--sample-idx', type=int, default=0, help='Sample index')
    parser.add_argument('--num-steps', type=int, default=32, help='Rollout steps')
    parser.add_argument('--save', type=str, default=None, help='Save path')
    parser.add_argument('--no-show', action='store_true', help='Do not display')

    args = parser.parse_args()

    visualize_error_accumulation(
        checkpoint_path=args.checkpoint,
        sample_idx=args.sample_idx,
        num_rollout_steps=args.num_steps,
        save_path=args.save,
        show=not args.no_show
    )
