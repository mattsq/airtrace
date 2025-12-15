"""
Visualize autoregressive rollout of a trained checkpoint.

This script:
1. Loads a trained model checkpoint
2. Loads a single sample from the test dataset
3. Runs autoregressive inference (using predictions as inputs for next step)
4. Compares rolled-out predictions vs ground truth
5. Shows how errors accumulate over time
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


def autoregressive_rollout(
    model,
    x_initial,
    num_steps,
    transform_pipeline
):
    """
    Roll out predictions autoregressively.

    Args:
        model: Trained model
        x_initial: Initial context window [1, T_in, D]
        num_steps: Number of steps to roll out
        transform_pipeline: Transform pipeline for inverse transforming predictions

    Returns:
        predictions_transformed: List of predictions in transformed space
        predictions_original: List of predictions in original space
    """
    model.eval()

    predictions_transformed = []
    predictions_original = []

    # Current context window (will be updated as we roll forward)
    x_current = x_initial.clone()

    # Get the transforms
    zscore_transform = transform_pipeline.transforms[0]
    diff_transform = transform_pipeline.transforms[1]

    for step in range(num_steps):
        # Make prediction
        with torch.no_grad():
            pred_dict = model(x_current)
            pred = pred_dict['preds']  # [1, 1, D]

        # Store transformed prediction
        pred_np = pred[0, 0].cpu().numpy()  # [D]
        predictions_transformed.append(pred_np.copy())

        # Inverse transform the prediction to original scale
        # Need to inverse as if it's a y value (using scaler_y)
        pred_for_inv = pred_np.reshape(1, -1)  # [1, D]
        pred_after_diff, _ = diff_transform.inverse(pred_for_inv, None)
        pred_flat = pred_after_diff.reshape(-1, pred_after_diff.shape[-1])
        pred_orig = zscore_transform.scaler_y.inverse_transform(pred_flat)[0]  # [D]
        predictions_original.append(pred_orig.copy())

        # Update context window: slide forward
        # Drop the oldest timestep, append the new prediction
        x_current = torch.cat([x_current[:, 1:, :], pred], dim=1)  # [1, T_in, D]

    return np.array(predictions_transformed), np.array(predictions_original)


def visualize_autoregressive_rollout(
    checkpoint_path: str,
    sample_idx: int = 0,
    num_rollout_steps: int = 32,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize autoregressive rollout on a single sample.

    Args:
        checkpoint_path: Path to the checkpoint file
        sample_idx: Index of the sample to visualize from test set
        num_rollout_steps: Number of steps to roll out autoregressively
        save_path: Optional path to save the figure
        show: Whether to display the figure
    """
    checkpoint_path = Path(checkpoint_path)

    # Load the config from the run directory
    run_dir = checkpoint_path.parent.parent
    config_path = run_dir / ".hydra" / "config.yaml"

    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)

    # Build transforms
    transform_pipeline = None
    if "transforms" in cfg and "pipeline" in cfg.transforms:
        transform_pipeline = build_transforms(cfg.transforms.pipeline)
        print(f"Transform pipeline: {transform_pipeline}")

    # Build data module
    print("Setting up data...")
    datamodule = SensorDataModule(
        data_config=cfg.data,
        transforms=transform_pipeline,
        batch_size=1,
        num_workers=0,
        shuffle=None
    )
    datamodule.setup()

    # Build model
    print("Building model...")
    model = build_model(
        config=cfg.model,
        input_dim=datamodule.in_dim,
        output_dim=datamodule.out_dim
    )

    # Load checkpoint weights
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Get test dataloader
    print("Creating test dataloader...")
    test_loader = datamodule.test_dataloader()

    # Get a single sample
    print(f"Getting sample {sample_idx} from test set...")
    for i, batch in enumerate(test_loader):
        if i == sample_idx:
            break

    # Unpack batch
    x = batch["x"]  # [1, T_in, D]
    y = batch["y"]  # [1, T_out, D]

    # Limit rollout steps to available ground truth
    num_rollout_steps = min(num_rollout_steps, y.shape[1])

    print(f"Rolling out {num_rollout_steps} steps autoregressively...")
    pred_transformed, pred_original = autoregressive_rollout(
        model, x, num_rollout_steps, transform_pipeline
    )

    print(f"Autoregressive predictions shape: {pred_original.shape}")

    # Get ground truth in original scale
    zscore_transform = transform_pipeline.transforms[0]
    diff_transform = transform_pipeline.transforms[1]

    x_np = x[0].cpu().numpy()
    y_np = y[0].cpu().numpy()

    # Inverse transform x and y
    x_orig, _ = transform_pipeline.inverse(x_np, None)

    y_after_diff, _ = diff_transform.inverse(y_np, None)
    y_flat = y_after_diff.reshape(-1, y_after_diff.shape[-1])
    y_orig = zscore_transform.scaler_y.inverse_transform(y_flat).reshape(y_after_diff.shape)

    # Truncate ground truth to rollout length
    y_orig = y_orig[:num_rollout_steps]

    # Get sensor names
    sensor_names = cfg.data.window.target_sensors
    n_sensors = len(sensor_names)

    # Create visualization
    print("Creating visualization...")
    n_cols = 3
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Time steps for plotting
    t_input = np.arange(len(x_orig))
    t_rollout = np.arange(len(x_orig), len(x_orig) + num_rollout_steps)

    for i, sensor in enumerate(sensor_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Plot input context
        ax.plot(t_input, x_orig[:, i], 'b-', label='Input Context', linewidth=2, alpha=0.7)

        # Plot ground truth
        ax.plot(t_rollout, y_orig[:, i], 'g-', label='Ground Truth',
                linewidth=2, alpha=0.7)

        # Plot autoregressive rollout
        ax.plot(t_rollout, pred_original[:, i], 'r--', label='AR Rollout',
                linewidth=2, marker='o', markersize=4, alpha=0.8)

        # Add vertical line at prediction boundary
        ax.axvline(x=len(x_orig), color='gray', linestyle=':', alpha=0.5, linewidth=1)

        # Calculate cumulative error metrics
        mae = np.mean(np.abs(y_orig[:, i] - pred_original[:, i]))
        rmse = np.sqrt(np.mean((y_orig[:, i] - pred_original[:, i])**2))

        # Calculate final error (at last timestep)
        final_error = np.abs(y_orig[-1, i] - pred_original[-1, i])
        final_rel_error = final_error / (np.abs(y_orig[-1, i]) + 1e-8) * 100

        ax.set_title(
            f'{sensor}\nMAE: {mae:.2f} | Final Error: {final_error:.2f} ({final_rel_error:.1f}%)',
            fontsize=9
        )
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Remove empty subplots
    for i in range(n_sensors, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])

    # Add title
    model_name = cfg.model.name
    task_name = cfg.task.name
    fig.suptitle(
        f'Autoregressive Rollout | Model: {model_name} | Sample: {sample_idx} | Steps: {num_rollout_steps}',
        fontsize=14,
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

    parser = argparse.ArgumentParser(description='Visualize autoregressive rollout')
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--sample-idx',
        type=int,
        default=0,
        help='Index of sample to visualize (default: 0)'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=32,
        help='Number of steps to roll out (default: 32)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save figure (default: None)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the figure'
    )

    args = parser.parse_args()

    visualize_autoregressive_rollout(
        checkpoint_path=args.checkpoint,
        sample_idx=args.sample_idx,
        num_rollout_steps=args.num_steps,
        save_path=args.save,
        show=not args.no_show
    )
