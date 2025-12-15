"""
Visualize how a trained checkpoint performs on an individual sample.

This script:
1. Loads a trained model checkpoint
2. Loads a single sample from the test dataset
3. Runs inference
4. Inverse transforms predictions back to original scale
5. Creates detailed visualizations comparing predictions vs ground truth
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


def visualize_sample(
    checkpoint_path: str,
    sample_idx: int = 0,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize a single sample prediction from a trained checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        sample_idx: Index of the sample to visualize from test set
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
        batch_size=1,  # Single sample
        num_workers=0,  # No multiprocessing for single sample
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
    x = batch["x"]  # [batch, T_in, D]
    y = batch["y"]  # [batch, T_out, D]

    # Get first sample in batch
    x_sample = x[0:1]  # [1, T_in, D]
    y_sample = y[0:1]  # [1, T_out, D]

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        pred_dict = model(x_sample)  # Returns dict with 'preds' key
        pred = pred_dict['preds']  # [1, T_out, D] or [1, 1, D] for one-step

    # Get sensor names
    sensor_names = cfg.data.window.target_sensors
    n_sensors = len(sensor_names)

    # Convert to numpy for plotting
    x_np = x_sample[0].cpu().numpy()  # [T_in, D]
    y_np = y_sample[0].cpu().numpy()  # [T_out, D]
    pred_np = pred[0].cpu().numpy()  # [T_out, D] or [1, D]

    print(f"Input shape: {x_np.shape}")
    print(f"Target shape: {y_np.shape}")
    print(f"Prediction shape: {pred_np.shape}")

    # Inverse transform to get original scale
    print("Inverse transforming to original scale...")

    # Get the transforms
    zscore_transform = transform_pipeline.transforms[0]
    diff_transform = transform_pipeline.transforms[1]

    if pred_np.shape[0] == 1:
        # One-step prediction - we only have one prediction step
        # Inverse transform x normally
        x_orig, _ = transform_pipeline.inverse(x_np, None)

        # For y, we need to manually apply inverse with correct scalers
        # Step 1: Inverse diff (cumsum)
        y_after_diff, _ = diff_transform.inverse(y_np, None)

        # Step 2: Inverse zscore using scaler_y
        y_flat = y_after_diff.reshape(-1, y_after_diff.shape[-1])
        y_orig = zscore_transform.scaler_y.inverse_transform(y_flat).reshape(y_after_diff.shape)

        # For prediction: same process
        y_with_pred = y_np.copy()
        y_with_pred[0] = pred_np[0]
        pred_after_diff, _ = diff_transform.inverse(y_with_pred, None)
        pred_flat = pred_after_diff.reshape(-1, pred_after_diff.shape[-1])
        pred_inv_full = zscore_transform.scaler_y.inverse_transform(pred_flat).reshape(pred_after_diff.shape)
        pred_orig = pred_inv_full[:1]  # Just the first step

    else:
        # Multi-step prediction
        x_orig, _ = transform_pipeline.inverse(x_np, None)

        # Inverse y with correct scaler
        y_after_diff, _ = diff_transform.inverse(y_np, None)
        y_flat = y_after_diff.reshape(-1, y_after_diff.shape[-1])
        y_orig = zscore_transform.scaler_y.inverse_transform(y_flat).reshape(y_after_diff.shape)

        # Inverse pred with correct scaler
        pred_after_diff, _ = diff_transform.inverse(pred_np, None)
        pred_flat = pred_after_diff.reshape(-1, pred_after_diff.shape[-1])
        pred_orig = zscore_transform.scaler_y.inverse_transform(pred_flat).reshape(pred_after_diff.shape)

    # Create visualization
    print("Creating visualization...")
    n_cols = 3
    n_rows = (n_sensors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Time steps for plotting
    t_input = np.arange(len(x_orig))
    t_target = np.arange(len(x_orig), len(x_orig) + len(y_orig))
    t_pred = np.arange(len(x_orig), len(x_orig) + len(pred_orig))

    for i, sensor in enumerate(sensor_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Plot input context
        ax.plot(t_input, x_orig[:, i], 'b-', label='Input Context', linewidth=2, alpha=0.7)

        # Plot ground truth target
        ax.plot(t_target, y_orig[:, i], 'g-', label='Ground Truth', linewidth=2, alpha=0.7)

        # Plot prediction - make it more visible
        ax.plot(t_pred, pred_orig[:, i], 'r', label='Prediction',
                linewidth=3, marker='o', markersize=8, linestyle='--', alpha=0.9)

        # Add vertical line at prediction boundary
        ax.axvline(x=len(x_orig), color='gray', linestyle=':', alpha=0.5, linewidth=1)

        # Calculate error for one-step
        if len(pred_orig) == 1:
            error = np.abs(y_orig[0, i] - pred_orig[0, i])
            rel_error = error / (np.abs(y_orig[0, i]) + 1e-8) * 100
            ax.set_title(f'{sensor}\nError: {error:.2f} ({rel_error:.1f}%)', fontsize=10)
        else:
            mae = np.mean(np.abs(y_orig[:, i] - pred_orig[:, i]))
            ax.set_title(f'{sensor}\nMAE: {mae:.2f}', fontsize=10)

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
        f'Model: {model_name} | Task: {task_name} | Sample: {sample_idx}',
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

    parser = argparse.ArgumentParser(description='Visualize checkpoint on individual sample')
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

    visualize_sample(
        checkpoint_path=args.checkpoint,
        sample_idx=args.sample_idx,
        save_path=args.save,
        show=not args.no_show
    )
