"""
Visualize one-step predictions rolled over the entire ground truth sequence.

This script demonstrates "teacher forcing" evaluation where:
1. At each timestep, we use actual ground truth as context
2. Make a one-step prediction
3. Roll the window forward by 1 step using ground truth
4. Repeat to build a full sequence of one-step predictions

This is different from autoregressive rollout where predictions feed back as inputs.
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


def rolling_one_step_predictions(
    model,
    x_initial,
    y_ground_truth,
    transform_pipeline
):
    """
    Make one-step predictions by rolling the window forward using ground truth.

    At each step:
    - Use ground truth context window
    - Make one-step prediction
    - Roll window forward by 1 step using ground truth (not prediction)

    Args:
        model: Trained model
        x_initial: Initial context window [1, T_in, D]
        y_ground_truth: Ground truth sequence [T_out, D] in transformed space
        transform_pipeline: Transform pipeline for inverse transforming

    Returns:
        predictions_original: [num_steps, D] predictions in original scale
    """
    model.eval()

    T_in = x_initial.shape[1]
    num_steps = len(y_ground_truth)

    predictions_transformed = []
    predictions_original = []

    # Get the transforms
    zscore_transform = transform_pipeline.transforms[0]
    diff_transform = transform_pipeline.transforms[1]

    # Build full sequence: [context, ground_truth]
    # Shape: [1, T_in + T_out, D]
    full_sequence = torch.cat([x_initial,
                                torch.tensor(y_ground_truth, dtype=torch.float32).unsqueeze(0)],
                               dim=1)

    for step in range(num_steps):
        # Extract context window ending at this position
        # Context is from [step : step + T_in]
        x_current = full_sequence[:, step:step + T_in, :]  # [1, T_in, D]

        # Make prediction
        with torch.no_grad():
            pred_dict = model(x_current)
            pred = pred_dict['preds']  # [1, 1, D]

        # Store transformed prediction
        pred_np = pred[0, 0].cpu().numpy()  # [D]
        predictions_transformed.append(pred_np.copy())

        # Inverse transform the prediction to original scale
        pred_for_inv = pred_np.reshape(1, -1)  # [1, D]
        pred_after_diff, _ = diff_transform.inverse(pred_for_inv, None)
        pred_flat = pred_after_diff.reshape(-1, pred_after_diff.shape[-1])
        pred_orig = zscore_transform.scaler_y.inverse_transform(pred_flat)[0]  # [D]
        predictions_original.append(pred_orig.copy())

    return np.array(predictions_transformed), np.array(predictions_original)


def visualize_rolling_one_step(
    checkpoint_path: str,
    sample_idx: int = 0,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize rolling one-step predictions over the full ground truth sequence.

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

    y_np = y[0].cpu().numpy()  # [T_out, D]

    print(f"Making rolling one-step predictions over {len(y_np)} steps...")
    pred_transformed, pred_original = rolling_one_step_predictions(
        model, x, y_np, transform_pipeline
    )

    print(f"Rolling predictions shape: {pred_original.shape}")

    # Get ground truth in original scale
    zscore_transform = transform_pipeline.transforms[0]
    diff_transform = transform_pipeline.transforms[1]

    x_np = x[0].cpu().numpy()

    # Inverse transform x and y
    x_orig, _ = transform_pipeline.inverse(x_np, None)

    y_after_diff, _ = diff_transform.inverse(y_np, None)
    y_flat = y_after_diff.reshape(-1, y_after_diff.shape[-1])
    y_orig = zscore_transform.scaler_y.inverse_transform(y_flat).reshape(y_after_diff.shape)

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
    t_target = np.arange(len(x_orig), len(x_orig) + len(y_orig))

    for i, sensor in enumerate(sensor_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        # Plot input context
        ax.plot(t_input, x_orig[:, i], 'b-', label='Input Context',
                linewidth=2, alpha=0.7)

        # Plot ground truth
        ax.plot(t_target, y_orig[:, i], 'g-', label='Ground Truth',
                linewidth=2, alpha=0.7)

        # Plot rolling one-step predictions
        ax.plot(t_target, pred_original[:, i], 'r--', label='One-Step Predictions',
                linewidth=2, marker='o', markersize=3, alpha=0.8)

        # Add vertical line at prediction boundary
        ax.axvline(x=len(x_orig), color='gray', linestyle=':', alpha=0.5, linewidth=1)

        # Calculate error metrics
        mae = np.mean(np.abs(y_orig[:, i] - pred_original[:, i]))
        rmse = np.sqrt(np.mean((y_orig[:, i] - pred_original[:, i])**2))
        max_error = np.max(np.abs(y_orig[:, i] - pred_original[:, i]))

        ax.set_title(
            f'{sensor}\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | Max Error: {max_error:.2f}',
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
    num_steps = len(y_orig)
    fig.suptitle(
        f'Rolling One-Step Predictions (Teacher Forcing) | Model: {model_name} | Sample: {sample_idx} | Steps: {num_steps}',
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

    parser = argparse.ArgumentParser(
        description='Visualize rolling one-step predictions with teacher forcing'
    )
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

    visualize_rolling_one_step(
        checkpoint_path=args.checkpoint,
        sample_idx=args.sample_idx,
        save_path=args.save,
        show=not args.no_show
    )
