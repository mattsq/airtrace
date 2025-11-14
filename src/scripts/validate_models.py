"""Model validation script for CI/CD.

This script runs all registered models against synthetic flight data
and reports validation statistics. It's designed to run in CI to ensure
all models can train and produce reasonable predictions.

Usage:
    python src/scripts/validate_models.py [--output results.json] [--seed 42]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from airtrace.data.synthetic import CruiseProfile, SyntheticCruiseGenerator
from airtrace.evaluation.metrics import compute_all_metrics, per_sensor_metrics
from airtrace.models.registry import list_models, build_model


# Sensor columns (excluding timestamp)
SENSOR_COLUMNS = ["fuel_flow", "mach", "altitude", "oat", "n1", "weight"]


def count_trainable_parameters(model: nn.Module) -> int:
    """Count initialized trainable parameters for a model."""
    total = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if isinstance(param, UninitializedParameter):
            # Lazy modules defer materialising parameters until the first forward
            # pass. Skip them here so validation can run without a warmup call.
            continue
        total += param.numel()
    return total


def generate_synthetic_data(
    data_root: Path,
    n_flights: int = 20,
    seed: int = 42
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """Generate synthetic flight data for training and validation.

    Args:
        data_root: Root directory for data storage
        n_flights: Total number of flights to generate
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_flights, val_flights)
    """
    print(f"🛫 Generating {n_flights} synthetic flights...")

    generator = SyntheticCruiseGenerator(data_root=data_root, seed=seed)

    # Create varied profiles for diversity
    profiles = []
    rng = np.random.default_rng(seed)

    for i in range(n_flights):
        profile = CruiseProfile(
            initial_altitude=rng.uniform(30000, 40000),
            initial_mach=rng.uniform(0.75, 0.85),
            initial_weight=rng.uniform(60000, 80000),
            cruise_duration=1800,  # 30 minutes for faster testing
            fuel_flow_base=rng.uniform(2000, 3000),
            n1_base=rng.uniform(80, 90),
            sample_rate=1.0,
            turbulence_level=rng.uniform(0.05, 0.15)
        )
        profiles.append(profile)

    # Generate flights
    flights = []
    for i, profile in enumerate(profiles):
        df = generator.generate_flight(
            flight_id=f"synthetic_{i:03d}",
            profile=profile,
            save=False  # Don't save to disk in CI
        )
        flights.append(df)

    # Split 80/20 train/val
    split_idx = int(0.8 * n_flights)
    train_flights = flights[:split_idx]
    val_flights = flights[split_idx:]

    print(f"✅ Generated {len(train_flights)} train, {len(val_flights)} val flights")
    return train_flights, val_flights


def create_windowed_dataset(
    flights: List[pd.DataFrame],
    window_size: int = 60,
    pred_horizon: int = 1,
    stride: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create windowed dataset from flight dataframes.

    Args:
        flights: List of flight dataframes
        window_size: Input sequence length (seconds)
        pred_horizon: Prediction horizon (seconds)
        stride: Stride between windows

    Returns:
        Tuple of (X, y) tensors
            X: [N, window_size, n_features]
            y: [N, pred_horizon, n_features]
    """
    X_list = []
    y_list = []

    for df in flights:
        # Extract sensor values (exclude timestamp)
        values = df[SENSOR_COLUMNS].values

        # Create sliding windows
        for i in range(0, len(values) - window_size - pred_horizon, stride):
            X_window = values[i:i + window_size]
            y_window = values[i + window_size:i + window_size + pred_horizon]

            X_list.append(X_window)
            y_list.append(y_window)

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)

    return X, y


def normalize_data(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """Z-score normalize data using training statistics.

    Args:
        X_train, y_train, X_val, y_val: Training and validation tensors

    Returns:
        Tuple of (X_train_norm, y_train_norm, X_val_norm, y_val_norm, mean, std)
    """
    # Compute stats from training data only
    mean = X_train.mean(dim=(0, 1)).numpy()
    std = X_train.std(dim=(0, 1)).numpy()
    std = np.where(std < 1e-6, 1.0, std)  # Avoid division by zero

    # Normalize
    X_train_norm = (X_train - torch.tensor(mean)) / torch.tensor(std)
    y_train_norm = (y_train - torch.tensor(mean)) / torch.tensor(std)
    X_val_norm = (X_val - torch.tensor(mean)) / torch.tensor(std)
    y_val_norm = (y_val - torch.tensor(mean)) / torch.tensor(std)

    return X_train_norm, y_train_norm, X_val_norm, y_val_norm, mean, std


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu"
) -> List[float]:
    """Train a model for a few epochs.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        n_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on

    Returns:
        List of training losses per epoch
    """
    model = model.to(device)

    # Check if model has trainable parameters
    n_trainable = count_trainable_parameters(model)

    if n_trainable == 0:
        # Model has no trainable parameters (e.g., baseline models)
        # Just compute and return losses without training
        print(f"  ⚠️  Model has no trainable parameters - skipping optimization")
        model.eval()
        criterion = nn.MSELoss()
        losses = []

        with torch.no_grad():
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0

                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    output = model(X_batch)
                    preds = output["preds"]
                    loss = criterion(preds, y_batch)

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / n_batches
                losses.append(avg_loss)

                if (epoch + 1) % 5 == 0:
                    print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

        return losses

    # Normal training for models with parameters
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(X_batch)
            preds = output["preds"]

            # Compute loss
            loss = criterion(preds, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.6f}")

    return losses


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on validation data.

    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Tuple of (predictions, targets) as numpy arrays
    """
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)

            output = model(X_batch)
            preds = output["preds"]

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return preds, targets


def validate_single_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    output_dim: int,
    n_epochs: int = 10,
    device: str = "cpu"
) -> Dict:
    """Validate a single model.

    Args:
        model_name: Name of registered model
        train_loader: Training data loader
        val_loader: Validation data loader
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        n_epochs: Number of training epochs
        device: Device to use

    Returns:
        Dictionary with validation results
    """
    print(f"\n{'=' * 60}")
    print(f"🔍 Validating model: {model_name}")
    print(f"{'=' * 60}")

    try:
        # Build model (use minimal config)
        config = {
            "name": model_name,
            "params": {}
        }

        # Add model-specific minimal params
        if model_name == "gru_ar":
            config["params"] = {"hidden_size": 64, "num_layers": 1, "dropout": 0.0}
        elif model_name == "tcn":
            config["params"] = {"num_channels": [32, 64], "kernel_size": 3, "dropout": 0.0}
        elif model_name == "transformer":
            config["params"] = {
                "d_model": 64,
                "nhead": 4,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 128,
                "dropout": 0.0
            }

        model = build_model(config, input_dim=input_dim, output_dim=output_dim)

        n_params = model.get_num_params()
        n_trainable = count_trainable_parameters(model)
        print(f"📊 Model parameters: {n_params:,} (trainable: {n_trainable:,})")

        # Training phase
        if n_trainable > 0:
            print(f"🏋️  Training for {n_epochs} epochs...")
        else:
            print(f"📐 Baseline model (no training) - computing loss...")
        start_time = time.time()
        train_losses = train_model(
            model=model,
            train_loader=train_loader,
            n_epochs=n_epochs,
            device=device
        )
        train_time = time.time() - start_time

        # Evaluation phase
        print(f"📈 Evaluating on validation set...")
        preds, targets = evaluate_model(
            model=model,
            val_loader=val_loader,
            device=device
        )

        # Compute metrics
        overall_metrics = compute_all_metrics(preds.flatten(), targets.flatten())
        sensor_metrics = per_sensor_metrics(preds, targets, SENSOR_COLUMNS)

        # Print results
        print(f"\n📊 Overall Metrics:")
        for metric_name, value in overall_metrics.items():
            print(f"  {metric_name.upper():8s}: {value:.6f}")

        print(f"\n⏱️  Training time: {train_time:.2f}s")
        print(f"✅ Model validation successful!")

        return {
            "model_name": model_name,
            "status": "success",
            "n_params": n_params,
            "train_time": train_time,
            "final_train_loss": float(train_losses[-1]),
            "overall_metrics": {k: float(v) for k, v in overall_metrics.items()},
            "sensor_metrics": {
                sensor: {k: float(v) for k, v in metrics.items()}
                for sensor, metrics in sensor_metrics.items()
            }
        }

    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            "model_name": model_name,
            "status": "failed",
            "error": str(e)
        }


def print_summary_table(results: List[Dict]):
    """Print a summary table of all model validation results.

    Args:
        results: List of validation result dictionaries
    """
    print(f"\n{'=' * 80}")
    print("📋 MODEL VALIDATION SUMMARY")
    print(f"{'=' * 80}\n")

    # Header
    print(f"{'Model':<15} {'Status':<10} {'Params':<12} {'Train Time':<12} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    print(f"{'-' * 15} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}")

    # Rows
    for result in results:
        model_name = result["model_name"]
        status = result["status"]

        if status == "success":
            n_params = f"{result['n_params']:,}"
            train_time = f"{result['train_time']:.2f}s"
            rmse = f"{result['overall_metrics']['rmse']:.4f}"
            mae = f"{result['overall_metrics']['mae']:.4f}"
            r2 = f"{result['overall_metrics']['r2']:.4f}"

            print(f"{model_name:<15} {'✅ PASS':<10} {n_params:<12} {train_time:<12} {rmse:<10} {mae:<10} {r2:<10}")
        else:
            print(f"{model_name:<15} {'❌ FAIL':<10} {'-':<12} {'-':<12} {'-':<10} {'-':<10} {'-':<10}")

    print()

    # Success rate
    n_success = sum(1 for r in results if r["status"] == "success")
    n_total = len(results)
    success_rate = (n_success / n_total * 100) if n_total > 0 else 0

    print(f"Success Rate: {n_success}/{n_total} ({success_rate:.1f}%)")
    print()


def main():
    """Main validation script."""
    parser = argparse.ArgumentParser(
        description="Validate all registered models against synthetic flight data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_validation_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-flights",
        type=int,
        default=20,
        help="Number of synthetic flights to generate"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="Number of training epochs per model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu or cuda)"
    )

    args = parser.parse_args()

    print(f"\n{'=' * 80}")
    print("🚀 AIRTRACE MODEL VALIDATION")
    print(f"{'=' * 80}\n")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate synthetic data
    data_root = Path("data")
    train_flights, val_flights = generate_synthetic_data(
        data_root=data_root,
        n_flights=args.n_flights,
        seed=args.seed
    )

    # Create windowed datasets
    print(f"\n🪟 Creating windowed datasets...")
    X_train, y_train = create_windowed_dataset(train_flights, window_size=60, pred_horizon=1)
    X_val, y_val = create_windowed_dataset(val_flights, window_size=60, pred_horizon=1)

    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")

    # Normalize data
    print(f"\n🔧 Normalizing data...")
    X_train, y_train, X_val, y_val, mean, std = normalize_data(
        X_train, y_train, X_val, y_val
    )

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    # Get all registered models
    model_names = list_models()
    print(f"\n📦 Found {len(model_names)} registered models: {model_names}")

    # Validate each model
    input_dim = X_train.shape[-1]
    output_dim = y_train.shape[-1]

    results = []
    for model_name in model_names:
        result = validate_single_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            input_dim=input_dim,
            output_dim=output_dim,
            n_epochs=args.n_epochs,
            device=args.device
        )
        results.append(result)

    # Print summary
    print_summary_table(results)

    # Save results
    output_path = Path(args.output)
    output_data = {
        "seed": args.seed,
        "n_flights": args.n_flights,
        "n_epochs": args.n_epochs,
        "batch_size": args.batch_size,
        "device": args.device,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"💾 Results saved to {output_path}")

    # Exit with error code if any model failed
    n_failed = sum(1 for r in results if r["status"] == "failed")
    if n_failed > 0:
        print(f"\n❌ {n_failed} model(s) failed validation")
        sys.exit(1)
    else:
        print(f"\n✅ All models passed validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
