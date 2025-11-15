"""Detect which models need validation in CI based on git changes.

This script identifies models that have been added or modified, plus baseline
models that should always run (they are fast and serve as sanity checks).

Usage:
    python src/scripts/get_models_to_validate.py [--base-ref BRANCH]
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Set


# Baseline models that always run (fast, non-trainable models)
ALWAYS_RUN_BASELINES = {
    "persistence",
    "moving_average",
    "zero",
    "linear_trend",
    "mean",
    "median",
    "drift",
    "exponential_smoothing",
    "seasonal_naive",
    "polynomial_trend",
    "holt_linear_trend",
    "linear_ar",  # Trainable but simple and fast
    "mlp_ar",     # Trainable but simple and fast
}


def get_changed_files(base_ref: str = "origin/main") -> Set[str]:
    """Get list of changed files compared to base branch.

    Args:
        base_ref: Base reference to compare against (e.g., 'origin/main')

    Returns:
        Set of changed file paths relative to repo root
    """
    try:
        # Get the merge base (common ancestor)
        result = subprocess.run(
            ["git", "merge-base", base_ref, "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        merge_base = result.stdout.strip()

        # Get changed files since merge base
        result = subprocess.run(
            ["git", "diff", "--name-only", merge_base, "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        changed_files = set(result.stdout.strip().split("\n"))

        # Also include untracked files (new models)
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True,
            text=True,
            check=True
        )
        untracked = result.stdout.strip()
        if untracked:
            changed_files.update(untracked.split("\n"))

        return {f for f in changed_files if f}

    except subprocess.CalledProcessError as e:
        print(f"Error getting changed files: {e}", file=sys.stderr)
        print(f"Falling back to validating all models", file=sys.stderr)
        return set()


def extract_model_name_from_config(config_path: Path) -> str:
    """Extract model name from a config file.

    Args:
        config_path: Path to model config YAML file

    Returns:
        Model name (e.g., 'gru_ar')
    """
    # Model name is the config filename without extension
    return config_path.stem


def extract_model_names_from_python(py_path: Path) -> Set[str]:
    """Extract registered model names from a Python file.

    Args:
        py_path: Path to Python model implementation file

    Returns:
        Set of model names registered in the file
    """
    model_names = set()

    try:
        content = py_path.read_text()

        # Find all @register("model_name") decorators
        pattern = r'@register\(["\']([^"\']+)["\']\)'
        matches = re.findall(pattern, content)
        model_names.update(matches)

    except Exception as e:
        print(f"Warning: Could not parse {py_path}: {e}", file=sys.stderr)

    return model_names


def get_models_to_validate(base_ref: str = "origin/main") -> Set[str]:
    """Determine which models need to be validated.

    Args:
        base_ref: Base reference to compare against

    Returns:
        Set of model names to validate
    """
    models = set()

    # Always include baseline models
    models.update(ALWAYS_RUN_BASELINES)

    # Get changed files
    changed_files = get_changed_files(base_ref)

    if not changed_files:
        print("No changed files detected, running baselines only", file=sys.stderr)
        return models

    print(f"Detected {len(changed_files)} changed files", file=sys.stderr)

    # Check for changed model configs
    for file_path in changed_files:
        path = Path(file_path)

        # Check if it's a model config file
        if path.parts and path.parts[0] == "configs" and "model" in path.parts:
            if path.suffix in {".yaml", ".yml"}:
                model_name = extract_model_name_from_config(path)
                models.add(model_name)
                print(f"  Added (config changed): {model_name}", file=sys.stderr)

        # Check if it's a model implementation file
        elif path.parts and path.parts[0] == "src" and "models" in path.parts:
            if path.suffix == ".py" and path.stem != "__init__":
                model_names = extract_model_names_from_python(path)
                for model_name in model_names:
                    models.add(model_name)
                    print(f"  Added (impl changed): {model_name}", file=sys.stderr)

        # Also check if validate_models.py itself changed (run all baselines)
        elif "validate_models.py" in str(path):
            print(f"  Validation script changed, keeping baselines only", file=sys.stderr)

    return models


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Determine which models to validate based on git changes"
    )
    parser.add_argument(
        "--base-ref",
        type=str,
        default="origin/main",
        help="Base git reference to compare against (default: origin/main)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all models (ignores changes)"
    )

    args = parser.parse_args()

    if args.all:
        # Don't output anything - let validate_models.py run all models
        print("", end="")
        return

    models = get_models_to_validate(args.base_ref)

    # Sort for consistency
    models_sorted = sorted(models)

    print(f"Models to validate ({len(models_sorted)}): {models_sorted}", file=sys.stderr)

    # Output space-separated list for shell consumption
    print(" ".join(models_sorted))


if __name__ == "__main__":
    main()
