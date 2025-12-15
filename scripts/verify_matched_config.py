"""Verify the itransformer_matched.yaml config has the correct parameter count."""

import torch
from airtrace.models.informer import InformerModel
from airtrace.models.itransformer import iTransformerModel


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model_name, model, config):
    """Print a summary of the model and its parameters."""
    params = count_parameters(model)
    print(f"\n{model_name} Model Summary")
    print("=" * 60)
    for key, value in config.items():
        if key not in ['input_dim', 'output_dim']:
            print(f"  {key}: {value}")
    print(f"\n  Total Parameters: {params:,}")
    print("=" * 60)
    return params


def main():
    # Use realistic dimensions from aircraft sensor data
    input_dim = 10
    output_dim = 10

    print("\nParameter Count Verification")
    print("=" * 60)
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print("=" * 60)

    # Informer config (from informer.yaml)
    informer_config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'd_model': 96,
        'nhead': 4,
        'e_layers': 2,
        'd_layers': 1,
        'ff_dim': 192,
        'factor': 5,
        'dropout': 0.1,
        'pred_len': 1,
        'distill': True
    }

    # iTransformer matched config (from itransformer_matched.yaml)
    itransformer_config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'pred_len': 1,
        'd_model': 128,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 296,
        'dropout': 0.1,
        'activation': 'gelu',
        'use_norm': True
    }

    # Create models
    informer = InformerModel(**informer_config)
    informer_params = print_model_summary("Informer", informer, informer_config)

    # Need to initialize iTransformer with a forward pass
    itransformer = iTransformerModel(**itransformer_config)
    dummy_input = torch.randn(1, 32, input_dim)
    _ = itransformer(dummy_input)
    itransformer_params = print_model_summary("iTransformer (matched)", itransformer, itransformer_config)

    # Compare
    diff = itransformer_params - informer_params
    ratio = itransformer_params / informer_params * 100

    print("\nParameter Comparison")
    print("=" * 60)
    print(f"Informer:        {informer_params:>10,} parameters")
    print(f"iTransformer:    {itransformer_params:>10,} parameters")
    print(f"Difference:      {diff:>10,} ({ratio:.2f}%)")
    print("=" * 60)

    if abs(ratio - 100) < 1:
        print(f"\nSUCCESS: Parameter counts match within 1% ({ratio:.2f}%)")
    else:
        print(f"\nWARNING: Parameter counts differ by more than 1% ({ratio:.2f}%)")

    # Architecture comparison
    print("\nArchitectural Differences")
    print("=" * 60)
    print("Informer:")
    print("  - Encoder-decoder architecture with sparse (ProbSparse) attention")
    print("  - Token representation: time points")
    print("  - Attention: across time points (with sparsity)")
    print("  - Distillation: reduces sequence length between encoder layers")
    print("")
    print("iTransformer:")
    print("  - Encoder-only architecture with standard attention")
    print("  - Token representation: variates (sensors)")
    print("  - Attention: across sensors (dense)")
    print("  - Feed-forward: learns temporal patterns per sensor")
    print("=" * 60)


if __name__ == '__main__':
    main()
