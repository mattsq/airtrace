"""Script to match iTransformer parameters to Informer configuration."""

import torch
from airtrace.models.informer import InformerModel
from airtrace.models.itransformer import iTransformerModel


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # Example dimensions - typical for aircraft sensor data
    input_dim = 10  # Number of input sensors
    output_dim = 10  # Number of output sensors

    # Informer configuration from user
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

    # Create Informer model
    informer = InformerModel(**informer_config)
    informer_params = count_parameters(informer)

    print(f"Informer Configuration:")
    print(f"  d_model: {informer_config['d_model']}")
    print(f"  nhead: {informer_config['nhead']}")
    print(f"  e_layers: {informer_config['e_layers']}")
    print(f"  d_layers: {informer_config['d_layers']}")
    print(f"  ff_dim: {informer_config['ff_dim']}")
    print(f"  factor: {informer_config['factor']}")
    print(f"  dropout: {informer_config['dropout']}")
    print(f"  pred_len: {informer_config['pred_len']}")
    print(f"  distill: {informer_config['distill']}")
    print(f"\nInformer Total Parameters: {informer_params:,}")

    # Try different iTransformer configurations to match parameter count
    print("\n" + "="*60)
    print("Finding matching iTransformer configurations...")
    print("="*60)

    # Strategy: iTransformer has simpler architecture (no decoder)
    # Main parameters come from:
    # 1. VariateEmbedding: input_length * input_dim * d_model
    # 2. Position embeddings: input_dim * d_model
    # 3. Transformer encoder layers: num_layers * [attention + ffn]
    # 4. Projection head: depends on output configuration

    configs_to_try = [
        # Similar d_model, fewer layers (since no decoder)
        {'d_model': 96, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 192},
        {'d_model': 96, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 192},

        # Slightly larger d_model, 2 layers
        {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 256},

        # Smaller d_model, more layers
        {'d_model': 80, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 160},

        # Match encoder hidden dim more precisely
        {'d_model': 96, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 384},
        {'d_model': 112, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 224},

        # Fine-tuning around the best match (d_model=128)
        {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 288},
        {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 320},
        {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 352},
        {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 384},

        # Try d_model around 128
        {'d_model': 136, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 272},
        {'d_model': 144, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 288},
        {'d_model': 120, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 240},

        # Try 3 layers with smaller dimensions
        {'d_model': 104, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 208},
        {'d_model': 112, 'nhead': 4, 'num_layers': 3, 'dim_feedforward': 224},

        # Fine-tune around the best (d_model=128, dim_feedforward=288)
        {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 296},
        {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 304},
        {'d_model': 128, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 312},

        # Try slight variations in d_model
        {'d_model': 132, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 264},
        {'d_model': 132, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 280},
    ]

    results = []

    for config in configs_to_try:
        itransformer_config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'pred_len': 1,
            'd_model': config['d_model'],
            'nhead': config['nhead'],
            'num_layers': config['num_layers'],
            'dim_feedforward': config['dim_feedforward'],
            'dropout': 0.1,
        }

        # Need to do a forward pass to initialize LazyLinear
        itransformer = iTransformerModel(**itransformer_config)
        # Dummy input: [batch, seq_len, input_dim]
        dummy_input = torch.randn(1, 32, input_dim)
        _ = itransformer(dummy_input)

        itransformer_params = count_parameters(itransformer)
        param_diff = itransformer_params - informer_params
        param_ratio = itransformer_params / informer_params

        results.append({
            'config': config,
            'params': itransformer_params,
            'diff': param_diff,
            'ratio': param_ratio
        })

    # Sort by absolute difference
    results.sort(key=lambda x: abs(x['diff']))

    print("\nTop matching configurations (sorted by closeness):\n")
    for i, result in enumerate(results[:5], 1):
        config = result['config']
        params = result['params']
        diff = result['diff']
        ratio = result['ratio']

        print(f"{i}. iTransformer Configuration:")
        print(f"   d_model: {config['d_model']}")
        print(f"   nhead: {config['nhead']}")
        print(f"   num_layers: {config['num_layers']}")
        print(f"   dim_feedforward: {config['dim_feedforward']}")
        print(f"   dropout: 0.1")
        print(f"   pred_len: 1")
        print(f"   ")
        print(f"   Total Parameters: {params:,}")
        print(f"   Difference: {diff:+,} ({ratio:.2%} of Informer)")
        print()

    print("="*60)
    print("\nRecommended iTransformer configuration:")
    print("="*60)
    best = results[0]
    print(f"""
model:
  name: itransformer
  params:
    d_model: {best['config']['d_model']}
    nhead: {best['config']['nhead']}
    num_layers: {best['config']['num_layers']}
    dim_feedforward: {best['config']['dim_feedforward']}
    dropout: 0.1
    pred_len: 1
""")
    print(f"Target Informer params: {informer_params:,}")
    print(f"Matched iTransformer params: {best['params']:,}")
    print(f"Difference: {best['diff']:+,} ({best['ratio']:.2%} of Informer)")


if __name__ == '__main__':
    main()
