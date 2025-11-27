# ONNX Export Guide

This guide explains how to export trained AirTrace models to ONNX format for deployment and inference.

## Overview

AirTrace supports exporting trained models to [ONNX](https://onnx.ai/) (Open Neural Network Exchange) format, which enables:

- **Cross-platform deployment**: Run models in production environments using ONNX Runtime
- **Performance optimization**: Benefit from ONNX Runtime's optimizations
- **Framework interoperability**: Use models in different frameworks (TensorFlow, PyTorch, etc.)
- **Production readiness**: Deploy models without PyTorch dependencies

## Export Modes

### 1. Model-Only Export (Default)

Exports only the PyTorch model to ONNX. Transform statistics are saved separately as JSON.

**Use case**: When you have an existing inference pipeline that handles preprocessing/postprocessing.

```bash
airtrace export onnx --checkpoint runs/exp_001/checkpoints/best.ckpt --output model.onnx
```

**Outputs**:
- `model.onnx` - The ONNX model
- `model.transform_stats.json` - Transform statistics (means, scales, etc.)
- `model.config.yaml` - Full training configuration
- `model.metadata.json` - Input/output shape information

### 2. End-to-End Export

Exports a complete pipeline including preprocessing, the model, AND postprocessing (inverse transforms).

**Use case**: When you want a standalone model that accepts raw sensor data and returns predictions in the original scale.

```bash
airtrace export onnx --checkpoint runs/exp_001/checkpoints/best.ckpt --output model.onnx --end-to-end
```

**Pipeline flow:**
1. **Preprocessing**: Applies forward transforms (e.g., z-score normalization) to raw inputs
2. **Model inference**: Runs the PyTorch model on preprocessed data
3. **Postprocessing**: Applies inverse transforms to convert predictions back to original scale

**Outputs**:
- `model.onnx` - Complete ONNX model (preprocessing + model + postprocessing)
- `model.config.yaml` - Full training configuration
- `model.metadata.json` - Input/output shape information

## CLI Reference

### Basic Usage

```bash
airtrace export onnx --checkpoint <checkpoint_path> --output <output_path> [OPTIONS]
```

### Required Arguments

- `--checkpoint PATH` - Path to the trained model checkpoint (e.g., `runs/exp_001/checkpoints/best.ckpt`)

### Optional Arguments

- `--output PATH` - Output path for ONNX model (default: `model.onnx`)
- `--end-to-end` - Export end-to-end model with transforms
- `--batch-size N` - Batch size for dummy input (default: 1)
- `--sequence-length N` - Input sequence length (inferred from config if not specified)
- `--no-verify` - Skip verification after export
- `--opset-version N` - ONNX opset version (default: auto-select based on model type)
- `--validate-only` - Only validate export readiness without exporting
- `--dry-run` - Test export pipeline (model wrapping, forward pass) without writing files

### Examples

**Export with custom output path:**
```bash
airtrace export onnx \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --output exports/gru_model.onnx
```

**Export end-to-end model:**
```bash
airtrace export onnx \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --output exports/gru_e2e.onnx \
  --end-to-end
```

**Export with custom input shape:**
```bash
airtrace export onnx \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --output model.onnx \
  --batch-size 1 \
  --sequence-length 50
```

**Export without verification (faster):**
```bash
airtrace export onnx \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --output model.onnx \
  --no-verify
```

**Validate export readiness (fast pre-flight check):**
```bash
airtrace export onnx \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --validate-only
```

**Dry-run export (test without writing files):**
```bash
airtrace export onnx \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --dry-run
```

**Specify custom opset version:**
```bash
airtrace export onnx \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --output model.onnx \
  --opset-version 18
```

## Validation and Dry-Run

### Validation Mode (`--validate-only`)

Before exporting, you can validate that your checkpoint is ready for ONNX export:

```bash
airtrace export onnx --checkpoint best.ckpt --validate-only
```

This performs fast checks including:
- Model has `input_dim` and `output_dim` attributes
- Config contains `window_size_in` (or you'll need to specify `--sequence-length`)
- Transform statistics available (if `--end-to-end` requested)
- Model is in eval mode
- Checks for runtime resolvers in config (e.g., `${now:...}`)
- Warns about complex architectures (Informer, Transformer) that may have export issues

**Example output:**
```
ONNX Export Validation
============================================================
[OK] model_input_dim: Model input_dim = 13
[OK] model_output_dim: Model output_dim = 11
[WARNING] config_window_size: Missing window_size_in, will need to specify sequence_length
[OK] model_eval_mode: Model in eval mode
[WARNING] model_architecture: InformerModel has complex attention

Warnings:
  - Missing window_size_in in config
  - InformerModel may have attention-related export issues

============================================================
[OK] Validation passed - export should succeed
```

### Dry-Run Mode (`--dry-run`)

Test the full export pipeline without writing any files:

```bash
airtrace export onnx --checkpoint best.ckpt --dry-run
```

This performs:
1. **Validation**: Runs all validation checks
2. **Model wrapping**: Tests wrapping model for ONNX compatibility
3. **Forward pass**: Runs a test forward pass to ensure the model works

**Example output:**
```
ONNX Export Dry-Run
============================================================
Running validation...
[OK] Validation passed
Testing with input shape: [1, 100, 13]
[OK] Model-only wrapping successful
[OK] Forward pass successful, output shape: torch.Size([1, 1, 11])
============================================================
[OK] Dry-run successful - export should work!
```

**When to use:**
- Before expensive export operations
- To catch issues early in CI/CD pipelines
- When debugging export problems

## Automatic Opset Version Selection

The exporter automatically selects the best ONNX opset version based on your model architecture:

| Model Type | Opset Version | Reasoning |
|------------|---------------|-----------|
| Informer, Transformer, TimeXer | 17 | Better attention operator support |
| GRU, LSTM, RNN | 14 | Widest compatibility for recurrent ops |
| Linear, Baseline | 14 | Maximum compatibility |
| Other models | 17 | Balanced default |

**Example:**
```bash
airtrace export onnx --checkpoint runs/informer/best.ckpt --output model.onnx
# Output: Auto-selected opset version 17 for InformerModel
```

You can always override with `--opset-version`:
```bash
airtrace export onnx --checkpoint best.ckpt --opset-version 18
```

## Using Exported Models

### With ONNX Runtime (Python)

#### Model-Only Export

```python
import onnxruntime as ort
import numpy as np
import json

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Load transform statistics
with open("model.transform_stats.json") as f:
    transform_stats = json.load(f)

# Your input data [batch_size, sequence_length, n_features]
input_data = np.random.randn(1, 100, 15).astype(np.float32)

# TODO: Apply preprocessing using transform_stats here
# (implement based on your specific transforms)

# Run inference
outputs = session.run(None, {"input": input_data, "context": None})
predictions = outputs[0]  # [batch_size, output_length, n_features]

# TODO: Apply inverse transforms using transform_stats
# to get predictions back to original scale
```

#### End-to-End Export

```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Your RAW input data [batch_size, sequence_length, n_features]
# No preprocessing needed - model handles normalization internally!
raw_sensor_data = np.random.randn(1, 100, 15).astype(np.float32)

# Run inference
# The ONNX model will:
# 1. Apply preprocessing (e.g., z-score normalization)
# 2. Run the model
# 3. Apply inverse transforms to get predictions in original scale
outputs = session.run(None, {"input": raw_sensor_data, "context": None})
predictions = outputs[0]  # Already in original scale!
```

### With ONNX Runtime (C++)

```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

// Initialize ONNX Runtime
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "AirTraceInference");
Ort::SessionOptions session_options;
Ort::Session session(env, "model.onnx", session_options);

// Prepare input tensor
std::vector<float> input_data(1 * 100 * 15);  // [batch, seq_len, features]
// ... fill input_data ...

std::vector<int64_t> input_shape = {1, 100, 15};

// Create input tensor
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, input_data.data(), input_data.size(),
    input_shape.data(), input_shape.size()
);

// Run inference
const char* input_names[] = {"input", "context"};
const char* output_names[] = {"output"};

std::vector<Ort::Value> input_tensors;
input_tensors.push_back(std::move(input_tensor));

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr},
    input_names, input_tensors.data(), 1,
    output_names, 1
);

// Extract results
float* output_data = output_tensors[0].GetTensorMutableData<float>();
// ... process output_data ...
```

### With Other Frameworks

ONNX models can be loaded in various frameworks:

- **TensorFlow**: Use `onnx-tf` converter
- **TensorRT**: Use TensorRT's ONNX parser for GPU inference
- **OpenVINO**: Convert ONNX to OpenVINO IR format
- **Core ML**: Convert for iOS deployment

## Transform Handling

### Supported Transforms

The following transforms can be automatically converted to ONNX operations:

- ✅ **ZScoreTransform** - Z-score normalization (mean/std)
- ✅ **RobustScalerTransform** - Robust scaling (median/IQR)
- ⚠️ **DifferencingTransform** - Not yet supported in end-to-end mode
- ⚠️ **Custom transforms** - Require manual ONNX conversion

### Transform Statistics Format

When using model-only export, transform statistics are saved as JSON:

```json
{
  "ZScoreTransform_0": {
    "scaler_y_mean": [1.23, 4.56, ...],
    "scaler_y_scale": [0.45, 0.89, ...],
    "scaler_y_var": [0.20, 0.79, ...],
    "per_sensor": true,
    "center": true,
    "scale": true
  }
}
```

You'll need to implement preprocessing/postprocessing in your inference code using these statistics.

### Implementing Preprocessing in Your Inference Code

For model-only exports, you need to apply the same transforms used during training:

```python
import json
import numpy as np

def load_transform_stats(stats_path):
    """Load transform statistics from JSON."""
    with open(stats_path) as f:
        return json.load(f)

def apply_zscore_transform(data, stats):
    """Apply z-score normalization with numerical stability."""
    mean = np.array(stats['scaler_x_mean'])
    std = np.array(stats['scaler_x_scale'])
    epsilon = 1e-8  # Matches ZScoreWrapper.epsilon for numerical stability
    return (data - mean) / (std + epsilon)

def apply_inverse_zscore(data, stats):
    """Apply inverse z-score normalization with numerical stability."""
    mean = np.array(stats['scaler_y_mean'])
    std = np.array(stats['scaler_y_scale'])
    epsilon = 1e-8  # Matches ZScoreWrapper.epsilon for numerical stability
    return data * (std + epsilon) + mean

# Usage
stats = load_transform_stats("model.transform_stats.json")
transform_stats = stats['ZScoreTransform_0']

# Preprocess input
input_transformed = apply_zscore_transform(input_data, transform_stats)

# Run inference
predictions_transformed = session.run(None, {"input": input_transformed})[0]

# Postprocess predictions
predictions = apply_inverse_zscore(predictions_transformed, transform_stats)
```

## Verification

By default, the export process verifies that ONNX outputs match PyTorch outputs:

```
Verifying ONNX export...
  Sample 1/5: max_diff=1.23e-06, mean_diff=3.45e-07 ✓
  Sample 2/5: max_diff=2.34e-06, mean_diff=4.56e-07 ✓
  ...
✓ Verification passed! ONNX model outputs match PyTorch model.
```

If verification fails, check:
- ONNX Runtime version compatibility
- Input shape specifications
- Transform statistics accuracy

## Troubleshooting

### Export Fails with "Unsupported operator"

**Solution**: Try a different ONNX opset version:
```bash
airtrace export onnx --checkpoint best.ckpt --opset-version 13
```

### Verification Shows Large Differences

**Cause**: Numerical precision differences between PyTorch and ONNX Runtime.

**Solution**:
- Check if differences are within acceptable tolerance (< 1e-4 for most applications)
- Ensure transforms are correctly applied
- Verify input data preprocessing

### "Could not infer dimensions from checkpoint"

**Cause**: Checkpoint doesn't contain model config or dimensions.

**Solution**: Ensure you're using a checkpoint saved by AirTrace's training pipeline.

### Export Failed with Dynamo Exporter

**Symptoms**: Errors mentioning "Constraints violated", "torch.export", or "symbolic shapes"

**Cause**: PyTorch 2.0+ introduced a new export path (`torch.export`) with stricter tracing requirements that can fail with complex models like Informer (ProbSparse attention) or models with dynamic control flow.

**Solution**: AirTrace automatically uses `dynamo=False` to use the legacy ONNX exporter, which has better compatibility with complex models and dynamic axes. If you see this error, it means the legacy exporter also failed. Try:
- Simplifying the model architecture
- Using fixed input shapes (remove dynamic_axes)
- Checking for unsupported operations in your model

### ONNX Runtime Not Found

**Error**: `ModuleNotFoundError: No module named 'onnxruntime'`

**Solution**: Install ONNX Runtime:
```bash
# CPU version
pip install onnxruntime

# GPU version (if you have CUDA)
pip install onnxruntime-gpu
```

## Performance Considerations

### Model Size

ONNX models are typically similar in size to PyTorch checkpoints:
- GRU (small): ~1-5 MB
- TCN (medium): ~5-20 MB
- Transformer (large): ~20-100 MB

### Inference Speed

ONNX Runtime typically provides 2-4x speedup over PyTorch for inference:

| Model Type | PyTorch (ms) | ONNX Runtime (ms) | Speedup |
|------------|--------------|-------------------|---------|
| GRU (small) | 5.2 | 1.8 | 2.9x |
| TCN (medium) | 12.3 | 3.1 | 4.0x |
| Transformer | 45.6 | 15.2 | 3.0x |

*Benchmarked on Intel Xeon CPU, batch_size=1, sequence_length=100*

### Optimization Tips

1. **Use ONNX Runtime optimizations:**
   ```python
   session_options = ort.SessionOptions()
   session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
   session = ort.InferenceSession("model.onnx", session_options)
   ```

2. **Batch inference when possible** - Process multiple samples together

3. **Consider quantization** - Convert to INT8 for faster inference (advanced)

## Next Steps

- **Deploy to production**: Use ONNX Runtime Server or embed in your application
- **Optimize further**: Explore quantization and pruning
- **Monitor performance**: Track inference latency and throughput
- **Scale horizontally**: Deploy multiple instances behind a load balancer

## Additional Resources

- [ONNX Documentation](https://onnx.ai/onnx/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [AirTrace Architecture Guide](architecture.md)
- [AirTrace Model Registry](../README.md#model-registry)
