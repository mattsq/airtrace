# ONNX Export Example

This example demonstrates how to export a trained AirTrace model to ONNX and use it for inference.

## Prerequisites

```bash
pip install onnxruntime numpy
```

## Step 1: Train a Model

First, train a model (or use an existing checkpoint):

```bash
airtrace train exp=exp_001_gru_zscore train.epochs=10
```

This creates a checkpoint at `runs/<date>/<exp_name>/checkpoints/best.ckpt`

## Step 2: Export to ONNX

### Option A: Model-Only Export

Export just the model:

```bash
airtrace export onnx \
  --checkpoint runs/20250115/exp_001_gru_zscore/checkpoints/best.ckpt \
  --output exports/gru_model.onnx
```

**Output files:**
- `exports/gru_model.onnx` - The model
- `exports/gru_model.transform_stats.json` - Transform statistics
- `exports/gru_model.config.yaml` - Training config
- `exports/gru_model.metadata.json` - Shape metadata

### Option B: End-to-End Export

Export with integrated inverse transforms:

```bash
airtrace export onnx \
  --checkpoint runs/20250115/exp_001_gru_zscore/checkpoints/best.ckpt \
  --output exports/gru_e2e.onnx \
  --end-to-end
```

## Step 3: Inference with ONNX Runtime

### Using Model-Only Export

```python
"""Inference with model-only export."""

import json
import numpy as np
import onnxruntime as ort


def load_transform_stats(path):
    """Load transform statistics."""
    with open(path) as f:
        return json.load(f)


def preprocess(data, stats):
    """Apply z-score normalization."""
    transform = stats.get('ZScoreTransform_0', {})
    mean = np.array(transform['scaler_x_mean'])
    std = np.array(transform['scaler_x_scale'])
    return (data - mean) / (std + 1e-8)


def postprocess(predictions, stats):
    """Apply inverse z-score normalization."""
    transform = stats.get('ZScoreTransform_0', {})
    mean = np.array(transform['scaler_y_mean'])
    std = np.array(transform['scaler_y_scale'])
    return predictions * (std + 1e-8) + mean


def main():
    # Load model and stats
    session = ort.InferenceSession("exports/gru_model.onnx")
    stats = load_transform_stats("exports/gru_model.transform_stats.json")

    # Load metadata to check input shape
    with open("exports/gru_model.metadata.json") as f:
        metadata = json.load(f)

    print(f"Expected input shape: {metadata['input_shape']}")
    print(f"Output dimension: {metadata['output_dim']}")

    # Create sample input (replace with your actual data)
    batch_size = 1
    seq_length = 100
    n_features = 15
    input_data = np.random.randn(batch_size, seq_length, n_features).astype(np.float32)

    print(f"\nInput shape: {input_data.shape}")

    # Preprocess input
    input_preprocessed = preprocess(input_data, stats)

    # Run inference
    outputs = session.run(
        None,
        {"input": input_preprocessed, "context": None}
    )
    predictions_transformed = outputs[0]

    print(f"Raw prediction shape: {predictions_transformed.shape}")

    # Postprocess to original scale
    predictions = postprocess(predictions_transformed, stats)

    print(f"Final prediction shape: {predictions.shape}")
    print(f"\nSample predictions (first 5 features):")
    print(predictions[0, 0, :5])


if __name__ == "__main__":
    main()
```

### Using End-to-End Export

```python
"""Inference with end-to-end export (much simpler!)."""

import json
import numpy as np
import onnxruntime as ort


def main():
    # Load model
    session = ort.InferenceSession("exports/gru_e2e.onnx")

    # Load metadata
    with open("exports/gru_e2e.metadata.json") as f:
        metadata = json.load(f)

    print(f"Expected input shape: {metadata['input_shape']}")

    # Create sample RAW input (replace with your actual sensor data)
    # This should be your raw, unnormalized sensor readings
    batch_size = 1
    seq_length = 100
    n_features = 15
    raw_sensor_data = np.random.randn(batch_size, seq_length, n_features).astype(np.float32)

    print(f"\nInput shape: {raw_sensor_data.shape}")

    # Run inference - the model handles ALL preprocessing internally!
    # 1. Applies z-score normalization (or other transforms)
    # 2. Runs the model
    # 3. Applies inverse transforms
    outputs = session.run(
        None,
        {"input": raw_sensor_data, "context": None}
    )
    predictions = outputs[0]

    # Predictions are already in original scale!
    print(f"Prediction shape: {predictions.shape}")
    print(f"\nSample predictions (first 5 features):")
    print(predictions[0, 0, :5])


if __name__ == "__main__":
    main()
```

## Step 4: Batch Inference

Process multiple samples efficiently:

```python
"""Batch inference example."""

import numpy as np
import onnxruntime as ort


def batch_inference(session, data_batch):
    """
    Run inference on a batch of samples.

    Args:
        session: ONNX Runtime session
        data_batch: np.ndarray of shape [batch_size, seq_length, n_features]

    Returns:
        Predictions of shape [batch_size, output_length, output_dim]
    """
    outputs = session.run(
        None,
        {"input": data_batch.astype(np.float32), "context": None}
    )
    return outputs[0]


def main():
    session = ort.InferenceSession("exports/gru_e2e.onnx")

    # Simulate batch of 10 samples
    batch_size = 10
    seq_length = 100
    n_features = 15

    data_batch = np.random.randn(batch_size, seq_length, n_features)

    print(f"Processing batch of {batch_size} samples...")

    predictions = batch_inference(session, data_batch)

    print(f"Output shape: {predictions.shape}")
    print(f"Predictions per sample: {predictions.shape[1]}")


if __name__ == "__main__":
    main()
```

## Step 5: Production Deployment

### Flask API Example

```python
"""Simple Flask API for model serving."""

from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Load model once at startup
session = ort.InferenceSession("exports/gru_e2e.onnx")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint.

    Expected JSON format:
    {
        "data": [[...], [...], ...]  # List of lists: [seq_length, n_features]
    }
    """
    try:
        # Parse input
        data = request.json['data']
        input_array = np.array(data, dtype=np.float32)

        # Add batch dimension if needed
        if input_array.ndim == 2:
            input_array = input_array[np.newaxis, :, :]

        # Run inference
        outputs = session.run(
            None,
            {"input": input_array, "context": None}
        )
        predictions = outputs[0]

        # Return predictions
        return jsonify({
            "predictions": predictions.tolist(),
            "shape": predictions.shape
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run the API:

```bash
python api.py
```

Test it:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.0, 2.0, 3.0, ...], ...]}'
```

## Performance Benchmarking

```python
"""Benchmark ONNX inference performance."""

import time
import numpy as np
import onnxruntime as ort


def benchmark(session, input_shape, num_iterations=100):
    """Benchmark inference speed."""
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {"input": input_data, "context": None})

    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        session.run(None, {"input": input_data, "context": None})
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations * 1000  # ms
    throughput = num_iterations / (end_time - start_time)  # samples/sec

    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {throughput:.2f} samples/sec")


def main():
    session = ort.InferenceSession("exports/gru_e2e.onnx")

    input_shape = (1, 100, 15)  # [batch, seq_length, features]

    print(f"Benchmarking with input shape {input_shape}...")
    benchmark(session, input_shape)


if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

**Issue: "Input shape mismatch"**
- Check `metadata.json` for expected input shape
- Ensure your data has the correct dimensions: `[batch, seq_length, n_features]`

**Issue: "Large prediction errors"**
- Verify transform statistics are correctly applied
- Check that input data is in the expected range

**Issue: "Slow inference"**
- Use batching to process multiple samples at once
- Enable ONNX Runtime optimizations (see docs/onnx_export.md)

## Next Steps

- Deploy with Docker: Create a containerized API
- Add monitoring: Track inference latency and errors
- Scale horizontally: Deploy multiple instances
- Explore quantization: Reduce model size and improve speed

## Additional Resources

- [Full ONNX Export Documentation](../onnx_export.md)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [AirTrace Architecture](../architecture.md)
