# Model Notes

## Chronos-Bolt Foundation Forecaster

Chronos-Bolt is a pretrained continuous-token foundation model that mixes dilated depth-wise
convolutions with gated self-attention. The AirTrace implementation keeps the core components:

- **Tokenizer-free patching** via a 1-D convolutional stem that embeds overlapping context
  windows without any discrete quantization.
- **Gated convolution-attention blocks** that alternate dilated depth-wise convolutions with
  multi-head self-attention plus lightweight feed-forward mixers.
- **Optional LoRA adapters** so teams can fine-tune only a few trainable parameters while
  freezing the pretrained backbone.

### Using the model

1. Download a checkpoint (or create your own) using the helper script:
   ```bash
   python src/scripts/download_chronos_bolt.py --output-dir checkpoints
   ```
2. Point `configs/model/chronos_bolt.yaml:model.params.pretrained_checkpoint` to the downloaded
   `.pt` file.
3. Launch experiments as usual, for example:
   ```bash
   airtrace train model=chronos_bolt data=qantas_737 transforms=zscore task=multi_step
   ```

### Zero-shot vs. fine-tuning

| Mode | Key config knobs | Notes |
|------|------------------|-------|
| **Zero-shot** | `freeze_backbone=true`, `train_head=false`, `lora_rank=0` | Best when you simply want the pretrained forecast. Only the regression head (if enabled) will update. |
| **LoRA fine-tune** | `freeze_backbone=true`, `lora_rank>0`, `train_head=true` | Keeps the heavy backbone frozen but trains lightweight low-rank adapters and the forecast head. Ideal for small datasets. |
| **Full fine-tune** | `freeze_backbone=false` | Allows end-to-end updates. Use lower learning rates to avoid destroying the pretrained features. |

Chronos-Bolt expects z-score normalized data. Ensure your transform stack stores the mean and
standard deviation alongside processed tensors so inference can replay the same normalization
statistics.
