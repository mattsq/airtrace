"""Timer: Generative Pre-trained Transformers for Time Series.

Timer is a decoder-only pre-trained Transformer for general time series analysis,
introduced in "Timer: Generative Pre-trained Transformers Are Large Time Series Models"
(ICML 2024). The model uses Single-Series Sequence (S3) format to handle diverse
time series with a unified GPT-style architecture.

Key features:
- Zero-shot forecasting capability (no training required)
- Pre-trained on 260B time points across multiple domains
- Flexible context lengths and prediction horizons
- Support for fine-tuning with LoRA adapters

Paper: https://arxiv.org/abs/2402.02368
Code: https://github.com/thuml/Large-Time-Series-Model
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register

LOGGER = logging.getLogger(__name__)


class TimerInputNormalizer(nn.Module):
    """Normalizes time series inputs using z-score normalization.

    Timer expects normalized inputs for stable generation. This module
    computes mean and std per series and applies z-score normalization.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize input and return normalization statistics.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Tuple of (normalized_x, mean, std) where:
                - normalized_x: [B, T, D] z-score normalized
                - mean: [B, 1, D] mean per series
                - std: [B, 1, D] std per series
        """
        # Compute statistics along time dimension
        mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        std = x.std(dim=1, keepdim=True)  # [B, 1, D]
        std = torch.clamp(std, min=self.eps)  # Avoid division by zero

        normalized = (x - mean) / std
        return normalized, mean, std

    def inverse(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize predictions using stored statistics.

        Args:
            x: Normalized predictions [B, T, D]
            mean: Mean used for normalization [B, 1, D]
            std: Std used for normalization [B, 1, D]

        Returns:
            Denormalized predictions [B, T, D]
        """
        return x * std + mean


@register("timer")
class TimerModel(ARBaseModel):
    """Timer foundation model for time series forecasting.

    This is a wrapper around the HuggingFace Timer model that adapts it to
    AirTrace's ARBaseModel interface. Timer is pre-trained on 260B time points
    and can perform zero-shot forecasting or be fine-tuned for specific tasks.

    Architecture:
        - Decoder-only Transformer (GPT-style)
        - Single-Series Sequence (S3) format
        - Autoregressive token generation

    Multivariate Handling:
        Timer is trained on univariate series. For multivariate inputs, we
        process each dimension independently and aggregate predictions.

    Args:
        input_dim: Number of input features/sensors
        output_dim: Number of output features to predict
        pred_len: Forecast horizon in timesteps
        checkpoint: HuggingFace model ID or local path to checkpoint
        lookback_length: Context window size (default: 512)
        normalize_inputs: Whether to z-score normalize inputs (recommended: True)
        freeze_backbone: Freeze Timer backbone for zero-shot or LoRA fine-tuning
        lora_rank: LoRA adapter rank (0 disables LoRA)
        lora_alpha: LoRA scaling factor
        lora_dropout: LoRA dropout rate
        device: Device to load model on ('cpu' or 'cuda')
        trust_remote_code: Allow loading custom HuggingFace code (required for Timer)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 24,
        checkpoint: str = "thuml/timer-base-84m",
        lookback_length: int = 512,
        normalize_inputs: bool = True,
        freeze_backbone: bool = False,
        lora_rank: int = 0,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.05,
        device: str = "cpu",
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)

        if not trust_remote_code:
            warnings.warn(
                "Timer requires trust_remote_code=True to load from HuggingFace. "
                "Setting trust_remote_code=True.",
                UserWarning,
            )
            trust_remote_code = True

        self.pred_len = pred_len
        self.checkpoint = checkpoint
        self.lookback_length = lookback_length
        self.normalize_inputs = normalize_inputs
        self.freeze_backbone = freeze_backbone
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device_str = device

        # Normalization module (applied per variate)
        self.normalizer = TimerInputNormalizer() if normalize_inputs else None

        # Load Timer backbone from HuggingFace
        self.timer_backbone = self._load_timer_backbone(
            checkpoint, trust_remote_code
        )

        # Apply LoRA adapters if requested
        if lora_rank > 0:
            self._apply_lora_adapters()

        # Freeze backbone if requested (for zero-shot or LoRA fine-tuning)
        if freeze_backbone:
            self._freeze_backbone()

        LOGGER.info(
            f"Loaded Timer model from {checkpoint} "
            f"({self.get_num_params():,} total params, "
            f"{self._count_trainable_params():,} trainable)"
        )

    def _load_timer_backbone(
        self, checkpoint: str, trust_remote_code: bool
    ) -> nn.Module:
        """Load Timer model from HuggingFace.

        Args:
            checkpoint: HuggingFace model ID or local path
            trust_remote_code: Whether to trust remote code execution

        Returns:
            Loaded Timer model
        """
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as e:
            raise ImportError(
                "Timer requires the 'transformers' package. "
                "Install it with: pip install transformers>=4.40.1"
            ) from e

        LOGGER.info(f"Loading Timer checkpoint from {checkpoint}...")

        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=trust_remote_code,
            device_map=self.device_str,
        )

        return model

    def _apply_lora_adapters(self) -> None:
        """Apply LoRA adapters to Timer backbone for efficient fine-tuning.

        Note: This is a placeholder for future LoRA implementation.
        Full implementation would use peft library.
        """
        if self.lora_rank > 0:
            warnings.warn(
                "LoRA adapters are not yet fully implemented for Timer. "
                "Set lora_rank=0 to disable this warning.",
                UserWarning,
            )

    def _freeze_backbone(self) -> None:
        """Freeze Timer backbone parameters for zero-shot or adapter fine-tuning."""
        for name, param in self.timer_backbone.named_parameters():
            # Keep LoRA parameters trainable if present
            if "lora" not in name.lower():
                param.requires_grad = False

        LOGGER.info("Froze Timer backbone parameters")

    def _count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _process_univariate_series(
        self, series: torch.Tensor, pred_len: int
    ) -> torch.Tensor:
        """Process a single univariate series through Timer.

        Args:
            series: Univariate series [B, T]
            pred_len: Number of steps to forecast

        Returns:
            Predictions [B, pred_len]
        """
        # Timer expects inputs as [B, T] for univariate
        # Use the model's generate method for autoregressive forecasting
        try:
            with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
                # Generate predictions autoregressively
                output = self.timer_backbone.generate(
                    series,
                    max_new_tokens=pred_len,
                )

                # Extract only the generated tokens (last pred_len positions)
                predictions = output[:, -pred_len:]

        except Exception as e:
            # Fallback: if generate fails, use forward pass
            LOGGER.warning(
                f"Timer generate failed ({e}), falling back to forward pass"
            )
            output = self.timer_backbone(series)
            # Extract logits and take last pred_len positions
            if hasattr(output, 'logits'):
                predictions = output.logits[:, -pred_len:]
            else:
                predictions = output[:, -pred_len:]

        return predictions

    def _process_multivariate(
        self, x: torch.Tensor, pred_len: int
    ) -> torch.Tensor:
        """Process multivariate input by handling each dimension independently.

        Timer is trained on univariate series, so we process each input
        dimension separately and stack the results.

        Args:
            x: Multivariate input [B, T, D]
            pred_len: Forecast horizon

        Returns:
            Predictions [B, pred_len, D]
        """
        B, T, D = x.shape
        predictions_per_dim = []

        # Process each dimension independently
        for d in range(D):
            # Extract dimension [B, T]
            series_d = x[:, :, d]

            # Get predictions for this dimension [B, pred_len]
            preds_d = self._process_univariate_series(series_d, pred_len)

            predictions_per_dim.append(preds_d)

        # Stack predictions [B, pred_len, D]
        predictions = torch.stack(predictions_per_dim, dim=-1)

        return predictions

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through Timer model.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context (not used by Timer)
            **kwargs: Additional arguments (not used)

        Returns:
            Dictionary containing:
                - preds: Predictions [B, pred_len, D_out]
                - extras: Dictionary with additional outputs
                    - normalization_stats: (mean, std) if normalization applied
                    - raw_preds: Predictions before denormalization
        """
        del context, kwargs  # Timer doesn't use context

        B, T, D = x.shape

        # Validate input dimensions
        if D != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {D}"
            )

        # Truncate or pad to lookback_length if needed
        if T > self.lookback_length:
            x = x[:, -self.lookback_length:, :]
            LOGGER.debug(f"Truncated input from {T} to {self.lookback_length} steps")
        elif T < self.lookback_length:
            # Pad with zeros at the beginning
            pad_length = self.lookback_length - T
            padding = torch.zeros(
                B, pad_length, D, dtype=x.dtype, device=x.device
            )
            x = torch.cat([padding, x], dim=1)
            LOGGER.debug(f"Padded input from {T} to {self.lookback_length} steps")

        # Normalize inputs if enabled
        normalization_stats = None
        if self.normalize_inputs and self.normalizer is not None:
            x, mean, std = self.normalizer(x)
            normalization_stats = (mean, std)

        # Process multivariate input
        # Timer processes each dimension independently
        preds_normalized = self._process_multivariate(x, self.pred_len)

        # Select output dimensions (support for different input/output dims)
        if self.output_dim != self.input_dim:
            # Take first output_dim dimensions
            preds_normalized = preds_normalized[:, :, :self.output_dim]

        # Denormalize predictions if normalization was applied
        if self.normalize_inputs and normalization_stats is not None:
            mean, std = normalization_stats
            # Adjust stats to match output dimensions
            if self.output_dim != self.input_dim:
                mean = mean[:, :, :self.output_dim]
                std = std[:, :, :self.output_dim]
            preds = self.normalizer.inverse(preds_normalized, mean, std)
        else:
            preds = preds_normalized

        # Prepare extras
        extras = {
            "raw_preds": preds_normalized,
        }
        if normalization_stats is not None:
            extras["normalization_mean"] = normalization_stats[0]
            extras["normalization_std"] = normalization_stats[1]

        return {"preds": preds, "extras": extras}

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  checkpoint={self.checkpoint},\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pred_len={self.pred_len},\n"
            f"  lookback_length={self.lookback_length},\n"
            f"  normalize_inputs={self.normalize_inputs},\n"
            f"  freeze_backbone={self.freeze_backbone},\n"
            f"  total_params={self.get_num_params():,},\n"
            f"  trainable_params={self._count_trainable_params():,}\n"
            f")"
        )
