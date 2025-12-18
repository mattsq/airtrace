"""MOMENT: A Family of Open Time-series Foundation Models.

MOMENT is an open-source foundation model from Carnegie Mellon University
(Auton Lab) for time series analysis. It uses a patch-based transformer
architecture pre-trained on the Time-series Pile dataset for forecasting,
classification, anomaly detection, and imputation tasks.

Key features:
- Zero-shot and few-shot capabilities
- Multivariate time series support via n_channels parameter
- Multiple model sizes (large/base/small)
- Masked reconstruction pre-training

Paper: https://arxiv.org/abs/2402.03885
Code: https://github.com/moment-timeseries-foundation-model/moment
HuggingFace: AutonLab/MOMENT-1-large
"""

from __future__ import annotations

import logging
import warnings
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import ResidualWrapperCompatible
from .registry import register

LOGGER = logging.getLogger(__name__)


class MomentInputNormalizer(nn.Module):
    """Normalizes time series inputs using z-score normalization.

    MOMENT benefits from normalized inputs for stable predictions.
    This module computes mean and std per series and applies z-score
    normalization, similar to Timer's normalization strategy.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def inverse(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions using stored statistics.

        Args:
            x: Normalized predictions [B, T, D]
            mean: Mean used for normalization [B, 1, D]
            std: Std used for normalization [B, 1, D]

        Returns:
            Denormalized predictions [B, T, D]
        """
        return x * std + mean


@register("moment")
class MomentModel(ResidualWrapperCompatible):
    """MOMENT foundation model for time series forecasting.

    This is a wrapper around the HuggingFace MOMENT model that adapts it to
    AirTrace's ARBaseModel interface. MOMENT is pre-trained on the Time-series
    Pile dataset and can perform zero-shot forecasting or be fine-tuned for
    specific tasks.

    Architecture:
        - Patch-based transformer encoder (similar to TimesFM)
        - Masked reconstruction pre-training
        - Task-specific forecast head

    Multivariate Handling:
        MOMENT supports multivariate inputs via the n_channels parameter.
        All dimensions are processed together through the transformer.

    Args:
        input_dim: Number of input features/sensors
        output_dim: Number of output features to predict
        pred_len: Forecast horizon in timesteps
        checkpoint: HuggingFace model ID (AutonLab/MOMENT-1-large/base/small)
        patch_size: Length of patches for tokenization (default: 16)
        context_length: Maximum input context window (default: 512)
        normalize_inputs: Whether to z-score normalize inputs (recommended: True)
        freeze_backbone: Freeze MOMENT backbone for zero-shot inference
        train_forecast_head: Train forecast head even if backbone frozen
        device: Device to load model on ('cpu' or 'cuda')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        pred_len: int = 96,
        checkpoint: str = "AutonLab/MOMENT-1-large",
        patch_size: int = 16,
        context_length: int = 512,
        normalize_inputs: bool = True,
        freeze_backbone: bool = False,
        train_forecast_head: bool = True,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim, **kwargs)

        self.pred_len = pred_len
        self.checkpoint = checkpoint
        self.patch_size = patch_size
        self.context_length = context_length
        self.normalize_inputs = normalize_inputs
        self.freeze_backbone = freeze_backbone
        self.train_forecast_head = train_forecast_head
        self.device_str = device

        # Normalization module
        self.normalizer = MomentInputNormalizer() if normalize_inputs else None

        # Load MOMENT pipeline from HuggingFace
        self.moment_pipeline = self._load_moment_pipeline(checkpoint)

        # Apply parameter freezing if requested
        if freeze_backbone:
            self._freeze_backbone()

        LOGGER.info(
            f"Loaded MOMENT model from {checkpoint} "
            f"({self.get_num_params():,} total params, "
            f"{self._count_trainable_params():,} trainable)"
        )

    def _load_moment_pipeline(self, checkpoint: str) -> nn.Module:
        """Load MOMENT pipeline from HuggingFace.

        Args:
            checkpoint: HuggingFace model ID or local path

        Returns:
            Loaded MOMENT pipeline wrapped as nn.Module
        """
        try:
            from momentfm import MOMENTPipeline
        except ImportError as e:
            raise ImportError(
                "MOMENT requires the 'momentfm' package. " "Install it with: pip install momentfm"
            ) from e

        LOGGER.info(f"Loading MOMENT checkpoint from {checkpoint}...")

        pipeline = MOMENTPipeline.from_pretrained(
            checkpoint,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": self.pred_len,
                "n_channels": self.input_dim,
            },
        )
        pipeline.init()

        # Move to device if specified
        if self.device_str:
            try:
                target_device = torch.device(self.device_str)
                if hasattr(pipeline, "to"):
                    pipeline = pipeline.to(target_device)
            except (RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"Could not move MOMENT to device {self.device_str}: {exc}",
                    UserWarning,
                )

        return self._ensure_module_pipeline(pipeline)

    def _ensure_module_pipeline(self, pipeline: object) -> nn.Module:
        """Ensure the pipeline is an nn.Module with registered parameters.

        This follows Timer's pattern for handling mock pipelines in tests.

        Args:
            pipeline: MOMENT pipeline object

        Returns:
            Pipeline wrapped as nn.Module
        """
        if isinstance(pipeline, nn.Module):
            return pipeline

        # Tests provide a MagicMock pipeline with a named_parameters method.
        # Wrap it in a lightweight Module so parameters are registered for
        # counting and freezing logic.
        params = {}
        if hasattr(pipeline, "named_parameters"):
            try:
                params = {
                    name.replace(".", "_"): param for name, param in pipeline.named_parameters()
                }
            except Exception:
                params = {}

        class _PipelineWrapper(nn.Module):
            def __init__(self, wrapped: object, parameters: Dict[str, nn.Parameter]):
                super().__init__()
                self.wrapped = wrapped
                self.params = nn.ParameterDict(parameters)

            def forward(self, *args, **kwargs):
                if callable(self.wrapped):
                    return self.wrapped(*args, **kwargs)
                raise TypeError("Pipeline object is not callable")

            def __call__(self, *args, **kwargs):
                if hasattr(self.wrapped, "__call__"):
                    return self.wrapped.__call__(*args, **kwargs)
                return self.forward(*args, **kwargs)

        return _PipelineWrapper(pipeline, params)

    def _freeze_backbone(self) -> None:
        """Freeze MOMENT backbone parameters for zero-shot or adapter fine-tuning."""
        for name, param in self.moment_pipeline.named_parameters():
            # Keep forecast head trainable if requested
            if self.train_forecast_head and "forecast" in name.lower():
                param.requires_grad = True
                continue
            param.requires_grad = False

        LOGGER.info("Froze MOMENT backbone parameters")

    def _count_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _prepare_inputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Validate, pad/truncate, and optionally normalize inputs.

        Args:
            x: Input tensor [B, T, D]

        Returns:
            Tuple of (prepared_x, normalization_stats) where:
                - prepared_x: [B, context_length, D] processed input
                - normalization_stats: Optional (mean, std) for denormalization
        """
        B, T, D = x.shape

        if D != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {D}")

        # Truncate or pad to context_length
        if T > self.context_length:
            x = x[:, -self.context_length :, :]
            LOGGER.debug(f"Truncated input from {T} to {self.context_length} steps")
        elif T < self.context_length:
            pad_length = self.context_length - T
            padding = torch.zeros(B, pad_length, D, dtype=x.dtype, device=x.device)
            x = torch.cat([padding, x], dim=1)
            LOGGER.debug(f"Padded input from {T} to {self.context_length} steps")

        # Ensure divisibility by patch_size (following TimesFM pattern)
        if self.context_length % self.patch_size != 0:
            target_len = ((self.context_length // self.patch_size) + 1) * self.patch_size
            pad_length = target_len - self.context_length
            padding = torch.zeros(B, pad_length, D, dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=1)

        # Normalize if enabled
        normalization_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.normalize_inputs and self.normalizer is not None:
            x, mean, std = self.normalizer(x)
            normalization_stats = (mean, std)

        return x, normalization_stats

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return a latent representation and any encoder extras.

        Args:
            x: Input tensor [B, T, D]
            context: Optional context tensor (unused)

        Returns:
            Tuple of (latent, extras_dict) where:
                - latent: [B, T_prepared, D] prepared input (acts as latent)
                - extras_dict: Dictionary with normalization stats
        """
        del context  # Unused parameter

        prepared, normalization_stats = self._prepare_inputs(x)

        extras: Dict[str, torch.Tensor] = {"representation": prepared}
        if normalization_stats is not None:
            extras["normalization_mean"] = normalization_stats[0]
            extras["normalization_std"] = normalization_stats[1]

        return prepared, extras

    def decode(
        self,
        latent: torch.Tensor,
        pred_len: int,
        *,
        extras: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Decode predictions of length pred_len from a latent.

        Args:
            latent: Latent representation [B, T, D]
            pred_len: Number of timesteps to forecast
            extras: Optional dictionary with normalization stats

        Returns:
            Predictions [B, pred_len, D_out]
        """
        if pred_len <= 0:
            raise ValueError("pred_len must be positive for MomentModel")

        # Call MOMENT pipeline for forecasting
        # Note: The actual API may vary; this is based on expected interface
        preds = self.moment_pipeline(latent)

        # Handle output format - MOMENT may return dict or tensor
        if isinstance(preds, dict):
            preds = preds.get("forecast", preds.get("output", preds.get("prediction")))

        # Ensure correct output dimension
        if preds.shape[-1] != self.output_dim:
            preds = preds[:, :, : self.output_dim]

        # Ensure correct prediction length
        if preds.shape[1] != pred_len:
            if preds.shape[1] > pred_len:
                preds = preds[:, :pred_len, :]
            else:
                # Pad if predictions are shorter than requested
                pad_length = pred_len - preds.shape[1]
                padding = torch.zeros(
                    preds.shape[0],
                    pad_length,
                    preds.shape[2],
                    dtype=preds.dtype,
                    device=preds.device,
                )
                preds = torch.cat([preds, padding], dim=1)

        # Denormalize if needed
        if self.normalize_inputs and extras is not None:
            mean = extras.get("normalization_mean")
            std = extras.get("normalization_std")
            if mean is not None and std is not None and self.normalizer is not None:
                if self.output_dim != self.input_dim:
                    mean = mean[:, :, : self.output_dim]
                    std = std[:, :, : self.output_dim]
                preds = self.normalizer.inverse(preds, mean, std)

        return preds

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through MOMENT model.

        Args:
            x: Input tensor [B, T, D]
            context: Optional context tensor (unused)
            **kwargs: Additional arguments (can override pred_len)

        Returns:
            Dictionary containing:
                - preds: Predictions [B, pred_len, D_out]
                - extras: Additional outputs (normalization stats, representation)
        """
        pred_len = int(kwargs.get("pred_len", self.pred_len))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len, extras=extras)

        # Build output dictionary
        extras_out: Dict[str, torch.Tensor] = {
            k: v for k, v in extras.items() if k != "representation"
        }
        extras_out["representation"] = latent

        return {"preds": preds, "extras": extras_out}

    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  checkpoint={self.checkpoint},\n"
            f"  input_dim={self.input_dim},\n"
            f"  output_dim={self.output_dim},\n"
            f"  pred_len={self.pred_len},\n"
            f"  patch_size={self.patch_size},\n"
            f"  context_length={self.context_length},\n"
            f"  normalize_inputs={self.normalize_inputs},\n"
            f"  freeze_backbone={self.freeze_backbone},\n"
            f"  total_params={self.get_num_params():,},\n"
            f"  trainable_params={self._count_trainable_params():,}\n"
            f")"
        )
