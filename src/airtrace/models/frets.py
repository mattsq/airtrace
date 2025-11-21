"""FreTS: Frequency-domain MLPs for Time Series Forecasting.

This module implements FreTS introduced in:
  Yi et al., "FreTS: Frequency-domain MLPs are More Effective Learners in
  Time Series Forecasting" (NeurIPS 2023).

Key design choices for AirTrace:
- Operates in frequency domain via FFT/iFFT transformations
- Applies MLPs to low-frequency Fourier coefficients
- Efficient and parameter-light compared to transformers
- Handles multivariate time series by processing each feature independently
  or with channel mixing
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


class FrequencyMLP(nn.Module):
    """MLP operating on frequency-domain representations.

    Processes complex-valued Fourier coefficients with MLPs.
    """

    def __init__(
        self,
        num_freqs: int,
        d_model: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        """Initialize frequency MLP.

        Args:
            num_freqs: Number of frequency components to process
            d_model: Model dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            dropout: Dropout rate
            activation: Activation function name
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.d_model = d_model

        # MLPs for real and imaginary parts
        layers_real = []
        layers_imag = []

        input_dim = num_freqs
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else num_freqs
            layers_real.append(nn.Linear(input_dim, output_dim))
            layers_imag.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim

        self.mlp_real = nn.ModuleList(layers_real)
        self.mlp_imag = nn.ModuleList(layers_imag)
        self.dropout = nn.Dropout(dropout)

        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """Process frequency-domain representation.

        Args:
            x_freq: Complex frequency tensor [B, num_freqs, D]

        Returns:
            Processed frequency tensor [B, num_freqs, D]
        """
        # Split complex tensor into real and imaginary parts
        x_real = x_freq.real  # [B, num_freqs, D]
        x_imag = x_freq.imag  # [B, num_freqs, D]

        # Process each feature dimension independently
        # Transpose to [B, D, num_freqs] for MLP over frequency dimension
        x_real = x_real.transpose(1, 2)  # [B, D, num_freqs]
        x_imag = x_imag.transpose(1, 2)  # [B, D, num_freqs]

        # Apply MLPs
        for i, (layer_real, layer_imag) in enumerate(zip(self.mlp_real, self.mlp_imag)):
            x_real = layer_real(x_real)
            x_imag = layer_imag(x_imag)

            # Apply activation and dropout (not on last layer)
            if i < len(self.mlp_real) - 1:
                x_real = self.activation(x_real)
                x_imag = self.activation(x_imag)
                x_real = self.dropout(x_real)
                x_imag = self.dropout(x_imag)

        # Transpose back to [B, num_freqs, D]
        x_real = x_real.transpose(1, 2)
        x_imag = x_imag.transpose(1, 2)

        # Reconstruct complex tensor
        x_out = torch.complex(x_real, x_imag)

        return x_out


class ChannelMixing(nn.Module):
    """Optional channel mixing layer for multivariate series."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        """Initialize channel mixing.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix channels.

        Args:
            x: Input tensor [B, T, D_in]

        Returns:
            Output tensor [B, T, D_out]
        """
        return self.dropout(self.linear(x))


@register("frets")
class FreTSModel(ARBaseModel):
    """FreTS: Frequency-domain MLP model for time series forecasting.

    This model:
    1. Transforms input to frequency domain via FFT
    2. Selects low-frequency components (most informative)
    3. Applies MLPs to process frequency coefficients
    4. Transforms back to time domain via inverse FFT
    5. Projects to prediction horizon
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int = 60,
        pred_len: int = 1,
        num_freqs: Optional[int] = None,
        d_model: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        channel_independence: bool = False,
        normalize_fft: bool = True,
    ) -> None:
        """Initialize FreTS model.

        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension
            seq_len: Input sequence length
            pred_len: Prediction horizon
            num_freqs: Number of frequency components to keep (default: seq_len // 2 + 1)
            d_model: Model dimension
            hidden_dim: Hidden dimension for MLPs
            num_layers: Number of MLP layers
            dropout: Dropout rate
            activation: Activation function
            channel_independence: If True, process each channel independently
            normalize_fft: If True, normalize FFT outputs
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.channel_independence = channel_independence
        self.normalize_fft = normalize_fft

        # Number of frequency components (keep low frequencies)
        # FFT of real signal of length seq_len has seq_len // 2 + 1 unique components
        max_freqs = seq_len // 2 + 1
        self.num_freqs = min(num_freqs or max_freqs, max_freqs)

        # Input projection to d_model
        self.input_projection = nn.Linear(input_dim, d_model)

        # Frequency-domain MLP
        self.freq_mlp = FrequencyMLP(
            num_freqs=self.num_freqs,
            d_model=d_model,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

        # Temporal projection from seq_len to pred_len
        # This operates on the time-domain output after iFFT
        self.temporal_projection = nn.Linear(seq_len, pred_len)

        # Output projection
        if not channel_independence and d_model != output_dim:
            self.output_projection = ChannelMixing(d_model, output_dim, dropout)
        elif channel_independence:
            self.output_projection = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor [B, T_in, D_in] where T_in == seq_len
            context: Optional context (not used in FreTS)
            **kwargs: Additional arguments (not used)

        Returns:
            Dictionary with:
                - preds: Predictions [B, T_out, D_out]
                - extras: Additional outputs (frequency components, etc.)
        """
        B, T, D = x.shape

        if T != self.seq_len:
            raise ValueError(
                f"Expected input sequence length {self.seq_len}, got {T}"
            )

        # Project input to d_model
        x = self.input_projection(x)  # [B, T, d_model]

        # Transform to frequency domain using FFT
        # Apply FFT along the time dimension
        x_freq = torch.fft.rfft(x, dim=1, norm="ortho" if self.normalize_fft else None)
        # x_freq: [B, T//2+1, d_model] (complex-valued)

        # Keep only low-frequency components
        x_freq_low = x_freq[:, :self.num_freqs, :]  # [B, num_freqs, d_model]

        # Store original for extras
        x_freq_orig = x_freq_low.clone()

        # Process in frequency domain with MLP
        x_freq_processed = self.freq_mlp(x_freq_low)  # [B, num_freqs, d_model]

        # Pad back to original frequency length if needed
        if self.num_freqs < x_freq.size(1):
            # Pad with zeros for high frequencies
            padding = torch.zeros(
                B,
                x_freq.size(1) - self.num_freqs,
                self.d_model,
                dtype=x_freq_processed.dtype,
                device=x_freq_processed.device,
            )
            x_freq_full = torch.cat([x_freq_processed, padding], dim=1)
        else:
            x_freq_full = x_freq_processed

        # Transform back to time domain using inverse FFT
        x_time = torch.fft.irfft(
            x_freq_full,
            n=self.seq_len,
            dim=1,
            norm="ortho" if self.normalize_fft else None,
        )  # [B, T, d_model]

        # Project from seq_len to pred_len
        # Transpose to [B, d_model, T] for temporal convolution
        x_time = x_time.transpose(1, 2)  # [B, d_model, T]
        preds = self.temporal_projection(x_time)  # [B, d_model, pred_len]
        preds = preds.transpose(1, 2)  # [B, pred_len, d_model]

        # Output projection to target dimension
        if self.output_projection is not None:
            preds = self.output_projection(preds)  # [B, pred_len, output_dim]

        # Prepare extras
        extras = {
            "freq_components_orig": x_freq_orig.abs(),  # [B, num_freqs, d_model]
            "freq_components_processed": x_freq_processed.abs(),  # [B, num_freqs, d_model]
            "time_reconstruction": x_time.transpose(1, 2),  # [B, T, d_model]
        }

        return {"preds": preds, "extras": extras}
