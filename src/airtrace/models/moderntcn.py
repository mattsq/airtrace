"""ModernTCN: A Modern Pure Convolution Structure for Time Series Analysis.

Based on: "ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis"
Luo et al., ICLR 2024 Spotlight
https://openreview.net/forum?id=vpJMJerXHU

Key innovations over classic TCN:
- Depthwise separable convolutions for efficiency
- Larger effective receptive fields (ERFs)
- Modern training techniques (LayerNorm, GELU)
- Better parameter efficiency and gradient flow
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import ResidualWrapperCompatible
from .registry import register


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for 1D time series.

    Splits convolution into:
    1. Depthwise: Each channel convolved independently
    2. Pointwise: 1x1 conv to mix channels

    This reduces parameters and computation while maintaining expressiveness.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        padding: int = 0,
    ):
        """Initialize depthwise separable convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolving kernel
            dilation: Dilation rate
            padding: Padding to add to input
        """
        super().__init__()

        # Store padding so we can remove it after convolution to maintain
        # the original temporal length (causal padding adds future context
        # that should be trimmed before mixing channels).
        self.padding = padding

        # Depthwise convolution: convolve each channel separately
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            groups=in_channels,  # Key: each channel is its own group
        )

        # Pointwise convolution: 1x1 conv to mix channels
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C_in, T]

        Returns:
            Output tensor [B, C_out, T]
        """
        x = self.depthwise(x)

        # Remove future padding to preserve input sequence length
        if self.padding > 0:
            x = x[:, :, :-self.padding]

        x = self.pointwise(x)
        return x


class ModernTCNBlock(nn.Module):
    """Modern TCN block with depthwise separable convolutions.

    Architecture:
        LayerNorm -> DepthwiseSeparableConv -> GELU -> Dropout ->
        DepthwiseSeparableConv -> Dropout -> Residual
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        """Initialize ModernTCN block.

        Args:
            channels: Number of channels (same for input and output)
            kernel_size: Kernel size for convolutions
            dilation: Dilation factor for temporal receptive field
            dropout: Dropout probability
        """
        super().__init__()

        # Causal padding to ensure we don't look into the future
        self.padding = (kernel_size - 1) * dilation

        # Pre-normalization (modern approach)
        self.norm = nn.LayerNorm(channels)

        # First depthwise separable conv
        self.conv1 = DepthwiseSeparableConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        # GELU activation (modern, smoother than ReLU)
        self.act = nn.GELU()

        # Dropout
        self.dropout1 = nn.Dropout(dropout)

        # Second depthwise separable conv
        self.conv2 = DepthwiseSeparableConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding,
        )

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C, T]
        """
        residual = x

        # Normalize (need to transpose for LayerNorm)
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        x = x.transpose(1, 2)  # [B, C, T]

        # First conv block
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout1(x)

        # Second conv block
        x = self.conv2(x)
        x = self.dropout2(x)

        # Residual connection
        return x + residual


class LargeKernelConv(nn.Module):
    """Large kernel convolution module for increased receptive field.

    Uses a single large kernel to capture long-range dependencies
    more directly than stacking smaller kernels.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dropout: float = 0.1,
    ):
        """Initialize large kernel convolution.

        Args:
            channels: Number of channels
            kernel_size: Large kernel size (e.g., 51, 101)
            dropout: Dropout probability
        """
        super().__init__()

        # Causal padding for large kernel
        self.padding = kernel_size - 1

        self.norm = nn.LayerNorm(channels)

        # Depthwise large kernel conv (efficient)
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=self.padding,
            groups=channels,  # Depthwise
        )

        # Pointwise conv to mix channels
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C, T]
        """
        residual = x

        # Normalize
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        # Large kernel conv
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]  # Causal

        # Pointwise
        x = self.pointwise(x)
        x = self.act(x)
        x = self.dropout(x)

        return x + residual


@register("moderntcn")
class ModernTCNModel(ResidualWrapperCompatible):
    """ModernTCN: Modern Temporal Convolutional Network.

    A modern pure convolution architecture for time series analysis with:
    - Much larger effective receptive fields than classic TCN
    - Depthwise separable convolutions for efficiency
    - Modern normalization (LayerNorm) and activations (GELU)
    - Strong residual connections for better gradient flow

    Reference:
        Luo et al., "ModernTCN: A Modern Pure Convolution Structure for
        General Time Series Analysis", ICLR 2024 Spotlight
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int = 6,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        large_kernel_size: int = 51,
        dilation_growth: int = 2,
        dropout: float = 0.1,
        use_large_kernel: bool = True,
        **kwargs
    ):
        """Initialize ModernTCN model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            num_blocks: Number of ModernTCN blocks
            hidden_channels: Number of hidden channels
            kernel_size: Kernel size for regular convolutions
            large_kernel_size: Kernel size for large kernel module
            dilation_growth: Growth rate for dilation (e.g., 2 for exponential)
            dropout: Dropout probability
            use_large_kernel: Whether to use large kernel module
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.large_kernel_size = large_kernel_size
        self.use_large_kernel = use_large_kernel

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, hidden_channels, kernel_size=1)

        # ModernTCN blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = dilation_growth ** i
            self.blocks.append(
                ModernTCNBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        # Optional large kernel module for very large receptive field
        if use_large_kernel:
            self.large_kernel = LargeKernelConv(
                channels=hidden_channels,
                kernel_size=large_kernel_size,
                dropout=dropout,
            )

        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_channels)

        # Output projection
        self.output_proj = nn.Linear(hidden_channels, output_dim)

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        conv_inp = x.transpose(1, 2)
        conv_inp = self.input_proj(conv_inp)

        for block in self.blocks:
            conv_inp = block(conv_inp)

        if self.use_large_kernel:
            conv_inp = self.large_kernel(conv_inp)

        sequence_output = conv_inp.transpose(1, 2)
        sequence_output = self.final_norm(sequence_output)
        final_hidden = sequence_output[:, -1, :]
        extras = {"hidden": final_hidden, "sequence_output": sequence_output}
        return final_hidden, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        preds = self.output_proj(latent).unsqueeze(1)
        if pred_len != 1:
            preds = preds.expand(-1, pred_len, -1)
        return preds

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        pred_len = int(kwargs.get("pred_len", 1))
        latent, extras = self.encode(x, context=context)
        preds = self.decode(latent, pred_len)
        extras["representation"] = latent
        return {"preds": preds, "extras": extras}
