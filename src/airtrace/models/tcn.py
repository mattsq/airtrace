"""Temporal Convolutional Network (TCN) model."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .base import ResidualWrapperCompatible
from .registry import register


class TemporalBlock(nn.Module):
    """Single temporal block in TCN.

    Consists of dilated causal convolutions with residual connections.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        """Initialize temporal block.

        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Kernel size
            dilation: Dilation factor
            dropout: Dropout probability
        """
        super().__init__()

        # Padding to ensure causality
        padding = (kernel_size - 1) * dilation

        # First conv layer
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)  # Remove future information
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second conv layer
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine layers
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual connection (1x1 conv if dimensions differ)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input [B, C_in, T]

        Returns:
            Output [B, C_out, T]
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the end to ensure causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


@register("tcn")
class TCNModel(ResidualWrapperCompatible):
    """Temporal Convolutional Network for timeseries.

    Uses dilated causal convolutions for long-range dependencies.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_channels: List[int] = [64, 128, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.2,
        causal: bool = True,
        use_skip_connections: bool = True,
        **kwargs
    ):
        """Initialize TCN model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            num_channels: List of hidden channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout probability
            causal: Whether to use causal convolutions
            use_skip_connections: Whether to use skip connections
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.use_skip_connections = use_skip_connections

        # Build TCN layers
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                dilation=dilation, dropout=dropout
            ))

        self.network = nn.Sequential(*layers)

        # Output projection
        self.fc_out = nn.Linear(num_channels[-1], output_dim)

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del context
        conv_inp = x.transpose(1, 2)
        out = self.network(conv_inp)
        sequence_output = out.transpose(1, 2)
        final_hidden = sequence_output[:, -1, :]
        extras = {"hidden": final_hidden, "sequence_output": sequence_output}
        return final_hidden, extras

    def decode(self, latent: torch.Tensor, pred_len: int) -> torch.Tensor:
        preds = self.fc_out(latent).unsqueeze(1)
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
