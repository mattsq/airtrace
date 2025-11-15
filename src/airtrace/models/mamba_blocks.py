"""Core building blocks for Mamba-based models.

This module implements the fundamental components of the Mamba architecture:
- Selective State Space Model (SSM) layer
- Bidirectional Mamba block for capturing inter-variate correlations

Reference:
    Mamba: Linear-Time Sequence Modeling with Selective State Spaces
    (Gu & Dao, 2023) https://arxiv.org/abs/2312.00752

    Is Mamba Effective for Time Series Forecasting?
    (Wen et al., 2024) https://arxiv.org/abs/2403.11144
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """Selective State Space Model (SSM) layer.

    Core innovation of Mamba: parameters (Δ, B, C) are input-dependent,
    making the SSM selective and able to filter relevant information.

    The SSM is defined by the continuous-time equations:
        h'(t) = A h(t) + B(x) x(t)
        y(t) = C(x) h(t)

    Where:
        - h(t) is the hidden state (dimension d_state)
        - A is a fixed state transition matrix
        - B(x), C(x) are input-dependent projection matrices
        - Δ(x) is the input-dependent discretization step size

    The discrete-time version is computed via parallel associative scan,
    maintaining O(L) complexity instead of the naive O(L * d_state²).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
    ):
        """Initialize Selective SSM layer.

        Args:
            d_model: Model dimension (input/output)
            d_state: SSM state expansion factor (N in paper)
            d_conv: Local convolution width
            expand: Block expansion factor
            dt_rank: Rank of Δ projection (default: d_model // 16)
            dt_min: Minimum value for Δ initialization
            dt_max: Maximum value for Δ initialization
            dt_init: Initialization method for Δ ('random', 'constant')
            dt_scale: Scale factor for Δ initialization
            bias: Whether to use bias in projections
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = dt_rank or (d_model // 16)

        # Input projection (expands to d_inner)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        # Convolutional layer for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,  # Depthwise convolution
            padding=d_conv - 1,
            bias=bias
        )

        # Projections for selective parameters (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # Δ (delta) projection - controls discretization step size
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize Δ projection bias to encourage initial values in [dt_min, dt_max]
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias to inverse softplus of uniform [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # Inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter - fixed state transition matrix
        # Initialize as in the paper: A = -exp(uniform(0, log(d_state)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Store log for numerical stability
        self.A_log._no_weight_decay = True

        # D parameter - skip connection (output gating)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan using parallel associative scan.

        This is a simplified PyTorch implementation. The official implementation
        uses custom CUDA kernels for efficiency.

        Args:
            u: Input [B, L, D]
            delta: Step sizes [B, L, D]
            A: State transition [D, N]
            B: Input matrix [B, L, N]
            C: Output matrix [B, L, N]
            D: Skip connection [D]

        Returns:
            Output [B, L, D]
        """
        B_batch, L, D = u.shape
        N = A.shape[1]

        # Discretize A and B using zero-order hold (ZOH)
        # Δ A = exp(Δ * A)
        # Δ B = (exp(Δ * A) - I) / A * B ≈ Δ * B for small Δ

        deltaA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # [B, L, D, N]
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2)  # [B, L, D, N]

        # Compute state h iteratively (recurrent form)
        # h[t] = deltaA[t] * h[t-1] + deltaB[t] * u[t]
        h = torch.zeros(B_batch, D, N, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(L):
            h = deltaA[:, t] * h + deltaB[:, t] * u[:, t:t+1]  # [B, D, N]
            y = (h @ C[:, t].unsqueeze(-1)).squeeze(-1)  # [B, D]
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [B, L, D]

        # Add skip connection
        y = y + u * D.unsqueeze(0).unsqueeze(0)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        B, L, D = x.shape

        # Input projection with gating (like Gated Linear Unit)
        x_and_res = self.in_proj(x)  # [B, L, 2*d_inner]
        x_proj, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # Apply local convolution (temporal context)
        x_conv = self.conv1d(x_proj.transpose(1, 2))[..., :L].transpose(1, 2)  # [B, L, d_inner]
        x_conv = F.silu(x_conv)  # SiLU activation

        # Compute selective parameters (input-dependent)
        x_dbl = self.x_proj(x_conv)  # [B, L, dt_rank + 2*d_state]
        dt, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # Project dt to d_inner dimension and apply softplus for positivity
        dt = F.softplus(self.dt_proj(dt))  # [B, L, d_inner]

        # Get A matrix (fixed, but learned)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        # Apply selective scan
        y = self._selective_scan(
            u=x_conv,
            delta=dt,
            A=A,
            B=B_proj,
            C=C_proj,
            D=self.D,
        )  # [B, L, d_inner]

        # Gating with residual
        y = y * F.silu(res)

        # Output projection
        output = self.out_proj(y)  # [B, L, D]

        return output


class BidirectionalMambaBlock(nn.Module):
    """Bidirectional Mamba block for capturing inter-variate correlations.

    Processes the sequence in both forward and backward directions,
    then combines the outputs. This is particularly effective for
    capturing relationships between different sensors/variates in
    multivariate time series.

    Used in S-Mamba for the inter-variate correlation (VC) encoding layer.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        norm_type: str = "layer",
    ):
        """Initialize bidirectional Mamba block.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution width
            expand: Expansion factor
            dropout: Dropout probability
            norm_type: Normalization type ('layer' or 'rms')
        """
        super().__init__()

        self.d_model = d_model

        # Forward and backward Mamba layers
        self.forward_mamba = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.backward_mamba = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # Normalization
        if norm_type == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif norm_type == "rms":
            self.norm = RMSNorm(d_model)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Combination projection
        self.combine = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, L, D]

        Returns:
            Output tensor [B, L, D]
        """
        # Store input for residual
        residual = x

        # Normalize
        x = self.norm(x)

        # Forward pass
        x_forward = self.forward_mamba(x)

        # Backward pass (reverse sequence)
        x_backward = self.backward_mamba(torch.flip(x, dims=[1]))
        x_backward = torch.flip(x_backward, dims=[1])  # Flip back

        # Combine forward and backward
        x_combined = torch.cat([x_forward, x_backward], dim=-1)  # [B, L, 2*D]
        x_out = self.combine(x_combined)  # [B, L, D]

        # Dropout and residual
        x_out = self.dropout(x_out)
        x_out = x_out + residual

        return x_out


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient alternative to LayerNorm, used in some modern architectures.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [*, D]

        Returns:
            Normalized tensor [*, D]
        """
        # Compute RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed
