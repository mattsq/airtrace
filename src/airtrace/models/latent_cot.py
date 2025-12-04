"""Latent Chain-of-Thought model with adaptive computation.

Implements PonderNet-style adaptive computation where the model iteratively
refines its latent representation before making predictions. The model learns
both how to refine representations and when to stop refining (halting).

References:
    - PonderNet: Learning to Ponder (Banino et al., 2021)
    - Adaptive Computation Time (Graves, 2016)
    - PALBERT: Teaching ALBERT to Ponder (Yun et al., 2021)
"""

from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ARBaseModel
from .registry import register


class PonderBlock(nn.Module):
    """Single recurrent pondering step that refines latent representations.

    This module performs one step of latent reasoning by:
    1. Taking current latent state
    2. Applying learned transformation (e.g., MLP, GRU cell)
    3. Outputting refined latent state

    The same block is applied iteratively during pondering.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        """Initialize ponder block.

        Args:
            latent_dim: Dimension of latent state
            hidden_dim: Hidden dimension for internal MLP
            dropout: Dropout probability
            use_residual: Whether to use residual connections
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.use_residual = use_residual

        # Two-layer MLP with nonlinearity
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(dropout)
        )

        # Layer norm for post-residual
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Perform one pondering step.

        Args:
            latent: Current latent state [B, latent_dim]

        Returns:
            Refined latent state [B, latent_dim]
        """
        refined = self.mlp(latent)

        if self.use_residual:
            latent = latent + refined
        else:
            latent = refined

        return self.norm(latent)


class HaltingModule(nn.Module):
    """Learned halting mechanism that decides when to stop pondering.

    Outputs a scalar halting probability for each sample in the batch.
    During training, we use geometric halting (sampling from the distribution).
    During inference, we can either use geometric halting or threshold-based halting.
    """

    def __init__(self, latent_dim: int):
        """Initialize halting module.

        Args:
            latent_dim: Dimension of latent state to condition on
        """
        super().__init__()
        # Small MLP to produce halting logit
        self.halt_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute halting probability.

        Args:
            latent: Current latent state [B, latent_dim]

        Returns:
            Halting probability [B] in range (0, 1)
        """
        logit = self.halt_net(latent).squeeze(-1)  # [B]
        halt_prob = torch.sigmoid(logit)
        return halt_prob


@register("latent_cot")
class LatentCOTModel(ARBaseModel):
    """Latent Chain-of-Thought model with adaptive computation.

    Architecture:
        1. Encoder: Maps input sequence to initial latent representation
        2. Pondering: Iteratively refines latent with learned halting
        3. Decoder: Maps final latent to output predictions

    During training, the model learns:
        - How to encode inputs to latent space
        - How to refine latent representations (pondering steps)
        - When to stop refining (halting policy)
        - How to decode latent to predictions

    The auxiliary ACT loss encourages the model to use fewer pondering
    steps while maintaining accuracy.

    Shape notation:
        B: Batch size
        T_in: Input sequence length
        D_in: Input feature dimension
        L: Latent dimension
        D_out: Output dimension
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 256,
        encoder_hidden_dim: int = 512,
        ponder_hidden_dim: int = 512,
        encoder_type: str = "gru",
        encoder_num_layers: int = 2,
        max_ponder_steps: int = 10,
        ponder_dropout: float = 0.1,
        use_residual: bool = True,
        halting_threshold: float = 0.99,
        act_loss_weight: float = 0.01,
        **kwargs
    ):
        """Initialize Latent COT model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            latent_dim: Dimension of latent reasoning space
            encoder_hidden_dim: Hidden dimension for encoder
            ponder_hidden_dim: Hidden dimension for ponder blocks
            encoder_type: Type of encoder ("gru", "lstm", "mlp")
            encoder_num_layers: Number of encoder layers
            max_ponder_steps: Maximum number of pondering steps
            ponder_dropout: Dropout probability in ponder blocks
            use_residual: Whether to use residual connections in ponder blocks
            halting_threshold: Cumulative halt probability threshold for inference
            act_loss_weight: Weight for Adaptive Computation Time regularization
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        self.max_ponder_steps = max_ponder_steps
        self.halting_threshold = halting_threshold
        self.act_loss_weight = act_loss_weight

        # === ENCODER: Input sequence -> Initial latent ===
        if encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=input_dim,
                hidden_size=encoder_hidden_dim,
                num_layers=encoder_num_layers,
                dropout=ponder_dropout if encoder_num_layers > 1 else 0.0,
                batch_first=True
            )
            encoder_output_dim = encoder_hidden_dim
        elif encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=input_dim,
                hidden_size=encoder_hidden_dim,
                num_layers=encoder_num_layers,
                dropout=ponder_dropout if encoder_num_layers > 1 else 0.0,
                batch_first=True
            )
            encoder_output_dim = encoder_hidden_dim
        elif encoder_type == "mlp":
            # Simple MLP encoder that pools over sequence
            self.encoder = nn.Sequential(
                nn.Flatten(start_dim=1),  # [B, T_in, D_in] -> [B, T_in * D_in]
                nn.Linear(input_dim * 100, encoder_hidden_dim),  # Assumes max T_in=100
                nn.LayerNorm(encoder_hidden_dim),
                nn.GELU(),
                nn.Dropout(ponder_dropout)
            )
            encoder_output_dim = encoder_hidden_dim
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Project encoder output to latent space
        self.encoder_to_latent = nn.Linear(encoder_output_dim, latent_dim)

        # === PONDERING: Iterative latent refinement ===
        self.ponder_block = PonderBlock(
            latent_dim=latent_dim,
            hidden_dim=ponder_hidden_dim,
            dropout=ponder_dropout,
            use_residual=use_residual
        )

        # === HALTING: Learned stopping criterion ===
        self.halting_module = HaltingModule(latent_dim=latent_dim)

        # === DECODER: Final latent -> Predictions ===
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(ponder_dropout),
            nn.Linear(latent_dim // 2, output_dim)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input sequence to initial latent representation.

        Args:
            x: Input tensor [B, T_in, D_in]

        Returns:
            Initial latent state [B, latent_dim]
        """
        B = x.size(0)

        if self.encoder_type in ["gru", "lstm"]:
            # Use final hidden state as initial latent
            if self.encoder_type == "gru":
                _, hidden = self.encoder(x)  # hidden: [num_layers, B, H]
                encoder_repr = hidden[-1]  # [B, H]
            else:  # LSTM
                _, (hidden, _) = self.encoder(x)
                encoder_repr = hidden[-1]  # [B, H]
        else:  # MLP
            # Pool over sequence dimension (e.g., mean pooling)
            encoder_repr = self.encoder(x.mean(dim=1))  # [B, H]

        # Project to latent space
        latent = self.encoder_to_latent(encoder_repr)  # [B, latent_dim]
        return latent

    def ponder(
        self,
        initial_latent: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Perform iterative pondering with learned halting.

        Args:
            initial_latent: Initial latent state [B, latent_dim]
            deterministic: If True, use threshold-based halting (inference mode)
                          If False, use geometric sampling (training mode)

        Returns:
            Tuple of:
                - final_latent: Final pondered latent [B, latent_dim]
                - latent_states: List of latent states at each step
                - halt_probs: List of halting probabilities at each step
                - num_steps: Actual number of steps taken [B]
        """
        B = initial_latent.size(0)
        device = initial_latent.device

        latent_states = [initial_latent]
        halt_probs = []

        # Track cumulative halt probability and active samples
        cum_halt_prob = torch.zeros(B, device=device)  # [B]
        active_mask = torch.ones(B, dtype=torch.bool, device=device)  # [B]
        num_steps = torch.zeros(B, device=device)  # [B]

        current_latent = initial_latent

        for step in range(self.max_ponder_steps):
            # Compute halting probability for current latent
            halt_prob = self.halting_module(current_latent)  # [B]
            halt_probs.append(halt_prob)

            # Update cumulative halting probability
            # Only for samples that haven't halted yet
            effective_halt_prob = halt_prob * active_mask.float()
            cum_halt_prob = cum_halt_prob + effective_halt_prob

            # Determine which samples should halt at this step
            if deterministic:
                # Threshold-based halting (inference)
                should_halt = (cum_halt_prob >= self.halting_threshold) & active_mask
            else:
                # Geometric sampling (training)
                # Sample from Bernoulli(halt_prob) for active samples
                halt_samples = torch.bernoulli(halt_prob) * active_mask.float()
                should_halt = halt_samples.bool()

            # Update step counter for samples that halted
            num_steps = num_steps + active_mask.float()

            # Update active mask
            active_mask = active_mask & (~should_halt)

            # If all samples halted, stop early
            if not active_mask.any():
                break

            # Ponder: refine latent for active samples
            # (We still compute for all samples to maintain batch structure)
            current_latent = self.ponder_block(current_latent)
            latent_states.append(current_latent)

        # Final latent is the last computed state
        final_latent = current_latent

        return final_latent, latent_states, halt_probs, num_steps

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode final latent representation to predictions.

        Args:
            latent: Final latent state [B, latent_dim]

        Returns:
            Predictions [B, D_out]
        """
        return self.decoder(latent)

    def compute_act_loss(
        self,
        halt_probs: List[torch.Tensor],
        num_steps: torch.Tensor
    ) -> torch.Tensor:
        """Compute Adaptive Computation Time (ACT) regularization loss.

        Encourages the model to use fewer pondering steps by penalizing
        expected computation time.

        Args:
            halt_probs: List of halting probabilities at each step, each [B]
            num_steps: Actual number of steps taken [B]

        Returns:
            Scalar ACT loss
        """
        # Expected number of steps (ponder cost)
        # E[N] = sum_{t=1}^T (1 - sum_{s=1}^{t-1} p_halt(s))
        # Simpler approximation: just take mean of actual steps
        ponder_cost = num_steps.mean()

        return ponder_cost

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_all_steps: bool = False,
        deterministic: bool = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with latent chain-of-thought.

        Args:
            x: Input tensor [B, T_in, D_in]
            context: Optional context tensor (not used currently)
            return_all_steps: If True, return predictions at all pondering steps
            deterministic: Override halting mode (True=threshold, False=sampling)
                          If None, uses self.training (False in train, True in eval)
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - preds: Final predictions [B, 1, D_out]
                - extras: Dict with:
                    - latent_states: List of latent states at each step
                    - halt_probs: List of halting probabilities at each step
                    - num_steps: Number of pondering steps taken [B]
                    - act_loss: ACT regularization loss (scalar)
                    - all_preds: (if return_all_steps) Predictions at each step
                    - initial_latent: Initial latent before pondering
                    - final_latent: Final latent after pondering
        """
        # Default: use sampling during training, threshold during eval
        if deterministic is None:
            deterministic = not self.training

        # === ENCODE ===
        initial_latent = self.encode(x)  # [B, latent_dim]

        # === PONDER ===
        final_latent, latent_states, halt_probs, num_steps = self.ponder(
            initial_latent,
            deterministic=deterministic
        )

        # === DECODE ===
        preds = self.decode(final_latent)  # [B, D_out]

        # Reshape to [B, 1, D_out] for consistency with task expectations
        preds = preds.unsqueeze(1)

        # === COMPUTE ACT LOSS ===
        act_loss = self.compute_act_loss(halt_probs, num_steps)

        # === PREPARE EXTRAS ===
        extras = {
            "initial_latent": initial_latent,
            "final_latent": final_latent,
            "latent_states": latent_states,
            "halt_probs": halt_probs,
            "num_steps": num_steps,
            "act_loss": act_loss,
            "mean_steps": num_steps.mean().item(),
            "max_steps": num_steps.max().item(),
        }

        # Optionally decode all intermediate latent states
        if return_all_steps:
            all_preds = []
            for latent in latent_states:
                pred = self.decode(latent).unsqueeze(1)  # [B, 1, D_out]
                all_preds.append(pred)
            extras["all_preds"] = all_preds

        return {
            "preds": preds,
            "extras": extras
        }
