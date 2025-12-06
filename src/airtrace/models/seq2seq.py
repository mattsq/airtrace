"""Sequence-to-Sequence encoder-decoder models."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base import ResidualWrapperCompatible
from .registry import register


class Seq2SeqEncoder(nn.Module):
    """Encoder for Seq2Seq model."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: str = "gru"
    ):
        """Initialize encoder.

        Args:
            input_dim: Input feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
            cell_type: Type of RNN cell ('gru' or 'lstm')
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type

        if cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")

    def forward(self, x: torch.Tensor):
        """Encode input sequence.

        Args:
            x: Input tensor [B, T_in, D_in]

        Returns:
            outputs: Encoder outputs [B, T_in, H]
            hidden: Final hidden state(s)
        """
        outputs, hidden = self.rnn(x)
        return outputs, hidden


class Seq2SeqDecoder(nn.Module):
    """Decoder for Seq2Seq model."""

    def __init__(
        self,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        cell_type: str = "gru",
        use_attention: bool = False
    ):
        """Initialize decoder.

        Args:
            output_dim: Output feature dimension
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
            cell_type: Type of RNN cell ('gru' or 'lstm')
            use_attention: Whether to use attention mechanism
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.use_attention = use_attention

        if cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=output_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        elif cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=output_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown cell_type: {cell_type}")

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            # Combine context and decoder output
            self.context_combine = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.attention = None

        # Output projection
        self.fc_out = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: Optional[torch.Tensor] = None
    ):
        """Decode one step.

        Args:
            x: Input tensor [B, 1, D_out]
            hidden: Hidden state from previous step
            encoder_outputs: Encoder outputs for attention [B, T_in, H]

        Returns:
            output: Decoded output [B, 1, D_out]
            hidden: Updated hidden state
            attn_weights: Attention weights (if attention is used)
        """
        # Pass through RNN
        rnn_output, hidden = self.rnn(x, hidden)  # [B, 1, H]

        # Apply attention if enabled
        attn_weights = None
        if self.use_attention and encoder_outputs is not None:
            # Compute attention
            attn_output, attn_weights = self.attention(
                rnn_output, encoder_outputs, encoder_outputs
            )  # [B, 1, H]

            # Combine with RNN output
            combined = torch.cat([rnn_output, attn_output], dim=2)  # [B, 1, 2*H]
            rnn_output = self.context_combine(combined)  # [B, 1, H]

        # Apply dropout
        rnn_output = self.dropout(rnn_output)

        # Project to output dimension
        output = self.fc_out(rnn_output)  # [B, 1, D_out]

        return output, hidden, attn_weights


@register("gru_seq2seq")
class GRUSeq2SeqModel(ResidualWrapperCompatible):
    """GRU-based Seq2Seq encoder-decoder model.

    Uses a GRU encoder to encode the input sequence and a GRU decoder
    to generate predictions. Supports teacher forcing during training
    and optional attention mechanism.

    This is a more sophisticated architecture for multi-step forecasting
    compared to single-shot encoder models.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        teacher_forcing_ratio: float = 0.5,
        **kwargs
    ):
        """Initialize GRU Seq2Seq model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            teacher_forcing_ratio: Probability of using teacher forcing
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Encoder
        self.encoder = Seq2SeqEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type="gru"
        )

        # Decoder
        self.decoder = Seq2SeqDecoder(
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type="gru",
            use_attention=use_attention
        )

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode input sequence for reuse by residual wrappers."""

        del context
        encoder_outputs, hidden = self.encoder(x)
        extras = {"encoder_outputs": encoder_outputs, "hidden": hidden}
        return (encoder_outputs, hidden), extras

    def decode(
        self,
        latent: Tuple[torch.Tensor, torch.Tensor],
        pred_len: int,
        target: Optional[torch.Tensor] = None,
        return_extras: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Decode predictions from encoded latent state.

        When ``target`` is provided, teacher forcing follows the original
        stochastic schedule used by the single-shot forward pass.
        """

        encoder_outputs, hidden = latent
        decoder_input = torch.zeros(
            encoder_outputs.size(0),
            1,
            self.output_dim,
            device=encoder_outputs.device,
            dtype=encoder_outputs.dtype,
        )

        predictions = []
        attention_weights = []
        for t in range(pred_len):
            output, hidden, attn_weights = self.decoder(
                decoder_input, hidden, encoder_outputs
            )
            predictions.append(output)

            if attn_weights is not None:
                attention_weights.append(attn_weights)

            if target is not None and t < target.shape[1]:
                use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
                if use_teacher_forcing and self.training:
                    decoder_input = target[:, t : t + 1, :]
                else:
                    decoder_input = output
            else:
                decoder_input = output

        preds = torch.cat(predictions, dim=1)
        decode_extras: Dict[str, torch.Tensor] = {
            "encoder_outputs": encoder_outputs,
            "hidden": hidden,
        }
        if attention_weights:
            decode_extras["attention_weights"] = torch.stack(attention_weights, dim=1)

        if return_extras:
            return preds, decode_extras
        return preds

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        pred_len: int = 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""

        latent, encode_extras = self.encode(x, context=context)
        preds, decode_extras = self.decode(
            latent, pred_len=pred_len, target=target, return_extras=True
        )
        extras = {**encode_extras, **decode_extras}
        return {"preds": preds, "extras": extras}


@register("lstm_seq2seq")
class LSTMSeq2SeqModel(ResidualWrapperCompatible):
    """LSTM-based Seq2Seq encoder-decoder model.

    Uses an LSTM encoder to encode the input sequence and an LSTM decoder
    to generate predictions. Supports teacher forcing during training
    and optional attention mechanism.

    Similar to GRU Seq2Seq but uses LSTM cells which can capture longer-term
    dependencies through their cell state mechanism.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = False,
        teacher_forcing_ratio: float = 0.5,
        **kwargs
    ):
        """Initialize LSTM Seq2Seq model.

        Args:
            input_dim: Dimension of input features
            output_dim: Dimension of output predictions
            hidden_size: Hidden state dimension
            num_layers: Number of RNN layers
            dropout: Dropout probability
            use_attention: Whether to use attention mechanism
            teacher_forcing_ratio: Probability of using teacher forcing
            **kwargs: Additional arguments
        """
        super().__init__(input_dim, output_dim, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.teacher_forcing_ratio = teacher_forcing_ratio

        # Encoder
        self.encoder = Seq2SeqEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type="lstm"
        )

        # Decoder
        self.decoder = Seq2SeqDecoder(
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type="lstm",
            use_attention=use_attention
        )

    def encode(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> Tuple[Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Dict[str, torch.Tensor]]:
        """Encode input sequence for reuse by residual wrappers."""

        del context
        encoder_outputs, (hidden, cell) = self.encoder(x)
        extras = {
            "encoder_outputs": encoder_outputs,
            "hidden": hidden,
            "cell": cell,
        }
        return (encoder_outputs, (hidden, cell)), extras

    def decode(
        self,
        latent: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        pred_len: int,
        target: Optional[torch.Tensor] = None,
        return_extras: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Decode predictions from encoded latent state."""

        encoder_outputs, (hidden, cell) = latent
        decoder_input = torch.zeros(
            encoder_outputs.size(0),
            1,
            self.output_dim,
            device=encoder_outputs.device,
            dtype=encoder_outputs.dtype,
        )

        predictions = []
        attention_weights = []
        for t in range(pred_len):
            output, (hidden, cell), attn_weights = self.decoder(
                decoder_input, (hidden, cell), encoder_outputs
            )
            predictions.append(output)

            if attn_weights is not None:
                attention_weights.append(attn_weights)

            if target is not None and t < target.shape[1]:
                use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
                if use_teacher_forcing and self.training:
                    decoder_input = target[:, t : t + 1, :]
                else:
                    decoder_input = output
            else:
                decoder_input = output

        preds = torch.cat(predictions, dim=1)
        decode_extras: Dict[str, torch.Tensor] = {
            "encoder_outputs": encoder_outputs,
            "hidden": hidden,
            "cell": cell,
        }
        if attention_weights:
            decode_extras["attention_weights"] = torch.stack(attention_weights, dim=1)

        if return_extras:
            return preds, decode_extras
        return preds

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        pred_len: int = 1,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""

        latent, encode_extras = self.encode(x, context=context)
        preds, decode_extras = self.decode(
            latent, pred_len=pred_len, target=target, return_extras=True
        )
        extras = {**encode_extras, **decode_extras}
        return {"preds": preds, "extras": extras}
