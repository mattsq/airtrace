"""Model implementations."""

from .base import ARBaseModel
from .baselines import (
    DriftModel,
    ExponentialSmoothingModel,
    HoltLinearTrendModel,
    LinearARModel,
    LinearTrendModel,
    MeanModel,
    MedianModel,
    MLPARModel,
    MovingAverageModel,
    PersistenceModel,
    PolynomialTrendModel,
    SeasonalNaiveModel,
    ThetaModel,
    ZeroModel,
)
from .gru_ar import GRUARModel
from .lstm_ar import LSTMARModel
from .patchtst import PatchTSTModel
from .registry import build_model, list_models, register
from .seq2seq import GRUSeq2SeqModel, LSTMSeq2SeqModel
from .tcn import TCNModel
from .transformer import TransformerModel

__all__ = [
    "ARBaseModel",
    "register",
    "build_model",
    "list_models",
    # Deep learning models
    "GRUARModel",
    "LSTMARModel",
    "GRUSeq2SeqModel",
    "LSTMSeq2SeqModel",
    "TCNModel",
    "TransformerModel",
    "PatchTSTModel",
    # Baseline models
    "PersistenceModel",
    "MovingAverageModel",
    "ZeroModel",
    "LinearTrendModel",
    "MeanModel",
    "MedianModel",
    "DriftModel",
    "ExponentialSmoothingModel",
    "SeasonalNaiveModel",
    "PolynomialTrendModel",
    "HoltLinearTrendModel",
    "ThetaModel",
    "LinearARModel",
    "MLPARModel",
]
