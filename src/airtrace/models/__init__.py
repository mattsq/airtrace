"""Model implementations."""

from .autoformer import AutoformerModel
from .base import ARBaseModel
from .baselines import (
    DriftModel,
    ExponentialSmoothingModel,
    HoltLinearTrendModel,
    HoltWintersModel,
    LinearARModel,
    LinearTrendModel,
    MeanModel,
    MedianModel,
    MLPARModel,
    MovingAverageModel,
    PersistenceModel,
    PolynomialTrendModel,
    SARIMAModel,
    SeasonalNaiveModel,
    ThetaModel,
    VARModel,
    ZeroModel,
)
from .chronos_bolt import ChronosBoltModel
from .autoformer import AutoformerModel
from .crossformer import CrossformerModel
from .cyclenet import CycleNetModel
from .dlinear import DLinearModel, NLinearModel
from .fedformer import FEDformerModel
from .gru_ar import GRUARModel
from .itransformer import iTransformerModel
from .lag_llama import LagLlamaModel
from .lstm_ar import LSTMARModel
from .mamba2 import Mamba2Model
from .mambats import MambaTSModel
from .moderntcn import ModernTCNModel
from .moirai import MoiraiModel
from .nbeats import NBeatsModel
from .nonstationary_transformer import NonStationaryTransformerModel
from .informer import InformerModel
from .patchtst import PatchTSTModel
from .registry import build_model, list_models, register
from .seq2seq import GRUSeq2SeqModel, LSTMSeq2SeqModel
from .softs import SOFTS
from .tcn import TCNModel
from .timemixer import TimeMixerModel
from .timesnet import TimesNetModel
from .timexer import TimeXerModel
from .transformer import TransformerModel
from .tft import TemporalFusionTransformer

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
    "ModernTCNModel",
    "TransformerModel",
    "PatchTSTModel",
    "iTransformerModel",
    "NonStationaryTransformerModel",
    "TimeMixerModel",
    "TimesNetModel",
    "TimeXerModel",
    "SOFTS",
    "NBeatsModel",
    "AutoformerModel",
    "FEDformerModel",
    "CycleNetModel",
    "CrossformerModel",
    "ChronosBoltModel",
    "MoiraiModel",
    "Mamba2Model",
    "MambaTSModel",
    "LagLlamaModel",
    "DLinearModel",
    "NLinearModel",
    "TemporalFusionTransformer",
    "InformerModel",
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
    "SARIMAModel",
    "HoltLinearTrendModel",
    "HoltWintersModel",
    "ThetaModel",
    "VARModel",
    "LinearARModel",
    "MLPARModel",
]
