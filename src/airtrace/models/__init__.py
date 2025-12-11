"""Model implementations."""

from .autoformer import AutoformerModel
from .base import ARBaseModel, ResidualWrapperCompatible
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
from .frets import FreTSModel
from .gru_ar import GRUARModel
from .itransformer import iTransformerModel
from .lag_llama import LagLlamaModel
from .lstm_ar import LSTMARModel
from .latent_ponder import LatentPonderWrapper
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
from .residual_solver import ResidualSolver
from .s_mamba import SMambaModel
from .tcn import TCNModel
from .timemixer import TimeMixerModel
from .timesnet import TimesNetModel
from .timexer import TimeXerModel
from .timesfm import TimesFMModel
from .timer import TimerModel
from .transformer import TransformerModel
from .tft import TemporalFusionTransformer
from .tsmixer import TSMixerModel

__all__ = [
    "ARBaseModel",
    "register",
    "build_model",
    "list_models",
    "ResidualWrapperCompatible",
    # Deep learning models
    "GRUARModel",
    "LSTMARModel",
    "ResidualSolver",
    "LatentPonderWrapper",
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
    "TimesFMModel",
    "TimerModel",
    "TSMixerModel",
    "SOFTS",
    "NBeatsModel",
    "AutoformerModel",
    "FEDformerModel",
    "FreTSModel",
    "CycleNetModel",
    "CrossformerModel",
    "ChronosBoltModel",
    "MoiraiModel",
    "Mamba2Model",
    "MambaTSModel",
    "SMambaModel",
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
