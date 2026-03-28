"""config package — configuration centralisée du projet."""

from config.settings import (
    DataConfig,
    EvaluationConfig,
    HMMConfig,
    MarkovSwitchingConfig,
    ProjectConfig,
    StrategyConfig,
    VAEConfig,
    DEFAULT_CONFIG,
)

__all__ = [
    "DataConfig",
    "EvaluationConfig",
    "HMMConfig",
    "MarkovSwitchingConfig",
    "ProjectConfig",
    "StrategyConfig",
    "VAEConfig",
    "DEFAULT_CONFIG",
]