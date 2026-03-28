"""
config/settings.py
==================
Centralisation de toute la configuration du projet via des dataclasses.

Chaque module possède sa propre Config (frozen=True pour l'immuabilité).
Le ProjectConfig agrège toutes les sous-configurations en un seul point d'entrée.

Design pattern : Configuration-as-Code, évite les magic numbers dispersés.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Configuration du téléchargement et feature engineering.

    Parameters
    ----------
    tickers : List[str]
        Liste des tickers yfinance (ex: SPY, QQQ, IEF, GLD, VIX).
    benchmark : str
        Ticker de référence pour le calcul de returns et régimes.
    start_date : str
        Date de début au format 'YYYY-MM-DD'.
    end_date : str
        Date de fin au format 'YYYY-MM-DD'.
    train_ratio : float
        Part des données allouée au train (défaut 0.70).
    val_ratio : float
        Part des données allouée à la validation (défaut 0.15).
    seq_length : int
        Longueur des séquences pour le LSTM-VAE.
    vol_windows : List[int]
        Fenêtres de volatilité glissante (en jours).
    cache_dir : Path
        Dossier de cache pour les données téléchargées.
    """

    tickers: Tuple[str, ...] = ("SPY", "QQQ", "IEF", "GLD", "^VIX")
    benchmark: str = "SPY"
    start_date: str = "2005-01-01"
    end_date: str = "2024-12-31"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    seq_length: int = 20
    vol_windows: Tuple[int, ...] = (5, 10, 21, 63)
    cache_dir: Path = Path("artifacts/data")


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VAEConfig:
    """Configuration du Variational Autoencoder (LSTM-VAE).

    Parameters
    ----------
    input_dim : int
        Nombre de features d'entrée (après feature engineering).
    hidden_dim : int
        Dimension des couches LSTM cachées.
    latent_dim : int
        Dimension de l'espace latent z.
    num_layers : int
        Nombre de couches LSTM empilées.
    dropout : float
        Taux de dropout (actif seulement si num_layers > 1).
    beta : float
        Poids du terme KL (β-VAE). 1.0 = VAE classique.
    learning_rate : float
        Taux d'apprentissage initial (Adam).
    batch_size : int
        Taille des mini-batches.
    max_epochs : int
        Nombre maximum d'époques d'entraînement.
    kl_anneal_epochs : int
        Nombre d'époques pour le warm-up du terme KL (annealing).
    patience : int
        Patience pour l'early stopping (sur la val loss).
    seed : int
        Seed global pour la reproductibilité.
    """

    input_dim: int = 15
    hidden_dim: int = 64
    latent_dim: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    beta: float = 1.0
    learning_rate: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 100
    kl_anneal_epochs: int = 20
    patience: int = 15
    seed: int = 42


# ---------------------------------------------------------------------------
# HMM
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HMMConfig:
    """Configuration du GaussianHMM ajusté sur l'espace latent.

    Parameters
    ----------
    n_regimes : int
        Nombre de régimes discrets (ex: 2 = bull/bear, 3 = bull/transition/bear).
    covariance_type : str
        Type de covariance des émissions : 'full', 'diag', 'tied', 'spherical'.
    n_iter : int
        Nombre d'itérations EM maximum.
    n_init : int
        Nombre d'initialisations aléatoires (best model kept).
    tol : float
        Seuil de convergence EM (variation log-likelihood).
    """

    n_regimes: int = 3
    covariance_type: str = "full"
    n_iter: int = 200
    n_init: int = 10
    tol: float = 1e-4


# ---------------------------------------------------------------------------
# Markov Switching Baseline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MarkovSwitchingConfig:
    """Configuration du modèle de référence Hamilton (1989).

    Parameters
    ----------
    k_regimes : int
        Nombre de régimes (2 pour la convention Hamilton).
    order : int
        Ordre AR du modèle (0 = Markov Regression, > 0 = Markov AR).
    switching_variance : bool
        Si True, la variance est aussi spécifique au régime.
    """

    k_regimes: int = 2
    order: int = 0
    switching_variance: bool = True


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyConfig:
    """Configuration de la stratégie adaptative multi-régimes.

    Parameters
    ----------
    regime_allocations : dict
        Allocation cible en actif risqué par régime.
        {0: 1.0} → 100% investi en régime 0 (bull/low-vol).
        {2: 0.0} → 0% investi en régime 2 (bear/high-vol).
    transaction_cost_bps : float
        Coût de transaction en points de base (1 bps = 0.01%).
    slippage_bps : float
        Glissement de marché estimé en bps.
    rebalance_threshold : float
        Seuil de dérive avant rebalancement (0.0 = rebalancement systématique).
    use_next_regime : bool
        Si True, utilise la prédiction du régime à t+1 (forward-looking).
        Si False, utilise le régime courant (plus conservateur).
    initial_capital : float
        Capital de départ en USD.
    risk_free_rate : float
        Taux sans risque annualisé pour le ratio de Sharpe.
    """

    regime_allocations: dict = field(
        default_factory=lambda: {0: 1.0, 1: 0.5, 2: 0.0}
    )
    transaction_cost_bps: float = 10.0
    slippage_bps: float = 5.0
    rebalance_threshold: float = 0.0
    use_next_regime: bool = True
    initial_capital: float = 100_000.0
    risk_free_rate: float = 0.02


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration du module d'évaluation.

    Parameters
    ----------
    annualization_factor : int
        Facteur d'annualisation (252 pour données journalières).
    rolling_window : int
        Fenêtre pour les métriques glissantes (Sharpe glissant, etc.).
    output_dir : Path
        Dossier de sortie pour les rapports et figures.
    figure_dpi : int
        Résolution des figures sauvegardées.
    figure_format : str
        Format de sauvegarde : 'png', 'pdf', 'svg'.
    """

    annualization_factor: int = 252
    rolling_window: int = 63  # trimestre
    output_dir: Path = Path("artifacts")
    figure_dpi: int = 150
    figure_format: str = "png"


# ---------------------------------------------------------------------------
# Project-level config (agrégateur)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectConfig:
    """Configuration globale du projet VAE-HMM.

    Agrège toutes les sous-configurations en un seul point d'entrée.
    Utilisé par main.py comme objet de configuration unique.

    Examples
    --------
    >>> cfg = ProjectConfig()
    >>> cfg.vae.latent_dim
    8
    >>> cfg.hmm.n_regimes
    3
    """

    data: DataConfig = field(default_factory=DataConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    markov_switching: MarkovSwitchingConfig = field(
        default_factory=MarkovSwitchingConfig
    )
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Noms canoniques des régimes (pour les graphiques)
    regime_names: Tuple[str, ...] = ("Bull / Low-Vol", "Transition", "Bear / High-Vol")
    regime_colors: Tuple[str, ...] = ("#2ecc71", "#f39c12", "#e74c3c")
    project_name: str = "VAE-HMM Market Regime Detection"


# ---------------------------------------------------------------------------
# Default instance (import direct)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = ProjectConfig()
