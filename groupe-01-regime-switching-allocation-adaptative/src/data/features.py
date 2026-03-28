"""
data/features.py
================
Feature Engineering pour la détection de régimes de marché.

Calcule un ensemble d'indicateurs couvrant :
  - Rendements (log et simples) sur plusieurs horizons
  - Volatilité réalisée glissante (proxy de l'état de volatilité)
  - Indicateurs techniques (RSI, Bandes de Bollinger, tendance EMA)
  - Corrélations glissantes inter-actifs (optionnel)

Classes
-------
FeatureEngineer
    Transforme des prix bruts en feature matrix prêt pour le VAE.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import ta
from loguru import logger


class FeatureEngineer:
    """
    Calcule et assemble les features pour la détection de régimes.

    Le pipeline est déterministe et sans look-ahead : toutes les
    statistiques glissantes utilisent des fenêtres passées uniquement.

    Parameters
    ----------
    vol_windows : list[int]
        Fenêtres (en jours) pour la volatilité réalisée glissante.
        Ex: [5, 21, 63] → vol hebdo, mensuelle, trimestrielle.
    return_windows : list[int]
        Fenêtres pour les rendements glissants cumulés.
    rsi_window : int
        Période du RSI.
    bb_window : int
        Période des Bandes de Bollinger.

    Examples
    --------
    >>> fe = FeatureEngineer(vol_windows=[5, 21], return_windows=[1, 5])
    >>> features = fe.fit_transform(prices_df)
    >>> features.shape
    (3400, 18)
    """

    def __init__(
        self,
        vol_windows: list[int] = None,
        return_windows: list[int] = None,
        rsi_window: int = 14,
        bb_window: int = 20,
    ) -> None:
        self.vol_windows = vol_windows or [5, 21, 63]
        self.return_windows = return_windows or [1, 5, 21]
        self.rsi_window = rsi_window
        self.bb_window = bb_window
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        """Noms des features dans l'ordre de construction."""
        return self._feature_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        prices: pd.DataFrame,
        benchmark_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calcule l'intégralité des features à partir d'un DataFrame de prix.

        Parameters
        ----------
        prices : pd.DataFrame
            Prix de clôture ajustés, index DatetimeIndex, une colonne par actif.
        benchmark_col : str, optional
            Colonne à utiliser pour les indicateurs techniques (RSI, BB).
            Si None, utilise la première colonne.

        Returns
        -------
        pd.DataFrame
            Feature matrix : même index temporel (NaN des débuts supprimés),
            toutes les features normalisées sur [-inf, +inf] (pas encore
            standardisées — la standardisation est gérée par MarketDataProcessor).

        Notes
        -----
        Les features retournées sont brutes (non standardisées). La
        standardisation z-score est appliquée par ``MarketDataProcessor``
        après le split train/val/test pour éviter le data leakage.
        """
        benchmark = benchmark_col or prices.columns[0]
        if benchmark not in prices.columns:
            raise ValueError(f"Colonne benchmark '{benchmark}' introuvable.")

        logger.info(f"Feature engineering — benchmark : {benchmark}")
        frames: list[pd.DataFrame] = []

        # --- 1. Log-rendements multi-horizons ---
        log_ret_df = self._compute_log_returns(prices)
        frames.append(log_ret_df)

        # --- 2. Volatilité réalisée glissante ---
        vol_df = self._compute_rolling_volatility(prices)
        frames.append(vol_df)

        # --- 3. Rendements glissants cumulés ---
        cum_ret_df = self._compute_cumulative_returns(prices)
        frames.append(cum_ret_df)

        # --- 4. RSI (sur le benchmark) ---
        rsi_series = self._compute_rsi(prices[benchmark])
        frames.append(rsi_series.to_frame())

        # --- 5. Bandes de Bollinger (sur le benchmark) ---
        bb_df = self._compute_bollinger_bands(prices[benchmark])
        frames.append(bb_df)

        # --- 6. Indicateur de tendance EMA ---
        trend_series = self._compute_trend(prices[benchmark])
        frames.append(trend_series.to_frame())

        # --- 7. Ratio de volatilité court/long (régime-vol) ---
        vol_ratio_df = self._compute_volatility_ratios(prices)
        frames.append(vol_ratio_df)

        # --- 8. Corrélation glissante croisée (si multi-actifs) ---
        if prices.shape[1] > 1:
            corr_df = self._compute_rolling_correlations(prices)
            frames.append(corr_df)

        # --- Assemblage ---
        features = pd.concat(frames, axis=1)
        self._feature_names = list(features.columns)

        # Supprime les lignes initiales avec NaN (dues aux lookbacks)
        n_before = len(features)
        features = features.dropna()
        n_after = len(features)
        logger.info(
            f"Features : {features.shape[1]} colonnes | "
            f"{n_before - n_after} lignes supprimées (lookback NaN) | "
            f"{n_after} observations valides"
        )
        return features

    # ------------------------------------------------------------------
    # Feature Builders
    # ------------------------------------------------------------------

    def _compute_log_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Log-rendements sur plusieurs horizons temporels.

        log_r_t = ln(P_t / P_{t-h})

        Avantage vs rendements simples : additivité temporelle,
        meilleure approximation gaussienne pour |r| < 0.2.
        """
        frames = {}
        for ticker in prices.columns:
            for h in self.return_windows:
                col_name = f"log_ret_{ticker}_{h}d"
                frames[col_name] = np.log(prices[ticker]).diff(h)
        return pd.DataFrame(frames, index=prices.index)

    def _compute_rolling_volatility(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Volatilité réalisée glissante = std des log-rendements journaliers.

        Annualisée par multiplication par sqrt(252).
        σ_t^{(w)} = sqrt(252) × std(log_r_{t-w+1..t})
        """
        log_ret_1d = np.log(prices).diff()
        frames = {}
        for ticker in prices.columns:
            for w in self.vol_windows:
                col_name = f"realized_vol_{ticker}_{w}d"
                frames[col_name] = (
                    log_ret_1d[ticker]
                    .rolling(w, min_periods=w // 2)
                    .std()
                    * np.sqrt(252)
                )
        return pd.DataFrame(frames, index=prices.index)

    def _compute_cumulative_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Rendements simples glissants cumulés (momentum signal).

        r_t^{(w)} = P_t / P_{t-w} - 1
        """
        frames = {}
        for ticker in prices.columns:
            for h in self.return_windows:
                if h == 1:
                    continue  # Doublon avec log_ret_1d (h=1 ≈ idem)
                col_name = f"momentum_{ticker}_{h}d"
                frames[col_name] = prices[ticker].pct_change(h)
        return pd.DataFrame(frames, index=prices.index)

    def _compute_rsi(self, series: pd.Series) -> pd.Series:
        """
        Relative Strength Index (Wilder, 1978).

        RSI = 100 - 100 / (1 + RS), RS = avg_gain / avg_loss

        Interprétation pour les régimes :
        - RSI < 30 : sur-vente → possible régime bear
        - RSI > 70 : sur-achat → possible régime bull
        """
        rsi = ta.momentum.RSIIndicator(
            close=series, window=self.rsi_window, fillna=False
        )
        result = rsi.rsi()
        result.name = f"rsi_{self.rsi_window}"
        # Normalise sur [0, 1] pour l'entrée du VAE
        return result / 100.0

    def _compute_bollinger_bands(self, series: pd.Series) -> pd.DataFrame:
        """
        Bandes de Bollinger (2σ).

        Deux features extraites :
        - ``bb_width`` : (upper - lower) / middle → proxy de volatilité
        - ``bb_position`` : (close - lower) / (upper - lower) → position
          normalisée dans la bande (0 = bas, 1 = haut)
        """
        bb = ta.volatility.BollingerBands(
            close=series,
            window=self.bb_window,
            window_dev=2,
            fillna=False,
        )
        upper = bb.bollinger_hband()
        lower = bb.bollinger_lband()
        middle = bb.bollinger_mavg()

        # Évite les divisions par zéro
        width = (upper - lower) / middle.replace(0, np.nan)
        position = (series - lower) / (upper - lower).replace(0, np.nan)

        return pd.DataFrame(
            {
                f"bb_width_{self.bb_window}": width,
                f"bb_position_{self.bb_window}": position,
            },
            index=series.index,
        )

    def _compute_trend(self, series: pd.Series) -> pd.Series:
        """
        Signal de tendance EMA50 vs EMA200.

        trend = (EMA_50 - EMA_200) / EMA_200

        - Positif → tendance haussière (bull)
        - Négatif → tendance baissière (bear)

        Cet indicateur est au cœur de l'identification du Golden/Death Cross.
        """
        ema_fast = series.ewm(span=50, adjust=False).mean()
        ema_slow = series.ewm(span=200, adjust=False).mean()
        trend = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)
        trend.name = "trend_ema50_200"
        return trend

    def _compute_volatility_ratios(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Ratios de volatilité court/long (signal de changement de régime).

        ratio = vol_court_terme / vol_long_terme

        Un ratio > 1 indique une hausse récente de la volatilité
        (signal précurseur de régime bear / stress).
        """
        frames = {}
        log_ret_1d = np.log(prices).diff()
        for ticker in prices.columns:
            short_vol = (
                log_ret_1d[ticker].rolling(5, min_periods=3).std() * np.sqrt(252)
            )
            long_vol = (
                log_ret_1d[ticker].rolling(63, min_periods=30).std() * np.sqrt(252)
            )
            frames[f"vol_ratio_{ticker}"] = short_vol / long_vol.replace(0, np.nan)
        return pd.DataFrame(frames, index=prices.index)

    def _compute_rolling_correlations(
        self, prices: pd.DataFrame, window: int = 63
    ) -> pd.DataFrame:
        """
        Corrélations glissantes entre paires d'actifs (fenêtre 63j).

        En période de stress (bear), les corrélations entre classes
        d'actifs risqués augmentent (risk-on/risk-off). La corrélation
        SPY/TLT (« flight-to-quality ») est particulièrement informative.
        """
        log_ret = np.log(prices).diff()
        cols = list(prices.columns)
        frames = {}
        for i, c1 in enumerate(cols):
            for j, c2 in enumerate(cols):
                if j <= i:
                    continue
                col_name = f"corr_{c1}_{c2}_{window}d"
                frames[col_name] = (
                    log_ret[[c1, c2]]
                    .rolling(window, min_periods=window // 2)
                    .corr()
                    .unstack()[c1][c2]
                )
        return pd.DataFrame(frames, index=prices.index)
