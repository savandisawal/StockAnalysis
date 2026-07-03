"""Pillar 2 — Fundamental Features for long-term value assessment.

Converts raw fundamental data (from Screener.in) into z-scores and
percentiles relative to sector peers. This makes features comparable
across stocks and sectors.

Features:
    - PE z-score vs sector
    - ROE percentile vs sector
    - Debt/Equity percentile vs sector
    - EPS CAGR 3Y (raw — already comparable)
    - Promoter holding change QoQ (raw — already comparable)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.fetch_fundamentals import (
    FundamentalData,
    FundamentalHistory,
    fetch_fundamentals,
    fetch_fundamentals_history,
)
from utils.logger import logger
from utils.sectors import get_sector_peers

# SEBI LODR reporting deadlines: quarterly results within 45 days of the
# quarter end, audited annual results within 60 days of the fiscal year end.
# A period's numbers only become model-visible after this lag.
REPORTING_LAG_DAYS_Q = 45
REPORTING_LAG_DAYS_A = 60

_MAX_PEERS = 8


@dataclass
class FundamentalFeatures:
    """Processed fundamental features ready for ML model."""

    pe_zscore: float | None = None  # vs sector median
    roe_percentile: float | None = None  # vs sector (0-1)
    de_percentile: float | None = None  # vs sector (0-1), lower = better
    eps_cagr_3y: float | None = None  # raw percentage
    promoter_change: float | None = None  # QoQ change in percentage points
    sector: str | None = None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "pe_zscore": self.pe_zscore,
            "roe_percentile": self.roe_percentile,
            "de_percentile": self.de_percentile,
            "eps_cagr_3y": self.eps_cagr_3y,
            "promoter_change": self.promoter_change,
        }


def _zscore(value: float, values: list[float]) -> float | None:
    """Compute z-score of value relative to a list of values."""
    if not values or len(values) < 2:
        return None
    import numpy as np

    arr = [v for v in values if v is not None]
    if len(arr) < 2:
        return None
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return round(float((value - mean) / std), 3)


def _percentile(value: float, values: list[float]) -> float | None:
    """Compute percentile rank (0-1) of value within a list."""
    valid = sorted([v for v in values if v is not None])
    if not valid:
        return None
    count_below = sum(1 for v in valid if v < value)
    return round(count_below / len(valid), 3)


def compute_fundamental_features(
    ticker: str,
    use_cache: bool = True,
) -> FundamentalFeatures:
    """Compute Pillar 2 features for a stock by comparing to sector peers.

    Fetches fundamentals for the target stock AND its sector peers,
    then computes z-scores and percentiles.

    Args:
        ticker: NSE ticker (e.g. "RELIANCE")
        use_cache: Use cached fundamental data.

    Returns:
        FundamentalFeatures with sector-relative metrics.
    """
    result = FundamentalFeatures()

    # Fetch target stock fundamentals
    target = fetch_fundamentals(ticker, use_cache=use_cache)
    if not target.is_valid:
        logger.warning(f"No fundamental data for {ticker}")
        return result

    result.sector = target.sector
    result.eps_cagr_3y = target.eps_cagr_3y
    result.promoter_change = target.promoter_holding_change

    # Fetch sector peers for comparison
    peers = get_sector_peers(ticker)
    if not peers:
        logger.warning(f"No sector peers found for {ticker}, using raw values only")
        return result

    # Fetch peer fundamentals (cached — won't hammer Screener.in)
    peer_data: list[FundamentalData] = []
    for peer in peers[:8]:  # Limit to top 8 peers to control API calls
        pf = fetch_fundamentals(peer, use_cache=True)
        if pf.is_valid:
            peer_data.append(pf)

    if not peer_data:
        logger.warning(f"No valid peer data for {ticker} sector comparison")
        return result

    # ── PE z-score vs sector ─────────────────────────────────
    if target.pe_ratio is not None:
        peer_pes = [p.pe_ratio for p in peer_data if p.pe_ratio is not None]
        all_pes = peer_pes + [target.pe_ratio]
        result.pe_zscore = _zscore(target.pe_ratio, all_pes)

    # ── ROE percentile vs sector ─────────────────────────────
    if target.roe is not None:
        peer_roes = [p.roe for p in peer_data if p.roe is not None]
        all_roes = peer_roes + [target.roe]
        result.roe_percentile = _percentile(target.roe, all_roes)

    # ── Debt/Equity percentile vs sector (lower is better) ───
    if target.debt_to_equity is not None:
        peer_des = [p.debt_to_equity for p in peer_data if p.debt_to_equity is not None]
        all_des = peer_des + [target.debt_to_equity]
        # Invert: low D/E → high percentile (good)
        raw_pct = _percentile(target.debt_to_equity, all_des)
        if raw_pct is not None:
            result.de_percentile = round(1.0 - raw_pct, 3)

    logger.info(
        f"Fundamental features for {ticker}: PE_z={result.pe_zscore}, "
        f"ROE_pct={result.roe_percentile}, D/E_pct={result.de_percentile}"
    )
    return result


# ── Point-in-time (per-date) feature construction ────────────────


def _lag_availability(series: pd.Series, lag_days: int) -> pd.Series:
    """Shift a period-end-indexed series to its public-availability dates."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    s = series.copy()
    s.index = s.index + pd.Timedelta(days=lag_days)
    return s


def _step_align(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a sparse availability-dated series onto a daily index.

    Dates before the first available value stay NaN (genuinely unknown then).
    """
    if series is None or series.empty:
        return pd.Series(index=index, dtype=float)
    combined = series.reindex(series.index.union(index)).ffill()
    return combined.reindex(index)


def _eps_cagr_series(annual_eps: pd.Series) -> pd.Series:
    """3Y EPS CAGR at each fiscal-year point (needs 4 annual values)."""
    if annual_eps is None or len(annual_eps) < 4:
        return pd.Series(dtype=float)
    out = {}
    values = annual_eps.sort_index()
    for i in range(3, len(values)):
        old, new = values.iloc[i - 3], values.iloc[i]
        if old and old > 0 and new > 0:
            out[values.index[i]] = round(((new / old) ** (1 / 3) - 1) * 100, 2)
    return pd.Series(out, dtype=float)


def _pe_series(history: FundamentalHistory, close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Daily PE = close(t) / TTM EPS available at t."""
    ttm = _step_align(_lag_availability(history.ttm_eps(), REPORTING_LAG_DAYS_Q), index)
    close_aligned = close.reindex(index).ffill()
    pe = close_aligned / ttm.replace(0, np.nan)
    return pe.where((ttm > 0), np.nan)


def _fetch_peer_data(
    ticker: str, index: pd.DatetimeIndex, use_cache: bool
) -> dict[str, tuple[FundamentalHistory, pd.Series]]:
    """Fetch fundamental histories + close prices for sector peers."""
    from data.fetch_ohlc import fetch_ohlc  # local import to avoid cycle at module load

    peers = get_sector_peers(ticker)[:_MAX_PEERS]
    years = max(2, int(np.ceil((index[-1] - index[0]).days / 365.25)) + 1)

    out: dict[str, tuple[FundamentalHistory, pd.Series]] = {}
    for peer in peers:
        try:
            hist = fetch_fundamentals_history(peer, use_cache=use_cache)
            if not hist.is_valid:
                continue
            ohlc = fetch_ohlc(peer, years=years)
            if ohlc.empty:
                continue
            out[peer] = (hist, ohlc["Close"])
        except Exception as e:
            logger.warning(f"Peer data failed for {peer}: {e}")
    return out


def build_fundamental_history_features(
    ticker: str,
    index: pd.DatetimeIndex,
    close: pd.Series | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Build leakage-free per-date fundamental features aligned to `index`.

    Every quarterly/annual figure only enters the features after its SEBI
    reporting deadline (45d quarterly / 60d annual), then forward-fills.
    Sector-relative z-scores/percentiles are computed per date against
    peers' own point-in-time values.

    Sets df.attrs["point_in_time"] = False when the Screener.in history was
    unavailable and columns fell back to NaN (filled with neutral defaults
    downstream).
    """
    df = pd.DataFrame(index=index, columns=FUNDAMENTAL_FEATURES, dtype=float)
    df.attrs["point_in_time"] = False

    try:
        target = fetch_fundamentals_history(ticker, use_cache=use_cache)
    except Exception as e:
        logger.warning(f"Fundamental history fetch failed for {ticker}: {e}")
        return df

    if not target.is_valid:
        logger.warning(f"No fundamental history for {ticker} — using neutral defaults")
        return df

    df.attrs["point_in_time"] = True

    # ── Raw (already comparable) features ────────────────────
    df["eps_cagr_3y"] = _step_align(
        _lag_availability(_eps_cagr_series(target.annual_eps), REPORTING_LAG_DAYS_A), index
    )
    if target.promoter_pct is not None and len(target.promoter_pct) >= 2:
        promoter_change = target.promoter_pct.sort_index().diff().dropna().round(2)
        df["promoter_change"] = _step_align(
            _lag_availability(promoter_change, REPORTING_LAG_DAYS_Q), index
        )

    # ── Sector-relative features (need peers) ────────────────
    peer_data = _fetch_peer_data(ticker, index, use_cache)
    if not peer_data:
        logger.warning(f"No peer data for {ticker} — sector-relative features stay neutral")
        return df

    # PE z-score: daily PE for target + peers, z-score per date
    if close is not None:
        pe_frame = pd.DataFrame(index=index)
        pe_frame["__target__"] = _pe_series(target, close, index)
        for peer, (hist, peer_close) in peer_data.items():
            pe_frame[peer] = _pe_series(hist, peer_close, index)
        df["pe_zscore"] = _cross_sectional_zscore(pe_frame, "__target__")

    # ROE / D-E percentiles: annual step series per entity
    roe_frame = pd.DataFrame(index=index)
    de_frame = pd.DataFrame(index=index)
    roe_frame["__target__"] = _step_align(
        _lag_availability(target.annual_roe(), REPORTING_LAG_DAYS_A), index
    )
    de_frame["__target__"] = _step_align(
        _lag_availability(target.annual_de(), REPORTING_LAG_DAYS_A), index
    )
    for peer, (hist, _) in peer_data.items():
        roe_frame[peer] = _step_align(
            _lag_availability(hist.annual_roe(), REPORTING_LAG_DAYS_A), index
        )
        de_frame[peer] = _step_align(
            _lag_availability(hist.annual_de(), REPORTING_LAG_DAYS_A), index
        )

    df["roe_percentile"] = _cross_sectional_percentile(roe_frame, "__target__")
    # Low D/E is good — invert so higher percentile = healthier balance sheet
    de_pct = _cross_sectional_percentile(de_frame, "__target__")
    df["de_percentile"] = (1.0 - de_pct).round(3)

    return df


def _cross_sectional_zscore(frame: pd.DataFrame, target_col: str) -> pd.Series:
    """Per-date z-score of target column vs all columns (incl. target)."""
    valid = frame.notna().sum(axis=1)
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0)
    z = (frame[target_col] - mean) / std.replace(0, np.nan)
    z = z.where(valid >= 2)
    return z.fillna(pd.Series(0.0, index=frame.index)).where(frame[target_col].notna()).round(3)


def _cross_sectional_percentile(frame: pd.DataFrame, target_col: str) -> pd.Series:
    """Per-date percentile rank (0-1) of target among all columns (incl. target)."""
    target = frame[target_col]
    others = frame
    count_below = others.lt(target, axis=0).sum(axis=1)
    valid = others.notna().sum(axis=1)
    pct = (count_below / valid.replace(0, np.nan)).where(valid >= 2)
    return pct.where(target.notna()).round(3)


# Column names for ML model
FUNDAMENTAL_FEATURES = [
    "pe_zscore",
    "roe_percentile",
    "de_percentile",
    "eps_cagr_3y",
    "promoter_change",
]
