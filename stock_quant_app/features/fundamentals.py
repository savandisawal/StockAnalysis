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

from data.fetch_fundamentals import FundamentalData, fetch_fundamentals
from utils.logger import logger
from utils.sectors import get_sector_peers


@dataclass
class FundamentalFeatures:
    """Processed fundamental features ready for ML model."""
    pe_zscore: float | None = None        # vs sector median
    roe_percentile: float | None = None   # vs sector (0-1)
    de_percentile: float | None = None    # vs sector (0-1), lower = better
    eps_cagr_3y: float | None = None      # raw percentage
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


# Column names for ML model
FUNDAMENTAL_FEATURES = [
    "pe_zscore",
    "roe_percentile",
    "de_percentile",
    "eps_cagr_3y",
    "promoter_change",
]
