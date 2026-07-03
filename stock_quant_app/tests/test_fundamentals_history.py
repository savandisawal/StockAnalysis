"""Screener.in history parser tests against a committed HTML snapshot.

If Screener changes its markup, these tests fail loudly instead of the
parser silently returning empty series (which would quietly degrade the
fundamental pillar to neutral defaults).
"""

from pathlib import Path

import pandas as pd
import pytest

from data.fetch_fundamentals import parse_fundamentals_history
from features.fundamentals import (
    REPORTING_LAG_DAYS_A,
    _eps_cagr_series,
    _lag_availability,
    _step_align,
)

SNAPSHOT = Path(__file__).resolve().parent / "fixtures" / "screener_sample.html"


@pytest.fixture(scope="module")
def parsed():
    return parse_fundamentals_history(SNAPSHOT.read_text(), "TESTCO")


class TestScreenerParsing:
    def test_quarterly_eps(self, parsed):
        assert len(parsed.quarterly_eps) == 5
        assert parsed.quarterly_eps.loc[pd.Timestamp("2024-03-31")] == 10.5
        assert parsed.quarterly_eps.loc[pd.Timestamp("2025-03-31")] == 13.0

    def test_annual_eps_excludes_ttm_column(self, parsed):
        assert len(parsed.annual_eps) == 5  # TTM column has no parseable date
        assert parsed.annual_eps.loc[pd.Timestamp("2025-03-31")] == 47.5

    def test_promoter_holding(self, parsed):
        assert len(parsed.promoter_pct) == 5
        assert parsed.promoter_pct.loc[pd.Timestamp("2024-09-30")] == 52.10

    def test_balance_sheet_derivations(self, parsed):
        de = parsed.annual_de()
        assert de.loc[pd.Timestamp("2021-03-31")] == pytest.approx(600 / 1500)
        roe = parsed.annual_roe()
        assert roe.loc[pd.Timestamp("2025-03-31")] == pytest.approx(475 / 2600 * 100)

    def test_ttm_eps_rolling_sum(self, parsed):
        ttm = parsed.ttm_eps()
        # Last 4 quarters: 11.0 + 11.5 + 12.0 + 13.0
        assert ttm.loc[pd.Timestamp("2025-03-31")] == pytest.approx(47.5)

    def test_comma_numbers_parsed(self, parsed):
        assert parsed.annual_net_profit.loc[pd.Timestamp("2024-03-31")] == 445.0

    def test_is_valid(self, parsed):
        assert parsed.is_valid


class TestAvailabilityLag:
    def test_annual_value_not_visible_before_lag(self, parsed):
        cagr = _eps_cagr_series(parsed.annual_eps)
        lagged = _lag_availability(cagr, REPORTING_LAG_DAYS_A)
        idx = pd.date_range("2025-03-01", "2025-07-01", freq="D")
        aligned = _step_align(lagged, idx)

        fy25_avail = pd.Timestamp("2025-03-31") + pd.Timedelta(days=REPORTING_LAG_DAYS_A)
        before = aligned.loc[fy25_avail - pd.Timedelta(days=1)]
        after = aligned.loc[fy25_avail + pd.Timedelta(days=1)]

        fy25_cagr = round(((47.5 / 38.0) ** (1 / 3) - 1) * 100, 2)
        assert after == pytest.approx(fy25_cagr)
        assert before != after  # FY25 number must not be visible earlier
