"""Tests for data layer — cache, OHLC, macro, fundamentals, news.

These are integration tests that hit external APIs. Run with:
    pytest tests/test_data_fetch.py -v

For CI, mock the external calls. For development, these validate
that scrapers and APIs are working correctly.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from data.cache import clear_all, get_dataframe, get_json, set_dataframe, set_json

# ── Cache Tests (unit, no network) ───────────────────────────────


class TestCache:
    def setup_method(self):
        clear_all()

    def test_dataframe_roundtrip(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4.0, 5.0, 6.0]})
        set_dataframe("test:df", df)
        result = get_dataframe("test:df", ttl=3600)
        assert result is not None
        assert list(result.columns) == ["A", "B"]
        assert len(result) == 3

    def test_dataframe_cache_miss(self):
        result = get_dataframe("nonexistent:key", ttl=3600)
        assert result is None

    def test_dataframe_expired(self):
        df = pd.DataFrame({"X": [1]})
        set_dataframe("test:expired", df)
        # TTL of 0 means immediately expired
        result = get_dataframe("test:expired", ttl=0)
        assert result is None

    def test_json_roundtrip(self):
        data = {"name": "S&P 500", "change": 1.5}
        set_json("test:json", data)
        result = get_json("test:json", ttl=3600)
        assert result is not None
        assert result["name"] == "S&P 500"
        assert result["change"] == 1.5

    def test_json_list_roundtrip(self):
        data = [{"a": 1}, {"b": 2}]
        set_json("test:list", data)
        result = get_json("test:list", ttl=3600)
        assert result is not None
        assert len(result) == 2

    def test_json_cache_miss(self):
        result = get_json("nonexistent:json", ttl=3600)
        assert result is None


# ── OHLC Fetch Tests (network required) ─────────────────────────


class TestFetchOHLC:
    @pytest.mark.slow
    def test_fetch_reliance(self):
        from data.fetch_ohlc import fetch_ohlc

        df = fetch_ohlc("RELIANCE", years=1, use_cache=False)
        assert not df.empty
        assert "Close" in df.columns
        assert "Volume" in df.columns
        assert len(df) > 100  # ~250 trading days per year

    @pytest.mark.slow
    def test_fetch_with_as_of_date(self):
        from data.fetch_ohlc import fetch_ohlc

        cutoff = date.today() - timedelta(days=30)
        df = fetch_ohlc("TCS", years=1, as_of_date=cutoff, use_cache=False)
        if not df.empty:
            assert df.index[-1].date() <= cutoff

    @pytest.mark.slow
    def test_ticker_normalization(self):
        from data.fetch_ohlc import fetch_ohlc

        # Should add .NS automatically
        df = fetch_ohlc("INFY", years=1, use_cache=False)
        assert not df.empty


# ── Macro Fetch Tests (network required) ─────────────────────────


class TestFetchMacro:
    @pytest.mark.slow
    def test_macro_snapshot(self):
        from data.fetch_macro import fetch_macro_snapshot

        results = fetch_macro_snapshot(use_cache=False)
        assert len(results) > 0

        # At least some should succeed
        valid = [r for r in results if r.is_valid]
        assert len(valid) >= 3, f"Only {len(valid)}/{len(results)} macro indicators valid"

        for r in valid:
            assert r.change_pct is not None
            assert -20 < r.change_pct < 20  # Sanity check — no 20%+ daily moves

    @pytest.mark.slow
    def test_macro_history(self):
        from data.fetch_macro import fetch_macro_history

        result = fetch_macro_history(days=10)
        assert isinstance(result, dict)
        assert "S&P 500" in result


# ── Fundamentals Fetch Tests (network required) ──────────────────


class TestFetchFundamentals:
    @pytest.mark.slow
    def test_fetch_reliance_fundamentals(self):
        from data.fetch_fundamentals import fetch_fundamentals

        data = fetch_fundamentals("RELIANCE", use_cache=False)
        assert data.ticker == "RELIANCE"
        # Reliance should have PE and ROE available
        assert data.pe_ratio is not None or data.roe is not None
        if data.pe_ratio:
            assert 0 < data.pe_ratio < 500  # Sanity check
        if data.promoter_holding:
            assert 0 < data.promoter_holding <= 100

    @pytest.mark.slow
    def test_fetch_with_ns_suffix(self):
        from data.fetch_fundamentals import fetch_fundamentals

        data = fetch_fundamentals("TCS.NS", use_cache=False)
        assert data.ticker == "TCS"


# ── News Fetch Tests (network required) ──────────────────────────


class TestFetchNews:
    @pytest.mark.slow
    def test_fetch_stock_news(self):
        from data.fetch_news import fetch_news_headlines

        headlines = fetch_news_headlines(stock="Reliance Industries", use_cache=False)
        assert len(headlines) > 0
        assert all(h.title for h in headlines)

    @pytest.mark.slow
    def test_fetch_sector_news(self):
        from data.fetch_news import fetch_news_headlines

        headlines = fetch_news_headlines(sector="Banking", use_cache=False)
        assert len(headlines) > 0

    @pytest.mark.slow
    def test_deduplication(self):
        from data.fetch_news import fetch_news_headlines

        headlines = fetch_news_headlines(
            stock="Infosys", sector="IT", max_headlines=20, use_cache=False
        )
        titles = [h.title.lower() for h in headlines]
        # No exact duplicate titles
        assert len(titles) == len(set(titles))
