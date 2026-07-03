"""Fetch fundamental data from Screener.in.

Extracts: PE, ROE, Debt/Equity, EPS (for CAGR calc), Promoter Holding.
Uses BeautifulSoup to parse the public stock page.

Note: Screener.in rate-limits aggressively. Always use caching and
add delays between requests. Fundamentals change quarterly so a
24-hour cache TTL is fine.
"""

import re
import time
from dataclasses import dataclass, field
from datetime import date

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from app.config import settings
from data.cache import get_json, set_json
from utils.logger import logger

_BASE_URL = "https://www.screener.in/company"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class FundamentalData:
    ticker: str
    pe_ratio: float | None = None
    sector_pe: float | None = None
    roe: float | None = None
    debt_to_equity: float | None = None
    eps_current: float | None = None
    eps_3y_ago: float | None = None
    eps_cagr_3y: float | None = None
    promoter_holding: float | None = None
    promoter_holding_change: float | None = None  # QoQ change in ppt
    sector: str | None = None
    fetch_date: str = ""
    quarterly_results: list[dict] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.pe_ratio is not None or self.roe is not None


def _clean_number(text: str) -> float | None:
    """Parse a number from screener text, handling commas and percentages."""
    if not text:
        return None
    text = text.strip().replace(",", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        return None


def _screener_ticker(ticker: str) -> str:
    """Convert NSE ticker to screener.in format (strip .NS/.BO suffix)."""
    return ticker.strip().upper().replace(".NS", "").replace(".BO", "")


def fetch_fundamentals(
    ticker: str,
    use_cache: bool = True,
    as_of_date: date | None = None,
) -> FundamentalData:
    """Fetch fundamental data for a single NSE stock from Screener.in.

    Args:
        ticker: NSE symbol (e.g. "RELIANCE" or "RELIANCE.NS")
        use_cache: Check cache first (24h TTL)
        as_of_date: Ignored for live fetch; used by feature_builder to
                    select the correct quarterly data point.

    Returns:
        FundamentalData with available fields populated.
    """
    clean_ticker = _screener_ticker(ticker)
    cache_key = f"fundamentals:{clean_ticker}"

    if use_cache:
        cached = get_json(cache_key, settings.cache_ttl_fundamentals)
        if cached is not None:
            logger.debug(f"Cache hit for {clean_ticker} fundamentals")
            return FundamentalData(**cached)

    url = f"{_BASE_URL}/{clean_ticker}/consolidated/"
    result = FundamentalData(ticker=clean_ticker, fetch_date=date.today().isoformat())

    try:
        logger.info(f"Fetching fundamentals for {clean_ticker} from Screener.in")
        resp = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)

        # If consolidated not found, try standalone
        if resp.status_code == 404:
            url = f"{_BASE_URL}/{clean_ticker}/"
            resp = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)

        if resp.status_code != 200:
            logger.warning(f"Screener returned {resp.status_code} for {clean_ticker}")
            return result

        soup = BeautifulSoup(resp.text, "lxml")

        # ── Extract from top ratios list ─────────────────────────
        ratios = soup.find("ul", id="top-ratios")
        if ratios:
            for li in ratios.find_all("li"):
                name_el = li.find("span", class_="name")
                value_el = li.find("span", class_="number")
                if not name_el or not value_el:
                    continue
                name = name_el.get_text(strip=True).lower()
                value = value_el.get_text(strip=True)

                if "stock p/e" in name:
                    result.pe_ratio = _clean_number(value)
                elif "roe" in name:
                    result.roe = _clean_number(value)
                elif "debt / equity" in name or "debt to equity" in name:
                    result.debt_to_equity = _clean_number(value)
                elif "sector pe" in name or "industry pe" in name:
                    result.sector_pe = _clean_number(value)
                elif "current price" in name or "eps" in name:
                    if "eps" in name:
                        result.eps_current = _clean_number(value)

        # ── Extract sector from company header ───────────────────
        sector_el = soup.find("a", href=re.compile(r"/screen/raw/\?sector="))
        if sector_el:
            result.sector = sector_el.get_text(strip=True)

        # ── Extract promoter holding from shareholding table ─────
        shp_section = soup.find("section", id="shareholding")
        if shp_section:
            table = shp_section.find("table")
            if table:
                rows = table.find_all("tr")
                for row in rows:
                    cells = row.find_all("td")
                    if cells and "promoter" in cells[0].get_text(strip=True).lower():
                        # Last two columns = latest and previous quarter
                        values = [_clean_number(c.get_text()) for c in cells[1:]]
                        values = [v for v in values if v is not None]
                        if values:
                            result.promoter_holding = values[-1]
                            if len(values) >= 2:
                                result.promoter_holding_change = round(values[-1] - values[-2], 2)
                        break

        # ── Extract EPS history for CAGR calculation ─────────────
        # Look in the profit-loss table for EPS row
        pl_section = soup.find("section", id="profit-loss")
        if pl_section:
            table = pl_section.find("table")
            if table:
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    header = row.find("td")
                    if header and "eps" in header.get_text(strip=True).lower():
                        eps_values = [_clean_number(c.get_text()) for c in cells[1:]]
                        eps_values = [v for v in eps_values if v is not None]
                        if eps_values:
                            result.eps_current = eps_values[-1]
                            # 3Y CAGR: need at least 4 annual values
                            if len(eps_values) >= 4:
                                eps_old = eps_values[-4]
                                eps_new = eps_values[-1]
                                if eps_old and eps_old > 0 and eps_new > 0:
                                    result.eps_3y_ago = eps_old
                                    result.eps_cagr_3y = round(
                                        ((eps_new / eps_old) ** (1 / 3) - 1) * 100, 2
                                    )
                        break

        logger.info(
            f"Fundamentals for {clean_ticker}: PE={result.pe_ratio}, "
            f"ROE={result.roe}, D/E={result.debt_to_equity}, "
            f"Promoter={result.promoter_holding}%"
        )

    except httpx.TimeoutException:
        logger.error(f"Timeout fetching {clean_ticker} from Screener.in")
    except Exception as e:
        logger.error(f"Error fetching {clean_ticker} fundamentals: {e}")

    # Cache result
    if use_cache and result.is_valid:
        set_json(
            cache_key,
            {
                "ticker": result.ticker,
                "pe_ratio": result.pe_ratio,
                "sector_pe": result.sector_pe,
                "roe": result.roe,
                "debt_to_equity": result.debt_to_equity,
                "eps_current": result.eps_current,
                "eps_3y_ago": result.eps_3y_ago,
                "eps_cagr_3y": result.eps_cagr_3y,
                "promoter_holding": result.promoter_holding,
                "promoter_holding_change": result.promoter_holding_change,
                "sector": result.sector,
                "fetch_date": result.fetch_date,
                "quarterly_results": result.quarterly_results,
            },
        )

    return result


# ── Point-in-time fundamental history ────────────────────────────


@dataclass
class FundamentalHistory:
    """Historical fundamental series parsed from Screener.in tables.

    All series are indexed by period-end date (quarter end or fiscal year
    end). Availability lags are applied downstream by the feature layer.
    """

    ticker: str
    quarterly_eps: "pd.Series" = None  # EPS per quarter
    promoter_pct: "pd.Series" = None  # promoter holding % per quarter
    annual_eps: "pd.Series" = None  # EPS per fiscal year
    annual_net_profit: "pd.Series" = None
    annual_equity: "pd.Series" = None  # equity capital + reserves
    annual_borrowings: "pd.Series" = None
    fetch_date: str = ""

    @property
    def is_valid(self) -> bool:
        return any(
            s is not None and len(s) > 0
            for s in (self.quarterly_eps, self.annual_eps, self.promoter_pct)
        )

    def ttm_eps(self) -> "pd.Series":
        """Trailing-12-month EPS: rolling sum of the last 4 quarterly EPS."""
        if self.quarterly_eps is None or len(self.quarterly_eps) < 4:
            return pd.Series(dtype=float)
        return self.quarterly_eps.rolling(4).sum().dropna()

    def annual_roe(self) -> "pd.Series":
        """ROE % per fiscal year = net profit / (equity capital + reserves)."""
        if self.annual_net_profit is None or self.annual_equity is None:
            return pd.Series(dtype=float)
        eq = self.annual_equity.replace(0, pd.NA)
        return ((self.annual_net_profit / eq) * 100).dropna().astype(float)

    def annual_de(self) -> "pd.Series":
        """Debt/Equity per fiscal year = borrowings / (equity capital + reserves)."""
        if self.annual_borrowings is None or self.annual_equity is None:
            return pd.Series(dtype=float)
        eq = self.annual_equity.replace(0, pd.NA)
        return (self.annual_borrowings / eq).dropna().astype(float)


def _parse_period_columns(headers: list[str]) -> list["pd.Timestamp | None"]:
    """Parse Screener column headers like 'Mar 2024' into period-end dates."""
    out = []
    for h in headers:
        h = h.strip()
        try:
            ts = pd.to_datetime(h, format="%b %Y")
            out.append(ts + pd.offsets.MonthEnd(0))
        except (ValueError, TypeError):
            out.append(None)
    return out


def _parse_screener_section_table(soup: "BeautifulSoup", section_id: str) -> "pd.DataFrame":
    """Parse a Screener.in section's data table.

    Returns DataFrame with index = lowercased row labels and columns =
    period-end Timestamps. Cells are parsed floats (NaN where unparseable).
    Empty DataFrame if the section/table is missing.
    """
    section = soup.find("section", id=section_id)
    if not section:
        return pd.DataFrame()
    table = section.find("table")
    if not table:
        return pd.DataFrame()

    header_row = table.find("thead")
    if not header_row:
        return pd.DataFrame()
    headers = [th.get_text(strip=True) for th in header_row.find_all("th")]
    # First header cell is the row-label column
    periods = _parse_period_columns(headers[1:])

    rows: dict[str, list] = {}
    body = table.find("tbody") or table
    for tr in body.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 2:
            continue
        label = cells[0].get_text(strip=True).lower().rstrip("+").strip()
        if not label:
            continue
        values = [_clean_number(c.get_text()) for c in cells[1:]]
        rows[label] = values

    if not rows or not any(p is not None for p in periods):
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    n = min(df.shape[1], len(periods))
    df = df.iloc[:, :n]
    df.columns = periods[:n]
    # Drop columns whose header didn't parse to a date (e.g. 'TTM').
    # None headers become NaT once pandas coerces to a DatetimeIndex.
    df = df.loc[:, [c for c in df.columns if pd.notna(c)]]
    return df


def _row_series(table: "pd.DataFrame", *label_substrings: str) -> "pd.Series":
    """Extract the first row whose label contains any substring, as a float Series."""
    if table.empty:
        return pd.Series(dtype=float)
    for sub in label_substrings:
        for label in table.index:
            if sub in label:
                s = table.loc[label].dropna()
                s.index = pd.DatetimeIndex(s.index)
                return s.sort_index().astype(float)
    return pd.Series(dtype=float)


def _series_to_jsonable(s: "pd.Series") -> dict:
    return {} if s is None or s.empty else {k.date().isoformat(): float(v) for k, v in s.items()}


def _series_from_jsonable(d: dict) -> "pd.Series":
    if not d:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(k): float(v) for k, v in d.items()})
    return s.sort_index()


def fetch_fundamentals_history(ticker: str, use_cache: bool = True) -> FundamentalHistory:
    """Fetch point-in-time fundamental history for a stock from Screener.in.

    Parses the quarterly results, shareholding, profit & loss, and balance
    sheet tables into dated series. Cached for 7 days (quarterly cadence).
    """
    clean_ticker = _screener_ticker(ticker)
    cache_key = f"fundamentals:history:{clean_ticker}"

    if use_cache:
        cached = get_json(cache_key, settings.cache_ttl_fundamental_history)
        if cached is not None:
            return FundamentalHistory(
                ticker=clean_ticker,
                quarterly_eps=_series_from_jsonable(cached.get("quarterly_eps", {})),
                promoter_pct=_series_from_jsonable(cached.get("promoter_pct", {})),
                annual_eps=_series_from_jsonable(cached.get("annual_eps", {})),
                annual_net_profit=_series_from_jsonable(cached.get("annual_net_profit", {})),
                annual_equity=_series_from_jsonable(cached.get("annual_equity", {})),
                annual_borrowings=_series_from_jsonable(cached.get("annual_borrowings", {})),
                fetch_date=cached.get("fetch_date", ""),
            )

    result = FundamentalHistory(ticker=clean_ticker, fetch_date=date.today().isoformat())
    url = f"{_BASE_URL}/{clean_ticker}/consolidated/"

    try:
        logger.info(f"Fetching fundamental history for {clean_ticker} from Screener.in")
        resp = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code == 404:
            url = f"{_BASE_URL}/{clean_ticker}/"
            resp = httpx.get(url, headers=_HEADERS, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            logger.warning(f"Screener returned {resp.status_code} for {clean_ticker} history")
            return result

        result = parse_fundamentals_history(resp.text, clean_ticker)

    except httpx.TimeoutException:
        logger.error(f"Timeout fetching {clean_ticker} history from Screener.in")
    except Exception as e:
        logger.error(f"Error fetching {clean_ticker} fundamental history: {e}")

    if use_cache and result.is_valid:
        set_json(
            cache_key,
            {
                "quarterly_eps": _series_to_jsonable(result.quarterly_eps),
                "promoter_pct": _series_to_jsonable(result.promoter_pct),
                "annual_eps": _series_to_jsonable(result.annual_eps),
                "annual_net_profit": _series_to_jsonable(result.annual_net_profit),
                "annual_equity": _series_to_jsonable(result.annual_equity),
                "annual_borrowings": _series_to_jsonable(result.annual_borrowings),
                "fetch_date": result.fetch_date,
            },
        )

    return result


def parse_fundamentals_history(html: str, ticker: str) -> FundamentalHistory:
    """Parse a Screener.in company page into a FundamentalHistory.

    Split out from the fetch so parsing can be unit-tested against a
    committed HTML snapshot without network access.
    """
    soup = BeautifulSoup(html, "lxml")
    result = FundamentalHistory(ticker=ticker, fetch_date=date.today().isoformat())

    quarters = _parse_screener_section_table(soup, "quarters")
    result.quarterly_eps = _row_series(quarters, "eps in rs", "eps")

    shareholding = _parse_screener_section_table(soup, "quarterly-shp")
    if shareholding.empty:
        shareholding = _parse_screener_section_table(soup, "shareholding")
    result.promoter_pct = _row_series(shareholding, "promoter")

    pl = _parse_screener_section_table(soup, "profit-loss")
    result.annual_eps = _row_series(pl, "eps in rs", "eps")
    result.annual_net_profit = _row_series(pl, "net profit")

    bs = _parse_screener_section_table(soup, "balance-sheet")
    equity_capital = _row_series(bs, "equity capital", "share capital")
    reserves = _row_series(bs, "reserves")
    if not equity_capital.empty or not reserves.empty:
        result.annual_equity = equity_capital.add(reserves, fill_value=0.0)
    result.annual_borrowings = _row_series(bs, "borrowings")

    def _n(s) -> int:
        return 0 if s is None else len(s)

    logger.info(
        f"Fundamental history for {ticker}: "
        f"{_n(result.quarterly_eps)} quarters EPS, "
        f"{_n(result.annual_eps)} years EPS, "
        f"{_n(result.promoter_pct)} quarters promoter"
    )
    return result


def fetch_fundamentals_batch(
    tickers: list[str],
    use_cache: bool = True,
) -> dict[str, FundamentalData]:
    """Fetch fundamentals for multiple tickers with rate limiting."""
    results = {}
    for ticker in tickers:
        results[_screener_ticker(ticker)] = fetch_fundamentals(ticker, use_cache=use_cache)
        time.sleep(2.0)  # Screener.in rate limit — be respectful
    return results
