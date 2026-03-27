"""NSE stock → sector mapping.

Provides sector classification for z-score comparisons in Pillar 2.
This is a curated mapping of top NSE stocks. For stocks not in this map,
the fetcher tries to get sector from Screener.in.
"""

# Sector → list of tickers (top constituents)
SECTOR_STOCKS: dict[str, list[str]] = {
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTI", "MPHASIS",
        "COFORGE", "PERSISTENT", "LTTS",
    ],
    "Banking": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "AXISBANK",
        "INDUSINDBK", "BANKBARODA", "PNB", "FEDERALBNK", "IDFCFIRSTB",
    ],
    "NBFC": [
        "BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN", "MANAPPURAM",
        "SHRIRAMFIN", "M&MFIN", "PEL", "LICHSGFIN",
    ],
    "Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
        "LUPIN", "AUROPHARMA", "BIOCON", "TORNTPHARM", "ALKEM",
    ],
    "Auto": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
        "EICHERMOT", "ASHOKLEY", "TVSMOTOR", "BHARATFORG",
    ],
    "Energy": [
        "RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "NTPC",
        "POWERGRID", "ADANIGREEN", "TATAPOWER", "COALINDIA",
    ],
    "Metals": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "VEDL", "SAIL",
        "NMDC", "NATIONALUM", "JINDALSTEL", "APLAPOLLO",
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "MARICO", "GODREJCP", "COLPAL", "TATACONSUM", "VBL",
    ],
    "Cement": [
        "ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC", "DALMIACMNT",
        "RAMCOCEM", "JKCEMENT", "BIRLASOFT",
    ],
    "Telecom": [
        "BHARTIARTL", "IDEA", "TTML",
    ],
    "Real Estate": [
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "BRIGADE",
        "PHOENIXLTD", "SOBHA",
    ],
    "Insurance": [
        "SBILIFE", "HDFCLIFE", "ICICIPRULI", "LICI", "NIACL",
        "STARHEALTH",
    ],
    "Capital Goods": [
        "LARSENTOUB", "LT", "SIEMENS", "ABB", "HAVELLS", "BHEL",
        "BEL", "CUMMINSIND", "THERMAX",
    ],
    "Chemicals": [
        "PIDILITIND", "SRF", "AARTI", "DEEPAKNTR", "CLEAN",
        "ATUL", "NAVINFLUOR",
    ],
}

# Inverted index: ticker → sector
_TICKER_TO_SECTOR: dict[str, str] = {}
for _sector, _tickers in SECTOR_STOCKS.items():
    for _t in _tickers:
        _TICKER_TO_SECTOR[_t] = _sector


def get_sector(ticker: str) -> str | None:
    """Look up sector for a ticker. Returns None if not found."""
    clean = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    return _TICKER_TO_SECTOR.get(clean)


def get_sector_peers(ticker: str) -> list[str]:
    """Get peer tickers in the same sector. Excludes the input ticker."""
    sector = get_sector(ticker)
    if not sector:
        return []
    clean = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    return [t for t in SECTOR_STOCKS[sector] if t != clean]


def get_all_sectors() -> list[str]:
    """List all available sector names."""
    return list(SECTOR_STOCKS.keys())
