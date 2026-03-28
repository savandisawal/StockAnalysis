"""NSE stock → sector mapping with company names.

Provides sector classification for z-score comparisons in Pillar 2.
This is a curated mapping of top NSE stocks. For stocks not in this map,
the fetcher tries to get sector from Screener.in.
"""

# Sector → list of (ticker, company_name) tuples
SECTOR_STOCKS: dict[str, list[tuple[str, str]]] = {
    "IT": [
        ("TCS", "Tata Consultancy Services"),
        ("INFY", "Infosys"),
        ("WIPRO", "Wipro"),
        ("HCLTECH", "HCL Technologies"),
        ("TECHM", "Tech Mahindra"),
        ("LTI", "LTIMindtree"),
        ("MPHASIS", "Mphasis"),
        ("COFORGE", "Coforge"),
        ("PERSISTENT", "Persistent Systems"),
        ("LTTS", "L&T Technology Services"),
    ],
    "Banking": [
        ("HDFCBANK", "HDFC Bank"),
        ("ICICIBANK", "ICICI Bank"),
        ("KOTAKBANK", "Kotak Mahindra Bank"),
        ("SBIN", "State Bank of India"),
        ("AXISBANK", "Axis Bank"),
        ("INDUSINDBK", "IndusInd Bank"),
        ("BANKBARODA", "Bank of Baroda"),
        ("PNB", "Punjab National Bank"),
        ("FEDERALBNK", "Federal Bank"),
        ("IDFCFIRSTB", "IDFC First Bank"),
    ],
    "NBFC": [
        ("BAJFINANCE", "Bajaj Finance"),
        ("BAJAJFINSV", "Bajaj Finserv"),
        ("CHOLAFIN", "Cholamandalam Finance"),
        ("MUTHOOTFIN", "Muthoot Finance"),
        ("MANAPPURAM", "Manappuram Finance"),
        ("SHRIRAMFIN", "Shriram Finance"),
        ("M&MFIN", "Mahindra & Mahindra Finance"),
        ("PEL", "Piramal Enterprises"),
        ("LICHSGFIN", "LIC Housing Finance"),
    ],
    "Pharma": [
        ("SUNPHARMA", "Sun Pharma"),
        ("DRREDDY", "Dr. Reddy's Labs"),
        ("CIPLA", "Cipla"),
        ("DIVISLAB", "Divi's Laboratories"),
        ("APOLLOHOSP", "Apollo Hospitals"),
        ("LUPIN", "Lupin"),
        ("AUROPHARMA", "Aurobindo Pharma"),
        ("BIOCON", "Biocon"),
        ("TORNTPHARM", "Torrent Pharma"),
        ("ALKEM", "Alkem Laboratories"),
    ],
    "Auto": [
        ("MARUTI", "Maruti Suzuki"),
        ("TATAMOTORS", "Tata Motors"),
        ("M&M", "Mahindra & Mahindra"),
        ("BAJAJ-AUTO", "Bajaj Auto"),
        ("HEROMOTOCO", "Hero MotoCorp"),
        ("EICHERMOT", "Eicher Motors"),
        ("ASHOKLEY", "Ashok Leyland"),
        ("TVSMOTOR", "TVS Motor"),
        ("BHARATFORG", "Bharat Forge"),
    ],
    "Energy": [
        ("RELIANCE", "Reliance Industries"),
        ("ONGC", "Oil & Natural Gas Corp"),
        ("IOC", "Indian Oil Corp"),
        ("BPCL", "Bharat Petroleum"),
        ("GAIL", "GAIL India"),
        ("NTPC", "NTPC"),
        ("POWERGRID", "Power Grid Corp"),
        ("ADANIGREEN", "Adani Green Energy"),
        ("TATAPOWER", "Tata Power"),
        ("COALINDIA", "Coal India"),
    ],
    "Metals": [
        ("TATASTEEL", "Tata Steel"),
        ("JSWSTEEL", "JSW Steel"),
        ("HINDALCO", "Hindalco Industries"),
        ("VEDL", "Vedanta"),
        ("SAIL", "Steel Authority of India"),
        ("NMDC", "NMDC"),
        ("NATIONALUM", "National Aluminium"),
        ("JINDALSTEL", "Jindal Steel & Power"),
        ("APLAPOLLO", "APL Apollo Tubes"),
    ],
    "FMCG": [
        ("HINDUNILVR", "Hindustan Unilever"),
        ("ITC", "ITC"),
        ("NESTLEIND", "Nestle India"),
        ("BRITANNIA", "Britannia Industries"),
        ("DABUR", "Dabur India"),
        ("MARICO", "Marico"),
        ("GODREJCP", "Godrej Consumer Products"),
        ("COLPAL", "Colgate-Palmolive India"),
        ("TATACONSUM", "Tata Consumer Products"),
        ("VBL", "Varun Beverages"),
    ],
    "Cement": [
        ("ULTRACEMCO", "UltraTech Cement"),
        ("SHREECEM", "Shree Cement"),
        ("AMBUJACEM", "Ambuja Cements"),
        ("ACC", "ACC"),
        ("DALMIACMNT", "Dalmia Bharat"),
        ("RAMCOCEM", "Ramco Cements"),
        ("JKCEMENT", "JK Cement"),
        ("BIRLASOFT", "Birlasoft"),
    ],
    "Telecom": [
        ("BHARTIARTL", "Bharti Airtel"),
        ("IDEA", "Vodafone Idea"),
        ("TTML", "Tata Teleservices"),
    ],
    "Real Estate": [
        ("DLF", "DLF"),
        ("GODREJPROP", "Godrej Properties"),
        ("OBEROIRLTY", "Oberoi Realty"),
        ("PRESTIGE", "Prestige Estates"),
        ("BRIGADE", "Brigade Enterprises"),
        ("PHOENIXLTD", "Phoenix Mills"),
        ("SOBHA", "Sobha"),
    ],
    "Insurance": [
        ("SBILIFE", "SBI Life Insurance"),
        ("HDFCLIFE", "HDFC Life Insurance"),
        ("ICICIPRULI", "ICICI Prudential Life"),
        ("LICI", "Life Insurance Corp"),
        ("NIACL", "New India Assurance"),
        ("STARHEALTH", "Star Health Insurance"),
    ],
    "Capital Goods": [
        ("LARSENTOUB", "Larsen & Toubro"),
        ("LT", "L&T"),
        ("SIEMENS", "Siemens"),
        ("ABB", "ABB India"),
        ("HAVELLS", "Havells India"),
        ("BHEL", "Bharat Heavy Electricals"),
        ("BEL", "Bharat Electronics"),
        ("CUMMINSIND", "Cummins India"),
        ("THERMAX", "Thermax"),
    ],
    "Chemicals": [
        ("PIDILITIND", "Pidilite Industries"),
        ("SRF", "SRF"),
        ("AARTI", "Aarti Industries"),
        ("DEEPAKNTR", "Deepak Nitrite"),
        ("CLEAN", "Clean Science & Technology"),
        ("ATUL", "Atul"),
        ("NAVINFLUOR", "Navin Fluorine"),
    ],
}

# Flat mappings
_TICKER_TO_SECTOR: dict[str, str] = {}
_TICKER_TO_NAME: dict[str, str] = {}
for _sector, _stocks in SECTOR_STOCKS.items():
    for _ticker, _name in _stocks:
        _TICKER_TO_SECTOR[_ticker] = _sector
        _TICKER_TO_NAME[_ticker] = _name


def get_sector(ticker: str) -> str | None:
    """Look up sector for a ticker. Returns None if not found."""
    clean = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    return _TICKER_TO_SECTOR.get(clean)


def get_company_name(ticker: str) -> str:
    """Look up company name for a ticker. Returns ticker if not found."""
    clean = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    return _TICKER_TO_NAME.get(clean, clean)


def get_sector_peers(ticker: str) -> list[str]:
    """Get peer tickers in the same sector. Excludes the input ticker."""
    sector = get_sector(ticker)
    if not sector:
        return []
    clean = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    return [t for t, _ in SECTOR_STOCKS[sector] if t != clean]


def get_all_tickers() -> list[str]:
    """All available ticker symbols, sorted."""
    return sorted(_TICKER_TO_SECTOR.keys())


def get_all_sectors() -> list[str]:
    """List all available sector names."""
    return list(SECTOR_STOCKS.keys())
