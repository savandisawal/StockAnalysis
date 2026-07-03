"""Fetch corporate announcements and filings from NSE India.

Categories that matter for prediction:
- Board meetings, quarterly results
- New orders, contracts
- Dividend declarations
- Share pledge changes
- Loan agreements, credit ratings
- Insider/promoter trades
- News verification responses
"""

from dataclasses import dataclass, field
from datetime import date, datetime

import httpx

from app.config import settings
from data.cache import get_json, set_json
from utils.logger import logger

_NSE_BASE = "https://www.nseindia.com"
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}

# Categories we care about for sentiment
_IMPORTANT_CATEGORIES = {
    "Board Meeting",
    "Financial Results",
    "Dividend",
    "New Orders",
    "Order",
    "Contract",
    "Acquisition",
    "Merger",
    "Share Pledge",
    "Pledge",
    "Loan",
    "Credit Rating",
    "Rating",
    "Buyback",
    "Bonus",
    "Split",
    "Rights Issue",
    "Insider Trading",
    "SAST",
    "Related Party",
    "News Verification",
    "Trading Window",
    "Fund Raising",
    "Outcome of Board Meeting",
}


@dataclass
class CorporateAnnouncement:
    """A single corporate announcement from NSE."""

    date: str
    category: str
    subject: str
    is_important: bool = False


@dataclass
class CorporateFilings:
    """Corporate filings summary for a stock."""

    ticker: str
    announcements: list[CorporateAnnouncement] = field(default_factory=list)
    total_count: int = 0
    fetch_date: str = ""

    def important_announcements(self) -> list[CorporateAnnouncement]:
        return [a for a in self.announcements if a.is_important]

    def summary_for_sentiment(self, max_items: int = 10) -> str:
        """Format announcements for Claude sentiment scoring."""
        important = self.important_announcements()
        items = important[:max_items] if important else self.announcements[:max_items]
        if not items:
            return ""
        lines = []
        for a in items:
            lines.append(f"[{a.date}] {a.category}: {a.subject}")
        return "\n".join(lines)

    def to_display_list(self, max_items: int = 15) -> list[dict]:
        """Format for UI display."""
        items = self.announcements[:max_items]
        return [
            {
                "Date": a.date,
                "Category": a.category,
                "Subject": a.subject,
                "Important": a.is_important,
            }
            for a in items
        ]


def _is_important(desc: str, subject: str) -> bool:
    """Check if an announcement is material for prediction."""
    text = f"{desc} {subject}".lower()
    keywords = [
        "board meeting",
        "financial result",
        "dividend",
        "new order",
        "contract",
        "acquisition",
        "merger",
        "pledge",
        "loan",
        "credit rating",
        "buyback",
        "bonus",
        "split",
        "rights issue",
        "insider",
        "sast",
        "fund raising",
        "outcome of board",
        "quarterly",
        "annual",
        "profit",
        "revenue",
        "loss",
    ]
    return any(kw in text for kw in keywords)


def fetch_corporate_announcements(
    ticker: str,
    days: int = 30,
    use_cache: bool = True,
) -> CorporateFilings:
    """Fetch recent corporate announcements from NSE for a stock.

    Args:
        ticker: NSE symbol (e.g. "VEDL", "RELIANCE")
        days: How many days back to consider
        use_cache: Check cache first

    Returns:
        CorporateFilings with announcements sorted by date (newest first).
    """
    symbol = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    cache_key = f"corp:{symbol}"

    if use_cache:
        cached = get_json(cache_key, ttl=settings.cache_ttl_sentiment)
        if cached and isinstance(cached, dict):
            filings = CorporateFilings(
                ticker=symbol,
                total_count=cached.get("total_count", 0),
                fetch_date=cached.get("fetch_date", ""),
                announcements=[CorporateAnnouncement(**a) for a in cached.get("announcements", [])],
            )
            if filings.announcements:
                return filings

    try:
        client = httpx.Client(
            headers=_NSE_HEADERS,
            follow_redirects=True,
            timeout=15,
        )
        # Get session cookies
        client.get(_NSE_BASE)

        url = f"{_NSE_BASE}/api/corporate-announcements?index=equities&symbol={symbol}"
        resp = client.get(url)
        resp.raise_for_status()
        data = resp.json()
        client.close()
    except Exception as e:
        logger.warning(f"Failed to fetch corporate announcements for {symbol}: {e}")
        return CorporateFilings(ticker=symbol)

    if not data:
        return CorporateFilings(ticker=symbol)

    cutoff = date.today()
    cutoff_start = cutoff.toordinal() - days
    announcements = []

    for item in data:
        ann_date_str = item.get("an_dt", "")
        desc = item.get("desc", "")
        subject = item.get("attchmntText", "") or item.get("smIndustry", "")

        # Parse date
        try:
            ann_dt = datetime.strptime(ann_date_str.split(" ")[0], "%d-%b-%Y").date()
        except (ValueError, IndexError):
            ann_dt = cutoff

        if ann_dt.toordinal() < cutoff_start:
            continue

        important = _is_important(desc, subject)
        announcements.append(
            CorporateAnnouncement(
                date=str(ann_dt),
                category=desc.split("\u2013")[0].strip() if "\u2013" in desc else desc.strip(),
                subject=subject.strip()[:200],
                is_important=important,
            )
        )

    filings = CorporateFilings(
        ticker=symbol,
        announcements=announcements,
        total_count=len(announcements),
        fetch_date=str(date.today()),
    )

    # Cache
    if use_cache:
        cache_data = {
            "ticker": symbol,
            "total_count": filings.total_count,
            "fetch_date": filings.fetch_date,
            "announcements": [
                {
                    "date": a.date,
                    "category": a.category,
                    "subject": a.subject,
                    "is_important": a.is_important,
                }
                for a in filings.announcements
            ],
        }
        set_json(cache_key, cache_data)

    logger.info(
        f"Fetched {len(announcements)} announcements for {symbol} "
        f"({len(filings.important_announcements())} important)"
    )
    return filings
