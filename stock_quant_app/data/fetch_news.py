"""Fetch news headlines from Google News RSS and MoneyControl RSS.

Returns sector/stock-relevant headlines for sentiment scoring.
No scraping of article bodies — just titles + source + timestamp.
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import quote_plus

import feedparser
import httpx

from app.config import settings
from data.cache import get_json, set_json
from utils.logger import logger


@dataclass
class NewsHeadline:
    title: str
    source: str
    published: str  # ISO date string
    url: str


# ── Google News RSS ──────────────────────────────────────────────

_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def _fetch_google_news(query: str, max_results: int = 10) -> list[NewsHeadline]:
    """Fetch headlines from Google News RSS for a search query."""
    try:
        url = f"{_GOOGLE_NEWS_RSS}?q={quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en"
        # Fetch with timeout then parse — feedparser.parse(url) has no timeout control
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        feed = feedparser.parse(resp.text)

        headlines = []
        for entry in feed.entries[:max_results]:
            # Google News wraps source in the title like "Headline - Source"
            title = entry.get("title", "")
            source = "Google News"

            # Extract source from title suffix
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title = parts[0].strip()
                source = parts[1].strip()

            # Parse published date
            published = entry.get("published", "")
            try:
                parsed = entry.get("published_parsed")
                dt = datetime(*parsed[:6]) if parsed else None
                published = dt.isoformat() if dt else date.today().isoformat()
            except Exception:
                published = date.today().isoformat()

            headlines.append(
                NewsHeadline(
                    title=title,
                    source=source,
                    published=published,
                    url=entry.get("link", ""),
                )
            )

        return headlines

    except Exception as e:
        logger.error(f"Google News RSS error for '{query}': {e}")
        return []


# ── MoneyControl RSS ─────────────────────────────────────────────

_MC_RSS_FEEDS = {
    "market": "https://www.moneycontrol.com/rss/marketreports.xml",
    "business": "https://www.moneycontrol.com/rss/business.xml",
    "stocks": "https://www.moneycontrol.com/rss/lateststnews.xml",
}


def _fetch_moneycontrol_news(
    category: str = "market",
    max_results: int = 10,
) -> list[NewsHeadline]:
    """Fetch headlines from MoneyControl RSS feeds."""
    url = _MC_RSS_FEEDS.get(category, _MC_RSS_FEEDS["market"])

    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        feed = feedparser.parse(resp.text)

        headlines = []
        for entry in feed.entries[:max_results]:
            title = entry.get("title", "").strip()
            if not title:
                continue

            published = entry.get("published", "")
            try:
                parsed = entry.get("published_parsed")
                dt = datetime(*parsed[:6]) if parsed else None
                published = dt.isoformat() if dt else date.today().isoformat()
            except Exception:
                published = date.today().isoformat()

            headlines.append(
                NewsHeadline(
                    title=title,
                    source="MoneyControl",
                    published=published,
                    url=entry.get("link", ""),
                )
            )

        return headlines

    except Exception as e:
        logger.error(f"MoneyControl RSS error for '{category}': {e}")
        return []


# ── Public API ───────────────────────────────────────────────────


def fetch_news_headlines(
    stock: str | None = None,
    sector: str | None = None,
    max_headlines: int | None = None,
    use_cache: bool = True,
) -> list[NewsHeadline]:
    """Fetch relevant news headlines for sentiment scoring.

    Combines Google News (stock/sector query) + MoneyControl (market feed).
    De-duplicates by title similarity.

    Args:
        stock: Stock name for targeted search (e.g. "Reliance Industries")
        sector: Sector name (e.g. "IT", "Banking")
        max_headlines: Max headlines to return. Defaults to config value.
        use_cache: Check cache first (4h TTL).

    Returns:
        List of NewsHeadline objects, sorted by recency.
    """
    max_headlines = max_headlines or settings.sentiment_max_headlines
    cache_key = f"news:{stock or ''}:{sector or ''}"

    if use_cache:
        cached = get_json(cache_key, settings.cache_ttl_sentiment)
        if cached is not None:
            logger.debug("Cache hit for news headlines")
            return [NewsHeadline(**h) for h in cached]

    all_headlines: list[NewsHeadline] = []

    # Google News — stock-specific search
    if stock:
        query = f"{stock} NSE India stock"
        all_headlines.extend(_fetch_google_news(query, max_results=max_headlines))

    # Google News — sector search
    if sector:
        query = f"{sector} sector India stock market"
        all_headlines.extend(_fetch_google_news(query, max_results=max_headlines // 2))

    # MoneyControl — general market news
    all_headlines.extend(_fetch_moneycontrol_news("market", max_results=max_headlines // 2))
    all_headlines.extend(_fetch_moneycontrol_news("stocks", max_results=max_headlines // 2))

    # De-duplicate by title similarity (simple word overlap check)
    seen_titles: set[str] = set()
    unique: list[NewsHeadline] = []
    for h in all_headlines:
        # Normalize: lowercase, strip punctuation
        normalized = re.sub(r"[^\w\s]", "", h.title.lower()).strip()
        if normalized and normalized not in seen_titles:
            seen_titles.add(normalized)
            unique.append(h)

    # Sort by published date (most recent first)
    unique.sort(key=lambda h: h.published, reverse=True)
    result = unique[:max_headlines]

    # Cache
    if use_cache and result:
        set_json(
            cache_key,
            [
                {"title": h.title, "source": h.source, "published": h.published, "url": h.url}
                for h in result
            ],
        )

    logger.info(f"Fetched {len(result)} unique headlines (stock={stock}, sector={sector})")
    return result
