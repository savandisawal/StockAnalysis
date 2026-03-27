"""Pillar 3 — Macro & Sentiment Features (catalyst detection).

Macro features: Daily % change of global indices (already fetched in data layer).
Sentiment: Claude Haiku scores news headlines on a -1 to +1 scale.

Features:
    - sp500_change, nasdaq_change, nifty_change (%)
    - brent_change, usdinr_change, vix_level (%)
    - news_sentiment_score (-1 to +1)
    - macro_mood (aggregate: -1 bearish, 0 neutral, +1 bullish)
"""

from dataclasses import dataclass

import anthropic

from app.config import settings
from data.cache import get_json, set_json
from data.fetch_macro import MacroSnapshot, fetch_macro_snapshot
from data.fetch_news import NewsHeadline, fetch_news_headlines
from utils.logger import logger


@dataclass
class MacroSentimentFeatures:
    """Processed Pillar 3 features ready for ML model."""
    sp500_change: float | None = None
    nasdaq_change: float | None = None
    nifty_change: float | None = None
    brent_change: float | None = None
    usdinr_change: float | None = None
    vix_value: float | None = None
    news_sentiment: float | None = None  # -1 to +1
    macro_mood: float | None = None      # aggregate score

    def to_dict(self) -> dict[str, float | None]:
        return {
            "sp500_change": self.sp500_change,
            "nasdaq_change": self.nasdaq_change,
            "nifty_change": self.nifty_change,
            "brent_change": self.brent_change,
            "usdinr_change": self.usdinr_change,
            "vix_value": self.vix_value,
            "news_sentiment": self.news_sentiment,
            "macro_mood": self.macro_mood,
        }


# ── Macro feature extraction ────────────────────────────────────

# Map display names to feature field names
_MACRO_FIELD_MAP = {
    "S&P 500": "sp500_change",
    "Nasdaq": "nasdaq_change",
    "Nifty 50": "nifty_change",
    "Brent Crude": "brent_change",
    "USD/INR": "usdinr_change",
    "India VIX": "vix_value",
}


def _extract_macro_features(snapshots: list[MacroSnapshot]) -> dict[str, float | None]:
    """Convert macro snapshots into feature dict."""
    features: dict[str, float | None] = {v: None for v in _MACRO_FIELD_MAP.values()}
    for snap in snapshots:
        field = _MACRO_FIELD_MAP.get(snap.name)
        if field and snap.is_valid:
            features[field] = snap.change_pct
    return features


def _compute_macro_mood(features: dict[str, float | None]) -> float:
    """Aggregate macro indicators into a single mood score (-1 to +1).

    Logic:
    - Positive S&P/Nasdaq/Nifty → bullish signal
    - Falling crude → bullish for India (net importer)
    - Rising USD/INR → bearish for equities
    - High VIX → bearish (fear gauge)
    """
    score = 0.0
    count = 0

    # Equity indices — positive = bullish
    for key in ["sp500_change", "nasdaq_change", "nifty_change"]:
        val = features.get(key)
        if val is not None:
            score += 1 if val > 0.3 else (-1 if val < -0.3 else 0)
            count += 1

    # Crude — falling = bullish for India
    brent = features.get("brent_change")
    if brent is not None:
        score += 1 if brent < -1.0 else (-1 if brent > 1.0 else 0)
        count += 1

    # USD/INR — rising = bearish (rupee weakening)
    usdinr = features.get("usdinr_change")
    if usdinr is not None:
        score += -1 if usdinr > 0.3 else (1 if usdinr < -0.3 else 0)
        count += 1

    # VIX — high absolute value = bearish (India VIX > 20 = elevated fear)
    vix = features.get("vix_value")
    if vix is not None:
        # VIX change_pct here, but if we had absolute: >20 = fear
        score += -1 if vix > 2.0 else (1 if vix < -2.0 else 0)
        count += 1

    if count == 0:
        return 0.0
    return round(max(-1.0, min(1.0, score / count)), 3)


# ── News sentiment via Claude ───────────────────────────────────


def score_sentiment_claude(
    headlines: list[NewsHeadline],
    sector: str | None = None,
) -> float | None:
    """Send headlines to Claude Haiku for sentiment scoring.

    Returns a score from -1 (very bearish) to +1 (very bullish).
    Returns None if API key is not configured or call fails.
    """
    if not settings.anthropic_api_key:
        logger.warning("ANTHROPIC_API_KEY not set — skipping sentiment scoring")
        return None

    if not headlines:
        return None

    # Build headline text
    headline_text = "\n".join(
        f"- {h.title} ({h.source})" for h in headlines[:settings.sentiment_max_headlines]
    )

    sector_context = f" related to the {sector} sector and" if sector else ""

    prompt = (
        f"Given these recent news headlines{sector_context} Indian/global markets:\n\n"
        f"{headline_text}\n\n"
        f"Return ONLY a JSON object with exactly two fields:\n"
        f'  "score": a number from -1.0 (very bearish) to +1.0 (very bullish)\n'
        f'  "reason": one sentence explaining the macro sentiment\n\n'
        f'Example: {{"score": 0.3, "reason": '
        f'"Mixed signals with positive US markets but rising crude prices."}}'
    )

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.sentiment_model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        # Parse the response
        import json
        text = response.content[0].text.strip()

        # Handle potential markdown code block wrapping
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        score = float(result["score"])
        reason = result.get("reason", "")

        # Clamp to [-1, 1]
        score = max(-1.0, min(1.0, score))

        logger.info(f"Sentiment score: {score} — {reason}")
        return round(score, 3)

    except Exception as e:
        logger.error(f"Claude sentiment scoring failed: {e}")
        return None


# ── Combined Pillar 3 computation ────────────────────────────────


def compute_macro_sentiment_features(
    stock: str | None = None,
    sector: str | None = None,
    use_cache: bool = True,
) -> MacroSentimentFeatures:
    """Compute all Pillar 3 features: macro % changes + news sentiment.

    Args:
        stock: Stock name for targeted news search.
        sector: Sector for sector-specific news.
        use_cache: Use cached data.

    Returns:
        MacroSentimentFeatures with all fields populated (or None if fetch failed).
    """
    result = MacroSentimentFeatures()

    # Macro indicators
    snapshots = fetch_macro_snapshot(use_cache=use_cache)
    macro_features = _extract_macro_features(snapshots)

    result.sp500_change = macro_features["sp500_change"]
    result.nasdaq_change = macro_features["nasdaq_change"]
    result.nifty_change = macro_features["nifty_change"]
    result.brent_change = macro_features["brent_change"]
    result.usdinr_change = macro_features["usdinr_change"]
    result.vix_value = macro_features["vix_value"]

    # Macro mood aggregate
    result.macro_mood = _compute_macro_mood(macro_features)

    # News sentiment — check cache first
    sentiment_cache_key = f"sentiment:{stock or ''}:{sector or ''}"
    cached_score = get_json(sentiment_cache_key, settings.cache_ttl_sentiment)

    if cached_score is not None and use_cache:
        result.news_sentiment = cached_score.get("score")
    else:
        # Fetch headlines and score
        headlines = fetch_news_headlines(stock=stock, sector=sector, use_cache=use_cache)
        if headlines:
            score = score_sentiment_claude(headlines, sector=sector)
            result.news_sentiment = score
            if score is not None:
                set_json(sentiment_cache_key, {"score": score})

    logger.info(
        f"Macro sentiment: mood={result.macro_mood}, "
        f"news_sentiment={result.news_sentiment}"
    )
    return result


# Column names for ML model
MACRO_SENTIMENT_FEATURES = [
    "sp500_change",
    "nasdaq_change",
    "nifty_change",
    "brent_change",
    "usdinr_change",
    "vix_value",
    "news_sentiment",
    "macro_mood",
]
