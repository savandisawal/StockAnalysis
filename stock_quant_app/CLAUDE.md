# Stock Quant App — Project Context

## What is this?
Production-grade NSE India next-day price range prediction system for a small team (2-5 users).

## Architecture
- **Backend**: FastAPI (async) — serves prediction API
- **Frontend**: Streamlit multi-page app — visualization + analysis
- **ML**: LightGBM quantile regression (P10/P50/P90) for calibrated price ranges
- **DB**: SQLite with WAL mode via SQLAlchemy async
- **Sentiment**: Claude Haiku via Anthropic SDK

## Triple-Pillar Feature Engine
1. **Technical** (pandas_ta): RSI, MACD, BB, EMA, ATR, ADX, Volume Z-score, Market Regime
2. **Fundamental** (Screener.in): PE/ROE/DE z-scores, EPS CAGR, Promoter holding
3. **Macro & Sentiment**: Global indices % change, India VIX, Claude-scored news sentiment

## Critical Rules
- **No look-ahead bias**: Every data fetch and feature takes `as_of_date` parameter
- **Fundamentals lagged 1 quarter**: Use last reported quarter BEFORE prediction date
- **All features must be stationary**: Returns, ratios, z-scores — never raw prices
- **Graceful degradation**: If one pillar fails, predict with remaining pillars

## Conventions
- Python 3.11+, type hints on all public functions
- Config via pydantic-settings, loaded from .env
- Logging via loguru (structured, to file + stdout)
- Tests in `tests/` using pytest + pytest-asyncio
- Linting with ruff

## Key Files
- `app/config.py` — all settings, loaded from .env
- `features/feature_builder.py` — combines all pillars, enforces look-ahead protection
- `model/train_model.py` — LightGBM quantile regression training
- `model/predict.py` — inference with confidence from interval width
