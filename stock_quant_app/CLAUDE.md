# Stock Quant App — Project Context

## What is this?
Production-grade NSE India next-day price range prediction system for a small team (2-5 users).
No buy/sell recommendations — probabilistic forecasts, risk metrics, and explainability only.

## Architecture
- **Backend**: FastAPI (async) — bcrypt Basic auth, slowapi rate limiting, request-ID middleware
- **Frontend**: Streamlit multi-page app — visualization + analysis
- **ML**: LightGBM quantile regression (P10/P50/P90) + CQR conformal calibration
- **DB**: SQLite WAL — raw sqlite3 for predictions/backtests/sentiment, async SQLAlchemy engine exists but unused
- **Sentiment**: Claude Haiku via Anthropic SDK
- **Deploy**: Docker (multi-stage, non-root) + docker-compose (api + ui share /data volume)

## Triple-Pillar Feature Engine (point-in-time)
1. **Technical** (pandas_ta w/ numpy fallbacks): RSI, MACD, BB, EMA, ATR, ADX, Volume Z-score, Regime
2. **Fundamental** (Screener.in quarterly/annual history): PE z-score, ROE/DE percentile, EPS CAGR, promoter change — lagged by SEBI reporting deadlines (45d quarterly / 60d annual), sector-relative per date
3. **Macro & Sentiment**: historical macro series (US series lagged +1 trading day), India VIX % change, Claude sentiment (history accumulates in sentiment_history table; never backfilled)

## Critical Rules
- **No look-ahead bias**: point-in-time joins everywhere; `tests/test_leakage.py` pins this — never weaken it
- **Train/serve parity**: both paths go through `_assemble_feature_frame` in feature_builder.py
- **Reproducibility**: LightGBM seeded + deterministic; smoke test proves identical retrains
- **All features must be stationary**: returns, ratios, z-scores — never raw prices
- **Graceful degradation**: if one pillar fails, predict with remaining pillars (neutral defaults)
- **Every served prediction is persisted** (predictions table) and outcome-backfilled next day
- **No secrets in repo**: .env is gitignored + pre-commit blocked; AUTH_USERS is bcrypt-hashed only

## ML Pipeline (model/)
- `train_model.py`: purged walk-forward CV (gap=2) → final fit on pre-calibration rows →
  CQR offsets from held-out 120-day window → bundle saved with conformal + feature_stats (schema v2)
- `conformal.py`: asymmetric per-quantile CQR offsets; applied at inference before guardrails
- `metrics.py`: pinball loss / coverage / crossing rate — single shared implementation
- `explain.py`: per-prediction SHAP (TreeExplainer, P50 model), never blocks a prediction
- `backtest.py`: walk-forward with rolling CQR, mirrors production training

## Conventions
- Python 3.13, type hints on all public functions
- Config via pydantic-settings (.env); paths derive from `settings.data_dir` (DATA_DIR env)
- Logging via loguru — JSON lines to logs/app.jsonl with request_id; `event=prediction` records
- Tests in `tests/` using pytest; offline + deterministic by default (fixtures in tests/fixtures/)
- Lint/format with ruff (`ruff check .` and `ruff format .` both enforced in CI)
- Locks: `uv pip compile pyproject.toml [--extra dev] --python-version 3.13 -o <lockfile>`
- CI + pre-commit configs live at the REPO ROOT (d:\Savan\StockAnalysis), not in stock_quant_app/

## Key Files
- `app/config.py` — all settings, loaded from .env
- `features/feature_builder.py` — shared assembly, look-ahead protection, neutral defaults
- `model/train_model.py` — training flow (CV → fit → calibrate → save)
- `model/predict.py` — inference: conformal → sort → guardrails → SHAP → structured log
- `services/prediction_store.py` — predictions/sentiment_history tables, outcome backfill, live accuracy
