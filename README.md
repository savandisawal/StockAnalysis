# Stock Quant App

Production-grade **next-day price range prediction system** for NSE India stocks, powered by LightGBM quantile regression with conformal calibration and a leakage-free triple-pillar feature engine.

![Python 3.13](https://img.shields.io/badge/python-3.13-blue)
![LightGBM](https://img.shields.io/badge/ML-LightGBM%20%2B%20CQR-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)

> **Not investment advice.** The system produces probabilistic forecasts, risk metrics,
> and explanations — never buy/sell recommendations. Use it as decision support with
> human oversight.

## What It Does

For any NSE-listed stock, the app:

1. **Predicts tomorrow's price range** (P10 / P50 / P90) using quantile regression with
   CQR conformal calibration for statistically valid interval coverage
2. **Scores market sentiment** via Claude AI analysis of news + corporate filings
3. **Explains every prediction** with per-prediction SHAP values, a signal summary, and
   a confidence breakdown
4. **Tracks accuracy honestly** — every served prediction is persisted and scored against
   the real next-day close (Truth Dashboard live tab), alongside walk-forward backtests

## Architecture

```
stock_quant_app/
|-- app/              # FastAPI backend (bcrypt auth, config, rate limiting, request IDs)
|-- data/             # Data fetchers (NSE OHLCV, point-in-time fundamentals/macro, news)
|-- features/         # Triple-pillar feature engine (point-in-time, leakage-free)
|-- model/            # Training (purged walk-forward CV), CQR conformal, SHAP, backtest,
|                     # metrics (pinball loss), model registry
|-- services/         # Prediction service + store (persistence, outcome backfill), scheduler
|-- scripts/          # hash_password (auth), make_fixture (test data)
|-- ui_streamlit/     # Streamlit multi-page frontend
|   |-- Home.py           # Main page: prediction, safeguards, explainer, fundamentals
|   |-- pages/
|       |-- 1_Active_Analysis.py   # Technical analysis + "Why this prediction" SHAP chart
|       |-- 2_Global_Pulse.py      # Live macro indicators dashboard
|       |-- 3_Truth_Dashboard.py   # Live served-prediction accuracy + backtest (separate tabs)
|-- utils/            # Logging (JSON), NSE holidays, sector mappings
|-- tests/            # Offline deterministic tests incl. smoke train + leakage regression
|-- Dockerfile        # Multi-stage, non-root
|-- docker-compose.yml# api + ui sharing a /data volume
```

## Triple-Pillar Feature Engine (point-in-time)

Every feature is **point-in-time correct**: a training row at date *t* only contains
information that was publicly available before the NSE close on day *t*.

| Pillar | Features | Point-in-time mechanics |
|--------|----------|------------------------|
| **Technical** | RSI, MACD, Bollinger %B, EMA 20/50/200, ADX, Volume Z-Score, ATR, Market Regime | Causal rolling windows on OHLCV |
| **Fundamental** | PE Z-Score, ROE/D-E Percentile, EPS CAGR, Promoter Holding change | Screener.in quarterly/annual history, lagged by SEBI reporting deadlines (45d/60d), sector-relative vs peers' own histories |
| **Macro & Sentiment** | S&P 500, Nasdaq, Nifty, Brent, USD/INR, India VIX changes + News Sentiment | Real historical series; US series lagged 1 trading day (only the completed prior US session is visible at IST close). Sentiment history accumulates from daily Claude snapshots — never backfilled |

## ML Model

- **LightGBM Quantile Regression** — three models per stock (P10 / P50 / P90)
- **Purged walk-forward CV** (expanding window, 2-row purge/embargo) for honest
  out-of-sample pinball loss and coverage metrics
- **CQR conformal calibration**: per-quantile asymmetric offsets computed on a held-out
  120-day calibration window give distribution-free ~80% interval coverage
- **Fully reproducible**: seeded, deterministic LightGBM — same data trains
  byte-identical models (proven by a CI test)
- **Per-prediction SHAP** (TreeExplainer on the P50 model) stored with every prediction
- Confidence score derived from calibrated interval width relative to ATR
- Direction classification: Bullish (>+0.3%), Bearish (<-0.3%), Neutral

## Prediction Safeguards

| Safeguard | Trigger | Effect |
|-----------|---------|--------|
| **Guardrails** | Predicted change > 2x ATR | Caps prediction to realistic range |
| **Feature Drift** | 3+ features outside the training distribution (per-bundle stats: \|z\| > 3 or outside [p01, p99]) | Warns model may not generalize |
| **Model Staleness** | Model trained > 7 days ago | Warns to retrain |
| **Circuit Breakers** | VIX change > 30% or Volume Z > 4 | Flags prediction as unreliable |
| **Calibration** | Backtest accuracy < 50% or coverage < 60% | Adjusts confidence downward |
| **Ensemble Sanity** | P10/P50/P90 disordered or spread > 8% | Reduces confidence by 30% |

## Data Sources

| Data | Source | Method |
|------|--------|--------|
| NSE OHLCV (primary) | [jugaad-data](https://github.com/jugaad-py/jugaad-data) | Direct NSE scraping |
| NSE OHLCV (fallback) | yfinance | Yahoo Finance API |
| Fundamentals (history) | Screener.in | HTML scraping (quarterly/annual tables) |
| News Headlines | Google News RSS | RSS feed parsing |
| Corporate Filings | NSE India API | `/api/corporate-announcements` |
| Macro Indicators | Yahoo Finance | Historical + live indices, commodities, forex |
| Sentiment Scoring | Anthropic Claude Haiku | LLM-based headline + filing analysis |

## Quick Start

### Prerequisites

- Python 3.13 (local) or Docker
- [Anthropic API key](https://console.anthropic.com/) (for sentiment scoring)

### Local Setup

```bash
git clone https://github.com/savandisawal/StockAnalysis.git
cd StockAnalysis/stock_quant_app

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# 1. Add your ANTHROPIC_API_KEY
# 2. Generate hashed credentials (plaintext passwords are not accepted):
python -m scripts.hash_password
#    → paste output into .env as AUTH_USERS=youruser:<hash>
```

### Run locally

```bash
streamlit run ui_streamlit/Home.py            # Streamlit UI
uvicorn app.main:app --reload --port 8000     # FastAPI backend
```

### Run with Docker (recommended for deployment)

```bash
cd stock_quant_app
cp .env.example .env    # fill in ANTHROPIC_API_KEY + hashed AUTH_USERS, set ENVIRONMENT=prod
docker compose up -d --build
# API on :8000, UI on :8501; all state (DBs, models, logs) on the shared appdata volume
```

In `ENVIRONMENT=prod` the API refuses to start unless `AUTH_USERS` contains bcrypt
hashes — there are no default credentials.

### Usage

1. Open http://localhost:8501
2. Select a stock from the sidebar (filter by sector or type any NSE ticker)
3. Click **Train Model** to train the quantile regression models
4. View the next-day prediction with confidence score and safeguard warnings
5. See **Why this prediction** (Active Analysis) for the per-prediction SHAP breakdown
6. Check the **Truth Dashboard**:
   - **Live predictions (served)** — real predictions scored against actual closes
   - **Backtest (simulated)** — walk-forward simulation with CQR calibration

## Configuration

Environment variables (set in `.env` — never committed):

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | `dev` or `prod` (prod enforces hashed auth) | `dev` |
| `ANTHROPIC_API_KEY` | Claude API key for sentiment scoring | *(required)* |
| `AUTH_USERS` | Comma-separated `user:bcrypt_hash` pairs (`python -m scripts.hash_password`) | *(empty = all requests rejected)* |
| `DATA_DIR` | Base dir for DBs, logs, models (Docker: `/data`) | project root |
| `DATABASE_URL` | SQLite database URL | derived from `DATA_DIR` |
| `LOG_LEVEL` / `LOG_JSON` | Logging level / JSON-lines file sink | `INFO` / `true` |
| `CORS_ORIGINS` | Allowed CORS origins (comma-separated) | *(empty)* |
| `RATE_LIMIT_DEFAULT` / `RATE_LIMIT_HEAVY` | API rate limits (heavy = train/backtest) | `60/minute` / `3/minute` |
| `TRACKED_STOCKS` | Tickers the scheduler refreshes/predicts/retrains | 10 NIFTY large-caps |
| `RETRAIN_CRON` | Cron expression for model retraining | `0 16 * * 5` (Fri 4 PM) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (no auth) |
| `/predict/{ticker}` | GET | Next-day prediction (persisted, with SHAP explanation) |
| `/predictions/{ticker}/history` | GET | Served predictions with realized outcomes |
| `/predictions/summary/live` | GET | Live accuracy metrics (pinball, coverage, direction) |
| `/train/{ticker}` | POST | Train models (rate-limited) |
| `/backtest/{ticker}` | POST | Walk-forward backtest with CQR (rate-limited) |
| `/backtest/{ticker}/history` | GET | Backtest predictions |
| `/backtest/summary/all` | GET | Backtest summaries |
| `/models` | GET | List trained model bundles |
| `/macro` | GET | Current macro indicators |

All endpoints except `/health` require HTTP Basic Auth (bcrypt-verified). Every response
carries an `X-Request-ID` header that correlates with the structured JSON logs.

## Observability

- **Structured logs**: JSON lines in `logs/app.jsonl` (rotated, gzipped) with request IDs;
  every served prediction emits an `event=prediction` record with model version,
  features hash, quantiles, and safeguard flags
- **Prediction provenance**: the `predictions` table stores the full input vector,
  features hash, SHAP values, warnings, and model version for every served prediction
- **Outcome backfill**: a daily scheduler job (16:30 IST) fills in realized next-day
  changes, feeding the live accuracy metrics
- **Drift monitoring**: each model bundle stores its training feature distribution;
  inference flags out-of-distribution inputs

## Development

```bash
ruff check . && ruff format --check .          # lint + format
pytest tests/ -m "not slow" -q                 # offline deterministic tests
pytest tests/test_smoke_train.py -q            # end-to-end smoke train (offline)
pytest tests/ -q                               # everything incl. network tests
pre-commit install                             # from the repo root
```

Dependency locking: `pyproject.toml` is the source of truth;
`requirements.lock` (dev) and `requirements-runtime.lock` (Docker) are generated with
`uv pip compile pyproject.toml [--extra dev] --python-version 3.13 -o <file>`.

## CI/CD

GitHub Actions (`.github/workflows/ci.yml` at the repo root) runs on every push/PR:
- Ruff lint + format check
- Offline unit tests
- **Deterministic smoke train** on a committed synthetic dataset (trains the full
  pipeline, checks conformal calibration, proves seed reproducibility) + leakage
  regression tests
- Docker image build

Pre-commit hooks (root `.pre-commit-config.yaml`): ruff, ruff-format, **gitleaks**
secret scanning, large-file check, private-key detection, and a hard block on
committing any `.env` file.

## API Credit Usage

The app uses Anthropic API credits **only** for sentiment scoring (Claude Haiku) when a
prediction or scheduled snapshot runs with macro/sentiment enabled. Model training
(LightGBM) uses **no API credits** — it runs locally.

## Tech Stack

- **ML**: LightGBM (seeded/deterministic), scikit-learn, SHAP, CQR conformal calibration
- **Data**: pandas, pandas_ta (with numpy fallbacks), jugaad-data, yfinance, BeautifulSoup
- **Backend**: FastAPI, slowapi, SQLAlchemy, SQLite (WAL), APScheduler, bcrypt
- **Frontend**: Streamlit, Plotly
- **AI**: Anthropic Claude Haiku (sentiment scoring)
- **Config**: pydantic-settings (SecretStr), python-dotenv
- **CI**: GitHub Actions, ruff, pytest, pre-commit, gitleaks, Docker

## License

MIT
