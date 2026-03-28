# Stock Quant App

Production-grade **next-day price range prediction system** for NSE India stocks, powered by LightGBM quantile regression and a triple-pillar feature engine.

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![LightGBM](https://img.shields.io/badge/ML-LightGBM-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FastAPI](https://img.shields.io/badge/API-FastAPI-teal)

## What It Does

For any NSE-listed stock, the app:

1. **Predicts tomorrow's price range** (P10 / P50 / P90) using quantile regression
2. **Scores market sentiment** via Claude AI analysis of news + corporate filings
3. **Explains predictions** with a signal summary, feature importance, and confidence breakdown
4. **Tracks accuracy** with walk-forward backtesting and an honesty dashboard

## Architecture

```
stock_quant_app/
|-- app/              # FastAPI backend (auth, config, database)
|-- data/             # Data fetchers (NSE OHLCV, fundamentals, news, macro, corporate filings)
|-- features/         # Triple-pillar feature engine
|-- model/            # LightGBM training, prediction, backtesting, model registry
|-- services/         # Prediction service, APScheduler (daily refresh + weekly retrain)
|-- ui_streamlit/     # Streamlit multi-page frontend
|   |-- Home.py           # Main page: prediction, safeguards, explainer, fundamentals, corporate actions, history
|   |-- pages/
|       |-- 1_Active_Analysis.py   # Deep technical analysis with indicator charts
|       |-- 2_Global_Pulse.py      # Live macro indicators dashboard
|       |-- 3_Truth_Dashboard.py   # Backtest accuracy & honesty metrics
|-- utils/            # Logging, NSE holidays, sector mappings
|-- tests/            # Unit & integration tests
```

## Triple-Pillar Feature Engine

| Pillar | Features | Source |
|--------|----------|--------|
| **Technical** | RSI, MACD, Bollinger %B, EMA 20/50/200, ADX, Volume Z-Score, ATR, Market Regime | pandas_ta on OHLCV data |
| **Fundamental** | PE Z-Score, ROE/D-E Percentile, EPS CAGR, Promoter Holding | Screener.in (sector-relative) |
| **Macro & Sentiment** | S&P 500, Nasdaq, Nifty, Brent, USD/INR, VIX changes + News Sentiment + Corporate Filings | Yahoo Finance, Google News RSS, NSE API, Claude Haiku |

## ML Model

- **LightGBM Quantile Regression** with three models per stock:
  - **P10** (10th percentile) — predicted low
  - **P50** (median) — best estimate
  - **P90** (90th percentile) — predicted high
- Walk-forward expanding window training with 80/20 time-series split
- Confidence score derived from prediction interval width relative to ATR
- Direction classification: Bullish (>+0.3%), Bearish (<-0.3%), Neutral

## Prediction Safeguards

Six layers of protection against unreliable predictions:

| Safeguard | Trigger | Effect |
|-----------|---------|--------|
| **Guardrails** | Predicted change > 2x ATR | Caps prediction to realistic range |
| **Feature Drift** | 3+ features outside normal bounds | Warns model may not generalize |
| **Model Staleness** | Model trained > 7 days ago | Warns to retrain |
| **Circuit Breakers** | VIX change > 30% or Volume Z > 4 | Flags prediction as unreliable |
| **Calibration** | Backtest accuracy < 50% or coverage < 60% | Adjusts confidence downward |
| **Ensemble Sanity** | P10/P50/P90 disordered or spread > 8% | Reduces confidence by 30% |

## Data Sources

| Data | Source | Method |
|------|--------|--------|
| NSE OHLCV (primary) | [jugaad-data](https://github.com/jugaad-py/jugaad-data) | Direct NSE scraping |
| NSE OHLCV (fallback) | yfinance | Yahoo Finance API |
| Fundamentals | Screener.in | HTML scraping |
| News Headlines | Google News RSS | RSS feed parsing |
| Corporate Filings | NSE India API | `/api/corporate-announcements` |
| Macro Indicators | Yahoo Finance | Global indices, commodities, forex |
| Sentiment Scoring | Anthropic Claude Haiku | LLM-based headline + filing analysis |

## Quick Start

### Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/) (for sentiment scoring)

### Setup

```bash
# Clone the repository
git clone https://github.com/savandisawal/StockAnalysis.git
cd StockAnalysis/stock_quant_app

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Run

```bash
# Start the Streamlit UI
streamlit run ui_streamlit/Home.py

# Or start the FastAPI backend
uvicorn app.main:app --reload --port 8000
```

### Usage

1. Open http://localhost:8501
2. Select a stock from the sidebar (filter by sector or type any NSE ticker)
3. Click **Train Model** to train the quantile regression models
4. View the next-day prediction with confidence score and safeguard warnings
5. Explore the **Prediction Explainer** for signal breakdown and feature importance
6. Check **Fundamentals Overview** for PE, ROE, D/E, EPS growth, promoter holding vs sector peers
7. Check **Corporate Actions & Filings** for recent NSE announcements
8. Navigate to other pages:
   - **Active Analysis** — technical indicators with interactive charts
   - **Global Pulse** — macro indicators and market mood
   - **Truth Dashboard** — backtest accuracy and model honesty metrics

## Configuration

Environment variables (set in `.env`):

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key for sentiment scoring | *(required)* |
| `AUTH_USERS` | Comma-separated `user:pass` pairs for API auth | `admin:changeme` |
| `DATABASE_URL` | SQLite database path | `sqlite+aiosqlite:///./stock_quant.db` |
| `NEWS_API_KEY` | Optional News API key (falls back to RSS) | *(empty)* |
| `LOG_LEVEL` | Logging level | `INFO` |
| `RETRAIN_CRON` | Cron expression for model retraining | `0 16 * * 5` (Fri 4 PM) |

## API Endpoints

The FastAPI backend provides:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/predict/{ticker}` | GET | Get next-day prediction |
| `/api/train/{ticker}` | POST | Train models for a stock |
| `/api/backtest/{ticker}` | POST | Run walk-forward backtest |
| `/api/models/{ticker}` | GET | List trained models |
| `/api/macro` | GET | Current macro indicators |

All `/api/*` endpoints require HTTP Basic Auth.

## Stock Coverage

150+ NSE stocks across 15 sectors pre-mapped with company names:

IT, Banking, NBFC, Pharma, Auto, Energy, Metals, FMCG, Cement, Telecom, Real Estate, Insurance, Capital Goods, Chemicals

Any NSE ticker can also be entered manually via the custom ticker input.

## Development

```bash
# Run linter
ruff check . --exclude .venv

# Run tests (skip slow API-dependent tests)
pytest tests/ -v -m "not slow"

# Run all tests
pytest tests/ -v
```

## CI/CD

GitHub Actions runs on every push/PR to `main`/`master`:
- Ruff linting
- Unit tests (excluding slow/external API tests)

## API Credit Usage

The app uses Anthropic API credits **only** for sentiment scoring (Claude Haiku). This happens when:
- Training a model with **"Include Macro/Sentiment"** checked
- Running a prediction with macro/sentiment features enabled

Training the model itself (LightGBM) uses **no API credits** — it runs locally.

## Tech Stack

- **ML**: LightGBM, scikit-learn, SHAP
- **Data**: pandas, pandas_ta, jugaad-data, yfinance, BeautifulSoup
- **Backend**: FastAPI, SQLAlchemy, SQLite (WAL mode), APScheduler
- **Frontend**: Streamlit, Plotly
- **AI**: Anthropic Claude Haiku (sentiment scoring)
- **Config**: pydantic-settings, python-dotenv
- **CI**: GitHub Actions, ruff, pytest

## License

MIT
