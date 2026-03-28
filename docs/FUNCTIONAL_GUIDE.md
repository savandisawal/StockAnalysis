# Stock Quant App — Functional Guide

A beginner-friendly guide for the team on how to use the app, how predictions work, and what each section means.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Home Page — The Main Dashboard](#2-home-page--the-main-dashboard)
3. [How Prediction Works](#3-how-prediction-works)
4. [Understanding the Prediction Explainer](#4-understanding-the-prediction-explainer)
5. [Active Analysis Page](#5-active-analysis-page)
6. [Global Pulse Page](#6-global-pulse-page)
7. [Truth Dashboard Page](#7-truth-dashboard-page)
8. [Key Concepts Explained](#8-key-concepts-explained)
9. [FAQ](#9-faq)

---

## 1. Getting Started

### Starting the App

```bash
cd StockAnalysis/stock_quant_app
streamlit run ui_streamlit/Home.py
```

Open **http://localhost:8501** in your browser.

### First-Time Setup

1. Make sure `.env` file exists with your `ANTHROPIC_API_KEY` (ask the team lead)
2. Start the app
3. Select any stock (e.g., RELIANCE)
4. Click **Train Model** — this trains the ML model for that stock
5. Once trained, you'll see predictions automatically

---

## 2. Home Page — The Main Dashboard

The Home page is the central hub. Here's what each section does:

### Sidebar (Left Panel)

| Control | What It Does |
|---------|-------------|
| **Filter by Sector** | Narrows the stock dropdown to a specific sector (IT, Banking, Pharma, etc.) |
| **Stock Dropdown** | Select a stock. Shows "TICKER — Company Name" format. Click and start typing to search. |
| **Custom Ticker** | Type any NSE ticker not in the dropdown (e.g., ZOMATO, PAYTM) |
| **Training Years** | How many years of historical data to train on (1-5 years). More years = more data but slower training. 2 years is a good default. |
| **Include Fundamentals** | Adds company financials (PE ratio, ROE, etc.) to the model. Makes training slower but can improve accuracy. |
| **Include Macro/Sentiment** | Adds market mood + news sentiment. Uses Anthropic API credits (costs money). |
| **Train Model** | Trains fresh ML models for the selected stock. Takes 30-60 seconds. |

### Price Metrics (Top Row)

Shows the **latest trading day** data:
- **Close** — Last closing price with daily % change
- **High** — Day's highest price
- **Low** — Day's lowest price
- **Volume** — Number of shares traded

### Next-Day Prediction

Appears after you train a model. Shows three price levels:

| Level | What It Means |
|-------|--------------|
| **P10 (Low)** | The model thinks there's only a 10% chance the price goes below this. This is the "worst case" estimate. |
| **P50 (Mid)** | The model's best estimate — the median prediction. This is the most likely price. |
| **P90 (High)** | The model thinks there's only a 10% chance the price goes above this. This is the "best case" estimate. |

**Direction**: Bullish (model expects price to go up), Bearish (expects down), or Neutral (sideways).

**Confidence**: 0-95% scale. Higher = model is more certain. Above 50% is decent. Below 30% means the model is very unsure.

### Prediction Explainer

Breaks down **why** the model made its prediction. Three expandable sections:

1. **Signal Summary** — Table of all technical signals with Bull/Bear/Neutral labels
2. **Feature Importance** — Bar chart showing which factors the model weighted most
3. **Confidence Breakdown** — Explains the confidence number and what it means

### Corporate Actions & Filings

Shows recent NSE corporate announcements for the stock:
- Board meetings, dividends, new orders, credit ratings, insider trading, etc.
- **Material** column = announcements that could move the stock price
- This data is also fed into the sentiment scoring AI

### Price Chart

Interactive candlestick chart (green = price went up, red = price went down). If a prediction exists, it shows predicted range as a shaded area on the right.

### Search History

A table tracking every stock you've looked at. Persists across page reloads. Shows close price, change %, prediction, direction, and confidence. Click **Clear History** to reset.

---

## 3. How Prediction Works

### The Simple Version

The app looks at **three categories of information** (called "pillars") and uses machine learning to predict tomorrow's price range:

```
Historical Price Patterns  ──┐
                              ├──▶  ML Model  ──▶  Tomorrow's Price Range
Company Financials  ──────────┤                    (Low / Mid / High)
                              │
Market Mood + News  ──────────┘
```

### The Detailed Version

#### Step 1: Data Collection

For each stock, the app collects:

**Pillar 1 — Technical Indicators** (from price history)
| Indicator | What It Measures | Example |
|-----------|-----------------|---------|
| RSI (14) | Is the stock overbought or oversold? | RSI > 70 = overbought, < 30 = oversold |
| MACD | Is momentum increasing or decreasing? | Positive histogram = bullish momentum |
| Bollinger %B | Where is the price within its normal range? | Near 0 = at bottom of range, near 1 = at top |
| EMA 20/50/200 | Is price above or below moving averages? | Above all 3 = strong uptrend |
| ADX | How strong is the current trend? | > 25 = strong trend, < 20 = sideways |
| Volume Z-Score | Is today's volume unusual? | > 2 = abnormally high volume |
| ATR % | How volatile is the stock? | Higher ATR = wider daily price swings |
| Market Regime | Is the stock trending or moving sideways? | Trending Up / Trending Down / Sideways |

**Pillar 2 — Fundamentals** (from Screener.in)
| Metric | What It Means |
|--------|--------------|
| PE Z-Score | Is the stock expensive compared to its sector peers? Positive = expensive, negative = cheap. |
| ROE Percentile | How profitable is the company vs peers? Higher = better. |
| D/E Percentile | How much debt does the company have vs peers? Lower = safer. |
| EPS CAGR 3Y | How fast are earnings growing over 3 years? |
| Promoter Holding | How much do founders/promoters own? Higher = more skin in the game. |

**Pillar 3 — Macro & Sentiment** (from multiple sources)
| Signal | Source | Why It Matters |
|--------|--------|---------------|
| S&P 500 / Nasdaq change | Yahoo Finance | US markets going up = positive for India |
| Nifty 50 change | Yahoo Finance | Direct Indian market benchmark |
| Brent Crude change | Yahoo Finance | Rising oil = bad for India (we import oil) |
| USD/INR change | Yahoo Finance | Rupee weakening = bad for Indian stocks |
| India VIX | Yahoo Finance | High VIX = market fear, expect volatility |
| News Sentiment | Google News + Claude AI | AI reads 10 headlines and scores -1 (bearish) to +1 (bullish) |
| Corporate Filings | NSE India | Board meetings, dividends, new orders — direct company events |

#### Step 2: Model Training

When you click **Train Model**, the app:

1. Downloads 2 years (configurable) of daily price data
2. Computes all the indicators above for every historical trading day
3. For each day, records what **actually happened the next day** (this is the "target")
4. Trains **three separate LightGBM models**:
   - **P10 model** — learns to predict the 10th percentile outcome (lower bound)
   - **P50 model** — learns to predict the median outcome (best estimate)
   - **P90 model** — learns to predict the 90th percentile outcome (upper bound)
5. Validates on the most recent 20% of data
6. Saves the trained models to disk

**Why three models?** A single "best guess" is often wrong. By predicting a **range**, we know how uncertain the model is. A narrow range = high confidence. A wide range = the model isn't sure.

#### Step 3: Making a Prediction

When you view a stock with a trained model:

1. App computes today's indicators (same features as training)
2. Feeds them into all three models
3. Converts the predicted % changes into actual price levels
4. Calculates confidence from how narrow the range is vs typical volatility (ATR)
5. Classifies direction as Bullish/Bearish/Neutral based on the P50 prediction

#### What is LightGBM?

LightGBM is a "gradient boosted decision tree" algorithm. Think of it like this:

- Imagine 500 simple yes/no decision trees (like a flowchart)
- Each tree learns from the mistakes of the previous trees
- Together, they form a powerful prediction engine
- It's fast, handles missing data well, and works great with tabular data (like financial features)

#### What is Quantile Regression?

Normal regression predicts one number (the average). **Quantile regression** predicts specific percentiles:

```
                    ┌─ P90: "90% chance price stays below this"
Prediction Range ───┤─ P50: "Best estimate (median)"
                    └─ P10: "90% chance price stays above this"
```

This gives us a calibrated prediction interval, not just a point estimate.

---

## 4. Understanding the Prediction Explainer

### Signal Summary Table

Each row shows one technical indicator:

| Column | What It Shows |
|--------|--------------|
| **Signal** | Name of the indicator |
| **Value** | Current numerical value |
| **Interpretation** | What the value means in plain English |
| **Sentiment** | Bullish (green) / Bearish (red) / Neutral (yellow) |

The **summary at the bottom** counts how many signals are bullish vs bearish. If most signals are bullish and the model predicts bullish, that's a strong consensus. If signals disagree, be cautious.

### Feature Importance Chart

Shows which features the P50 model weighted most heavily when making predictions. This is based on how often each feature was used in the decision trees and how much it reduced prediction error.

**How to read it:**
- Features at the top are the most important
- If `rsi_14` is at the top, RSI was the biggest driver of predictions
- If `volume_zscore` is near the bottom, volume wasn't very useful for this stock

This changes per stock — RSI might be important for RELIANCE but not for TCS.

### Confidence Breakdown

- **Confidence %**: Derived from prediction range width vs ATR (average daily volatility)
  - If the predicted range (P90 - P10) is **narrower** than typical daily movement → high confidence
  - If the predicted range is **wider** than typical daily movement → low confidence
- **Range Width**: The spread between P10 and P90 in percentage terms
- **Direction**: Based on whether P50 predicts positive or negative change

---

## 5. Active Analysis Page

Deep technical analysis for the selected stock. Useful for understanding current market conditions.

### Sections

| Section | What It Shows |
|---------|--------------|
| **Prediction** | Same prediction card as Home page |
| **Candlestick Chart** | OHLC chart with adjustable period (30D to 1Y) |
| **RSI Tab** | RSI chart with overbought (>70) and oversold (<30) zones |
| **MACD Tab** | MACD histogram — positive = bullish momentum |
| **Bollinger %B Tab** | Position within Bollinger Bands |
| **ADX Tab** | Trend strength + market regime (Trending Up/Down/Sideways) |
| **Volume Z Tab** | Volume relative to 20-day average. Above 2 = unusual |
| **Key Metrics** | Quick view of ATR, EMA distances, BB width, 1D return |
| **Macro Mood** | Expandable section with global indicators and aggregate mood score |

### How to Use It

1. Check the **Regime** (ADX tab) — is the stock trending or sideways?
2. Check **RSI** — is it overbought (potential pullback) or oversold (potential bounce)?
3. Check **Volume** — unusual volume with a breakout = stronger signal
4. Check **EMA positions** — price above EMA 20/50/200 = bullish structure

---

## 6. Global Pulse Page

Shows real-time global macro indicators that affect Indian markets.

### Market Mood Banner

An aggregate score from -1 (very bearish) to +1 (very bullish), combining all indicators:
- **BULLISH** (green): Score > +0.3 — positive global signals
- **NEUTRAL** (yellow): Score between -0.3 and +0.3 — mixed
- **BEARISH** (red): Score < -0.3 — negative global signals

### Indicators

| Indicator | Bullish When | Bearish When |
|-----------|-------------|-------------|
| S&P 500 | Rising (global risk-on) | Falling (global risk-off) |
| Nasdaq | Rising (tech sentiment positive) | Falling |
| Nifty 50 | Rising (Indian market strength) | Falling |
| Brent Crude | Falling (lower input costs for India) | Rising (India imports 85% of oil) |
| USD/INR | Falling / rupee strengthening | Rising / rupee weakening (FII outflows) |
| India VIX | Below 15 (calm) | Above 25 (fear / expect big swings) |

### How to Use It

- Check this page **before market open** to gauge global sentiment
- If most indicators are red, be cautious with bullish predictions
- VIX above 25 = expect wider intraday ranges regardless of direction

---

## 7. Truth Dashboard Page

This is the **honesty page** — it shows how accurate the model actually is.

### Running a Backtest

Click **Run Walk-Forward Backtest** to simulate how the model would have performed historically:

1. It takes historical data and splits it into chunks
2. Trains on earlier data, predicts the next day
3. Moves forward, retrains, predicts again
4. Compares every prediction against what actually happened

This takes 1-3 minutes depending on the stock and years selected.

### Performance Metrics

| Metric | What It Means | Good Value |
|--------|--------------|------------|
| **MAE** | Average prediction error in % | Lower is better. Below 1.5% is good. |
| **RMSE** | Like MAE but penalizes big errors more | Lower is better. |
| **Direction Accuracy** | % of times model correctly predicted up/down | Above 55% is good. 50% = coin flip. |
| **80% Coverage** | % of times actual price fell within P10-P90 range | Should be around 80%. Below 60% = model is overconfident. |
| **Model Grade** | A/B/C/D based on direction accuracy + coverage | A = reliable, D = needs improvement. |

### Charts

- **Predicted vs Actual**: Scatter plot comparing P50 predictions against actual outcomes. Points on the diagonal = perfect prediction.
- **Error Trend**: Shows if errors are getting larger or smaller over time. Upward trend = model degrading, needs retraining.
- **Raw Data**: Full table of every prediction with the error for each.

### How to Use It

1. Always backtest a stock **before trusting predictions**
2. If direction accuracy < 50%, the model is worse than a coin flip — don't use it
3. If coverage < 60%, the prediction range is too narrow — model is overconfident
4. Re-train periodically (weekly) as new data comes in

---

## 8. Key Concepts Explained

### What is RSI?

**Relative Strength Index** — measures if a stock has been going up too much (overbought) or down too much (oversold) recently.

- Scale: 0 to 100
- Above 70 = overbought (might pull back)
- Below 30 = oversold (might bounce)
- Between 30-70 = normal

### What is MACD?

**Moving Average Convergence Divergence** — measures momentum.

- When the MACD histogram is positive and growing = bullish momentum increasing
- When negative and shrinking = bearish momentum increasing
- Crossover from negative to positive = potential buy signal

### What is Bollinger %B?

Bollinger Bands create an envelope around the price based on volatility.

- %B = 0 means price is at the lower band (potential support)
- %B = 1 means price is at the upper band (potential resistance)
- %B > 1 means price broke above the upper band (strong breakout or overbought)

### What is EMA?

**Exponential Moving Average** — a smoothed average of recent prices.

- EMA 20 = short-term trend (last ~1 month)
- EMA 50 = medium-term trend (last ~2.5 months)
- EMA 200 = long-term trend (last ~10 months)
- Price above all three = strong uptrend
- "Golden cross" = EMA 50 crosses above EMA 200 (bullish)
- "Death cross" = EMA 50 crosses below EMA 200 (bearish)

### What is ADX?

**Average Directional Index** — measures trend strength (not direction).

- Below 20 = sideways / no trend (range-bound trading)
- 20-25 = emerging trend
- Above 25 = strong trend
- Above 50 = very strong trend

### What is VIX?

**Volatility Index** — measures expected market volatility over the next 30 days.

- Below 15 = calm, low fear
- 15-20 = normal
- 20-25 = elevated concern
- Above 25 = high fear, expect big daily swings
- Often called the "fear gauge"

### What is a Z-Score?

A Z-score tells you how far a value is from its average, measured in standard deviations.

- Z = 0 means the value is exactly at the average
- Z = +2 means the value is 2 standard deviations above average (unusually high)
- Z = -2 means 2 standard deviations below average (unusually low)

Used for volume (is today's volume unusual?) and PE ratio (is this stock expensive vs peers?).

---

## 9. FAQ

### How often should I retrain models?

**Weekly** is recommended. The app has a scheduler that auto-retrains every Friday at 4 PM IST when the FastAPI backend is running. For manual retraining, just click "Train Model" on the Home page.

### Does training cost money?

- **Basic training** (technical features only) = **FREE**, runs locally
- **With Fundamentals** = **FREE**, scrapes Screener.in
- **With Macro/Sentiment** = **Uses API credits** (Claude Haiku is very cheap — roughly $0.001-0.005 per stock)

### Can I predict any stock?

Yes. Type any valid NSE ticker in the custom ticker input. The app has 150+ pre-mapped stocks with company names and sector info, but any NSE-listed stock works.

### How accurate are predictions?

Varies by stock. Run a backtest on the Truth Dashboard to find out. Typical results:
- Direction accuracy: 52-60% (anything above 55% is useful)
- 80% interval coverage: 65-85%

Stock markets are inherently unpredictable. This tool provides **probabilistic estimates**, not guarantees.

### What does "Neutral" direction mean?

The model predicts less than 0.3% change in either direction — essentially flat. Don't read too much into small predicted changes.

### Why is confidence low?

Low confidence means the P10-P90 range is wide relative to the stock's typical daily volatility. This happens when:
- The stock is highly volatile (metals, small caps)
- Technical signals are conflicting
- The model hasn't seen similar patterns before

### What if prediction and signals disagree?

If the model says "Bullish" but most signals say "Bearish", be cautious. The model weighs all features together (some you can't see in the signal table), but disagreement = uncertainty. Check the Truth Dashboard to see if the model has been accurate for this stock.

### How is news sentiment scored?

1. App fetches 10 recent headlines from Google News RSS
2. App fetches corporate filings from NSE (dividends, board meetings, new orders, etc.)
3. Both are sent to **Claude Haiku** (Anthropic's fast AI model)
4. Claude reads them and returns a score from -1 (very bearish) to +1 (very bullish) with a one-sentence reason
5. This score becomes one feature the ML model uses

### What's the difference between Home and Active Analysis?

- **Home** = quick overview with prediction, explainer, and history
- **Active Analysis** = deep dive into each technical indicator with interactive charts

Use Home for a quick check, Active Analysis when you want to understand the technicals in detail.
