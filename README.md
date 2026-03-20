# Quantitative Pair Analysis Dashboard

An interactive quantitative pair analysis dashboard built in Python for JupyterLab. Analyses two stocks side-by-side across correlation, lead/lag detection, rolling correlation, OLS expected return modelling, risk metrics, valuation, and spread z-score mean reversion signals — outputting a styled HTML recommendations panel with colour-coded trading signals.

Built as part of MSc Financial Technology research at Bristol Business School, UWE (2025–2026).

---

## Overview

Pairs trading and relative value analysis require understanding not just whether two stocks are correlated, but *how* that correlation behaves over time, whether one stock leads the other, and whether the spread between them is currently stretched or compressed. This dashboard brings together all of that analysis in a single interactive tool.

Four core analytical questions this tool answers:

1. **How correlated are these two stocks — and has that changed?** Rolling correlation shows how the relationship has evolved, not just what it is on average.
2. **Does one stock lead the other?** Cross-correlation at every lag from -N to +N bars surfaces lead/lag relationships that simple correlation misses entirely.
3. **Is the spread currently stretched?** A rolling z-score of the normalised price spread generates mean reversion signals at ±2 standard deviations.
4. **How much of one stock's return is explained by the other?** The OLS expected return model quantifies alpha, beta, R², and the current residual — the same framework used by active fund managers measuring performance versus a benchmark.

---

## Key Features

### Data & Configuration
- Downloads adjusted price data from Yahoo Finance for both stocks and the benchmark
- Configurable date range and timeframe: weekly (1wk), daily (1d), hourly (1h), 30m, 15m, 5m
- Log returns or percentage returns, user-selectable
- Configurable rolling window, max lag, and risk-free rate
- User designates one stock as "tracking" and one as "active" for the return model

### Risk Metrics
All metrics are interval-aware — annualisation factor adjusts automatically for weekly, daily, or intraday data:
- Annualised return estimate
- Annualised volatility
- Sharpe ratio
- Sortino ratio (downside deviation only)
- Maximum drawdown
- Beta versus the selected benchmark

### Valuation
- Trailing P/E and forward P/E for both stocks via yfinance
- Relative valuation comparison with discount detection

### Correlation Analysis
- Overall Pearson correlation between the two stocks
- Rolling correlation over a configurable window — visualises regime changes in the relationship
- Reference lines at ±0.8 to highlight high-correlation regimes

### Lead/Lag Detection
- Cross-correlation calculated at every integer lag from -N to +N bars
- Peak absolute cross-correlation identifies the lag at which the relationship is strongest
- Positive lag = Stock A tends to lead Stock B; negative lag = Stock B leads
- Bar chart with highlighted peak lag and directional interpretation

### Spread Z-Score & Mean Reversion Signals
- Normalised price spread calculated from rebased prices
- Rolling z-score over the configurable window
- Trading signals generated at ±2 standard deviations:
  - z > +2: spread stretched high → Short A / Long B signal
  - z < −2: spread stretched low → Long A / Short B signal
  - |z| > 1: elevated, monitor
  - |z| ≤ 1: near mean, no signal

### OLS Expected Return Model
- Fits: Active(t) = α + β × Tracking(t) + ε via ordinary least squares
- Outputs: alpha, beta, R², expected return series, and residual series
- Alpha interpretation: positive = active stock generating excess return above tracking
- Latest residual signal: >1.5σ above = active outperforming model (watch for reversion); <1.5σ below = underperforming
- Directly analogous to how active fund managers measure performance versus a benchmark

### Beta-Neutral Sizing
- Beta of each stock versus the benchmark calculated via OLS
- When betas differ significantly, hedge ratio calculated for a market-neutral pair trade

### Styled HTML Recommendations Panel
Eight sections with colour-coded badges and signals:
1. **Trend & Momentum** — bullish/bearish/neutral with return, vol, and Sharpe per stock
2. **Relative Strength** — identifies momentum leader for long/short tilt
3. **Correlation & Pair Trading** — correlation regime classification + spread z-score signal
4. **Lead/Lag** — directional signal with bar count and cross-correlation strength
5. **Expected Return Model** — model quality, alpha, beta, and latest residual signal
6. **Market Sensitivity** — beta classification and beta-neutral hedge ratio
7. **Valuation** — trailing/forward P/E comparison with relative discount detection
8. **Summary** — consolidated signal table across all dimensions

### Eight Interactive Subplots
All charts use a consistent dark theme (IBM Plex Mono, dark background, annotated reference lines):
1. Normalised prices (rebased to 100)
2. Returns series
3. Rolling correlation with ±0.8 reference lines
4. Cross-correlation bar chart with peak lag annotation
5. Lagged scatter plot (A vs shifted B)
6. Drawdown with −20% reference line
7. Actual vs expected active return (OLS model overlay)
8. Model residuals with zero line

---

## Dashboard Structure

| Control | Description |
|---|---|
| Stock A / Stock B | Any Yahoo Finance ticker |
| Benchmark | Benchmark for beta calculation (default: SPY) |
| Start / End date | Date range for historical data |
| Timeframe slider | 1wk → 1d → 1h → 30m → 15m → 5m |
| Returns type | Log returns or percentage returns |
| Tracking / Active | Designate which stock is tracking vs active for OLS model |
| Max lag | Maximum lag for cross-correlation (bars) |
| Rolling window | Window for rolling correlation and spread z-score |
| Rf (ann) | Annual risk-free rate for Sharpe/Sortino calculation |

---

## Screenshots

*(Add screenshots here)*

---

## Technical Stack

| Component | Library |
|---|---|
| Data download | yfinance |
| Numerical computation | NumPy |
| Data manipulation | Pandas |
| Visualisation | Plotly |
| Interactive UI | ipywidgets |
| OLS regression | NumPy (lstsq) |
| Styled output | IPython HTML |

---

## Installation & Usage

### Requirements
```
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.36
plotly>=5.18.0
ipywidgets>=8.0.0
jupyterlab>=4.0.0
```

Install all dependencies:
```bash
pip install numpy pandas yfinance plotly ipywidgets jupyterlab
```

### Running the Dashboard
1. Clone or download this repository
2. Open `pair_analysis_dashboard.ipynb` in JupyterLab
3. Run the single cell (Shift+Enter)
4. Enter Stock A, Stock B, and Benchmark tickers
5. Set date range and timeframe
6. Click **"▶ Calculate"**

> **Note:** Intraday intervals (5m, 15m, 30m, 1h) are subject to Yahoo Finance data limits. For intraday analysis use a shorter date range (30–60 days). Weekly and daily intervals work for any date range.

---

## Methodology Notes

### Why Cross-Correlation for Lead/Lag?
Simple correlation tells you whether two stocks move together but not *when*. Cross-correlation at multiple lags reveals whether returns at time t in one stock are correlated with returns at time t+k in the other — a lead/lag relationship that can be exploited in a pairs trading context by using the leading stock's moves as an early entry signal.

### Why Spread Z-Score?
Raw price spreads are non-stationary and scale-dependent. Normalising the spread by its rolling standard deviation produces a dimensionless z-score that is directly comparable across different pairs and time periods. Extreme z-scores (±2σ) historically correspond to mean reversion opportunities in cointegrated pairs.

### Why OLS Expected Return Model?
The active ~ alpha + beta × tracking framework is the same structure used in the CAPM and in active portfolio management to decompose returns into systematic (beta) and idiosyncratic (alpha) components. A positive alpha indicates the active stock is generating returns above what its relationship to the tracking stock would predict — potentially indicating relative outperformance or a mispricing.

### Limitations
- **Data source:** yfinance provides adjusted prices suitable for research. Licensed data required for production use.
- **Cointegration not tested:** spread z-score assumes approximate stationarity. For rigorous pairs trading, cointegration testing (ADF, Johansen) should be applied first.
- **OLS assumptions:** the expected return model assumes a linear, stationary relationship. In practice, the alpha and beta of a pair may be time-varying.
- **Lead/lag stability:** cross-correlation identifies historical lead/lag patterns which may not persist out of sample.

---

## Academic Context

This tool was developed as part of MSc Financial Technology research at Bristol Business School, University of the West of England (2025–2026), alongside companion repositories:
- [`portfolio-optimisation-dashboard`](https://github.com/ThomasOxley/portfolio-optimisation-dashboard) — multi-asset portfolio construction and risk analytics
- [`stock-screening-forecasting-dashboard`](https://github.com/ThomasOxley/stock-screening-forecasting-dashboard) — S&P 500 screening and price forecasting

---

## Disclaimer

This tool is an academic research project. All outputs are for educational and illustrative purposes only. Nothing in this repository constitutes financial advice or a recommendation to buy or sell any security. All signals are derived from historical data — past performance does not guarantee future results.

---

*Thomas Oxley | MSc Financial Technology | Bristol Business School, UWE*
*linkedin.com/in/thomas-oxley-868047174 | github.com/ThomasOxley*
```

---
