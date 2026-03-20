#!/usr/bin/env python
# coding: utf-8

# In[7]:


# ================================================================================
# Quantitative Pair Analysis Dashboard
# ================================================================================
# Author:      Thomas Oxley
# Institution: Bristol Business School, UWE — MSc Financial Technology
# Date:        2024–2025
# Version:     2.0

# Description:
#    An interactive quantitative pair analysis dashboard built for JupyterLab.
#    Analyses two stocks side-by-side across correlation, lead/lag detection,
#    rolling correlation, OLS expected return modelling, risk metrics, valuation,
#    and spread z-score mean reversion signals. Outputs a styled HTML
#    recommendations panel with colour-coded trading signals.

# Key Features:
#    - Correlation analysis: overall and rolling correlation over configurable window
#    - Lead/lag detection: cross-correlation at every lag from -N to +N bars,
#      identifies which stock tends to move first
#    - OLS expected return model: active ~ alpha + beta * tracking, with R²,
#      residual series, and latest residual signal
#    - Spread z-score: rolling normalised spread with mean reversion signals
#     (long/short signals at ±2 standard deviations)
#    - Risk metrics: annualised return, volatility, Sharpe, Sortino, max drawdown,
#      beta vs benchmark — all interval-aware
#    - Valuation: trailing and forward P/E ratios for both stocks
#    - Beta-neutral sizing: calculates hedge ratio for market-neutral pair trade
#    - Eight interactive subplots: normalised prices, returns, rolling correlation,
#      cross-correlation bar chart, lagged scatter, drawdowns, expected vs actual
#    - Styled HTML recommendations panel: trend, relative strength, correlation
#      active return, model residuals
#      regime, lead/lag signal, expected return model, beta sensitivity,
#    - Configurable timeframes: weekly down to 5-minute intraday bars
#      valuation comparison, and summary table
#    - Dark theme UI with IBM Plex Mono typography throughout

# Dependencies:
#    See requirements.txt

# Usage:
#    Run the single cell in JupyterLab. Configure Stock A, Stock B, benchmark,
#    date range, and timeframe, then click Calculate.

# Note:
#    This is an academic research tool. Outputs are for educational and
#    illustrative purposes and do not constitute financial advice.
# ================================================================================

import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


# ── Palette & style constants ──────────────────────────────────────────────────
_CLR_BG        = "#0f1117"
_CLR_PANEL     = "#1a1d27"
_CLR_BORDER    = "#2e3248"
_CLR_ACCENT    = "#5b8dee"   # blue
_CLR_GREEN     = "#3ecf8e"
_CLR_RED       = "#f66d6d"
_CLR_YELLOW    = "#f5c842"
_CLR_TEXT      = "#e2e6f0"
_CLR_MUTED     = "#7b82a0"

_PLOTLY_THEME = dict(
    paper_bgcolor=_CLR_BG,
    plot_bgcolor=_CLR_PANEL,
    font=dict(family="'IBM Plex Mono', 'Courier New', monospace", size=11, color=_CLR_TEXT),
    title_font=dict(family="'IBM Plex Mono', 'Courier New', monospace", size=13, color=_CLR_TEXT),
    legend=dict(
        bgcolor=_CLR_PANEL,
        bordercolor=_CLR_BORDER,
        borderwidth=1,
        font=dict(size=10),
    ),
    hoverlabel=dict(
        bgcolor=_CLR_PANEL,
        bordercolor=_CLR_BORDER,
        font=dict(family="'IBM Plex Mono', monospace", size=11),
    ),
)

def _rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour + alpha float to an rgba() string Plotly accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

_AXIS_STYLE = dict(
    gridcolor=_CLR_BORDER,
    zerolinecolor=_CLR_BORDER,
    tickfont=dict(size=10, color=_CLR_MUTED),
    linecolor=_CLR_BORDER,
)

# Chart explanation annotations (shown as subtitles beneath each subplot title)
_CHART_EXPLANATIONS = {
    "price":    "Rebased to 100 at start. Divergence = one outperforming.",
    "returns":  "Period log/pct returns. Clustering = volatility regimes.",
    "rollcorr": "How correlated they've been over time. Near ±1 = tight coupling.",
    "xcorr":    "Cross-correlation at each lag. Peak away from 0 = lead/lag exists.",
    "scatter":  "Returns of A vs lagged B. Slope = predictive relationship.",
    "drawdown": "% fall from each stock's prior peak. Deeper = higher risk.",
    "expected": "OLS model: active ≈ α + β·tracking. Blue=actual, red=model.",
    "residual": "Actual minus model. Spikes = unexplained moves / alpha opportunities.",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def _to_date(x):
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return x.date()
    if hasattr(x, "year") and hasattr(x, "month") and hasattr(x, "day") and not isinstance(x, str):
        try:
            return x
        except Exception:
            pass
    return pd.to_datetime(x).date()

def _download_prices(ticker: str, start_date, end_date, interval: str) -> pd.Series:
    ticker = ticker.strip().upper()
    start_date = _to_date(start_date)
    end_date   = _to_date(end_date)
    if start_date is None or end_date is None:
        raise ValueError("Start/End date missing.")
    end_plus = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).date()

    df = yf.download(ticker, start=start_date, end=end_plus,
                     interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker} @ {interval}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    s = df[col].squeeze().dropna().rename(ticker)
    if s.empty:
        raise ValueError(f"No usable price column for {ticker} @ {interval}.")
    return s

def _returns(prices: pd.Series, kind="log") -> pd.Series:
    r = np.log(prices).diff() if kind == "log" else prices.pct_change()
    return r.dropna().rename(prices.name)

def _annualization_factor(interval: str) -> float:
    if interval.endswith("m"):
        return float(252 * 390 / int(interval[:-1]))
    if interval.endswith("h"):
        return float(252 * 6.5 / int(interval[:-1]))
    if interval == "1wk":
        return 52.0
    return 252.0

def _risk_metrics(r: pd.Series, ann_factor: float, rf_annual: float) -> dict:
    rf_bar   = rf_annual / ann_factor
    ex       = r - rf_bar
    ann_vol  = float(r.std(ddof=1) * np.sqrt(ann_factor))
    ann_ret  = float(r.mean() * ann_factor)
    sharpe   = float(ex.mean() / (r.std(ddof=1) + 1e-12) * np.sqrt(ann_factor))
    downside = r[r < 0]
    sortino  = float(ex.mean() / (downside.std(ddof=1) + 1e-12) * np.sqrt(ann_factor))
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "sortino": sortino}

def _max_drawdown(price: pd.Series):
    peak = price.cummax()
    dd   = price / peak - 1.0
    return float(dd.min()), dd

def _cross_corr_lags(x: pd.Series, y: pd.Series, max_lag: int) -> pd.Series:
    z = pd.concat([x, y], axis=1).dropna()
    xj, yj = z.iloc[:, 0], z.iloc[:, 1]
    lags = range(-max_lag, max_lag + 1)
    vals = [xj.corr(yj.shift(-lag)) for lag in lags]
    return pd.Series(vals, index=pd.Index(lags, name="lag"), name="cross_corr")

def _beta(asset_ret: pd.Series, bench_ret: pd.Series) -> float:
    z = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    a, b = z.iloc[:, 0].values, z.iloc[:, 1].values
    var_b = np.var(b, ddof=1)
    if var_b <= 0:
        return np.nan
    return float(np.cov(a, b, ddof=1)[0, 1] / var_b)

def _ols_expected(active: pd.Series, tracking: pd.Series):
    z = pd.concat([active.rename("active"), tracking.rename("tracking")], axis=1).dropna()
    y, x = z["active"].values, z["tracking"].values
    X    = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    yhat   = alpha + beta * x
    resid  = y - yhat
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2) + 1e-12)
    r2     = 1.0 - ss_res / ss_tot
    return (alpha, beta,
            pd.Series(yhat,  index=z.index, name="expected_active"),
            pd.Series(resid, index=z.index, name="residual"),
            r2)

def _get_pe(ticker: str):
    try:
        info = yf.Ticker(ticker.strip().upper()).info or {}
    except Exception:
        info = {}
    return info.get("trailingPE", np.nan), info.get("forwardPE", np.nan)

def _zscore(series: pd.Series, window: int) -> pd.Series:
    roll = series.rolling(window)
    return (series - roll.mean()) / (roll.std(ddof=1) + 1e-12)


# ── Styled HTML recommendations ───────────────────────────────────────────────
def _badge(text: str, color: str) -> str:
    return (f'<span style="background:{color}22;color:{color};border:1px solid {color}66;'
            f'border-radius:4px;padding:1px 7px;font-size:11px;font-weight:600;'
            f'letter-spacing:0.05em">{text}</span>')

def _row(label: str, value_html: str) -> str:
    return (f'<tr>'
            f'<td style="color:{_CLR_MUTED};padding:5px 16px 5px 0;white-space:nowrap">{label}</td>'
            f'<td style="padding:5px 0">{value_html}</td>'
            f'</tr>')

def _section(icon: str, title: str, body_html: str) -> str:
    return (f'<div style="margin-bottom:18px">'
            f'<div style="font-size:12px;font-weight:700;letter-spacing:0.08em;'
            f'color:{_CLR_ACCENT};margin-bottom:8px;padding-bottom:5px;'
            f'border-bottom:1px solid {_CLR_BORDER}">{icon}&nbsp; {title}</div>'
            f'{body_html}'
            f'</div>')

def _p(text: str, color: str = _CLR_TEXT) -> str:
    return f'<p style="margin:3px 0;color:{color};font-size:12px;line-height:1.6">{text}</p>'

def _generate_recommendations_html(
        name_a, name_b, met_a, met_b, mdd_a, mdd_b,
        corr, best_lag, best_cc, alpha, beta_model, r2,
        beta_a, beta_b, prices, ra, rb, roll_window,
        pe_a_tr, pe_a_fw, pe_b_tr, pe_b_fw,
        active_name, tracking_name, residual):

    sections = []

    # ── 1. Trend & Momentum ──
    rows = []
    for name, met, mdd in [(name_a, met_a, mdd_a), (name_b, met_b, mdd_b)]:
        ret, vol, sharpe = met["ann_return"], met["ann_vol"], met["sharpe"]
        trend_color = _CLR_GREEN if ret > 0.05 else (_CLR_RED if ret < -0.05 else _CLR_YELLOW)
        trend_lbl   = "BULLISH"  if ret > 0.05 else ("BEARISH" if ret < -0.05 else "NEUTRAL")
        sr_color    = _CLR_GREEN if sharpe > 1.0 else (_CLR_YELLOW if sharpe > 0.5 else _CLR_RED)
        sr_lbl      = "Strong"   if sharpe > 1.0 else ("Moderate" if sharpe > 0.5 else "Weak")
        val_html    = (f'{_badge(trend_lbl, trend_color)}&nbsp;'
                       f'<span style="color:{_CLR_MUTED};font-size:11px">'
                       f'Ret {ret:.1%} · Vol {vol:.1%} · Sharpe {sharpe:.2f} '
                       f'({_badge(sr_lbl, sr_color)})</span>')
        if mdd < -0.20:
            val_html += (f'&nbsp;{_badge(f"MDD {mdd:.1%}", _CLR_RED)}')
        rows.append(_row(name, val_html))
    sections.append(_section("📈", "TREND &amp; MOMENTUM",
                              f'<table style="border-collapse:collapse">{"".join(rows)}</table>'))

    # ── 2. Relative Strength ──
    ret_diff, sharpe_diff = met_a["ann_return"] - met_b["ann_return"], met_a["sharpe"] - met_b["sharpe"]
    if ret_diff > 0.03 and sharpe_diff > 0.2:
        winner, loser = name_a, name_b
    elif ret_diff < -0.03 and sharpe_diff < -0.2:
        winner, loser = name_b, name_a
    else:
        winner, loser = None, None
    if winner:
        rs_html = _p(f'{_badge("MOMENTUM TILT", _CLR_GREEN)} {winner} leads on both return and Sharpe. '
                     f'Consider <b>Long {winner} / Short {loser}</b> momentum exposure.', _CLR_TEXT)
    else:
        rs_html = _p(f'{_badge("NEUTRAL", _CLR_YELLOW)} No dominant leader — metrics are broadly comparable. '
                     f'Equal-weight or pair trade may be appropriate.')
    sections.append(_section("⚖️", f"RELATIVE STRENGTH ({name_a} vs {name_b})", rs_html))

    # ── 3. Correlation & Pair Trading ──
    if corr > 0.80:
        corr_lbl, corr_clr, corr_msg = "VERY HIGH", _CLR_GREEN, "Classic pair trade candidate — look for spread z-score divergence."
    elif corr > 0.50:
        corr_lbl, corr_clr, corr_msg = "MODERATE", _CLR_YELLOW, "Pair trading feasible but noisier. Use tighter z-score thresholds."
    elif corr > 0.0:
        corr_lbl, corr_clr, corr_msg = "LOW", _CLR_YELLOW, "Good for diversification; pair trading is higher risk."
    else:
        corr_lbl, corr_clr, corr_msg = "NEGATIVE", _CLR_RED, "Moves independently or inversely — useful as hedge or diversifier."

    spread  = (prices[name_a] / prices[name_a].iloc[0]) - (prices[name_b] / prices[name_b].iloc[0])
    zs      = _zscore(spread, int(roll_window))
    last_z  = float(zs.dropna().iloc[-1]) if not zs.dropna().empty else np.nan

    if not np.isnan(last_z):
        if last_z > 2.0:
            z_lbl, z_clr  = f"SHORT {name_a} / LONG {name_b}", _CLR_RED
            z_msg         = f"Spread is STRETCHED HIGH (z={last_z:.2f}) — mean-reversion signal."
        elif last_z < -2.0:
            z_lbl, z_clr  = f"LONG {name_a} / SHORT {name_b}", _CLR_GREEN
            z_msg         = f"Spread is STRETCHED LOW (z={last_z:.2f}) — mean-reversion signal."
        elif abs(last_z) > 1.0:
            z_lbl, z_clr  = "WATCH", _CLR_YELLOW
            z_msg         = f"Spread is elevated (z={last_z:.2f}) but not extreme — monitor for further move."
        else:
            z_lbl, z_clr  = "NEUTRAL", _CLR_MUTED
            z_msg         = f"Spread near mean (z={last_z:.2f}) — no active signal."
        z_html = (_p(f'{_badge(z_lbl, z_clr)} {z_msg}'))
    else:
        z_html = _p("Spread z-score unavailable.", _CLR_MUTED)

    corr_html = (_p(f'{_badge(corr_lbl, corr_clr)} Correlation = <b>{corr:.3f}</b> — {corr_msg}') + z_html)
    sections.append(_section("🔗", "CORRELATION &amp; PAIR TRADING", corr_html))

    # ── 4. Lead / Lag ──
    if abs(best_lag) >= 2 and abs(best_cc) > 0.15:
        leader = name_a if best_lag > 0 else name_b
        lagger = name_b if best_lag > 0 else name_a
        bars   = abs(best_lag)
        ll_html = _p(f'{_badge("LEAD/LAG DETECTED", _CLR_ACCENT)} '
                     f'<b>{leader}</b> tends to lead <b>{lagger}</b> by ~{bars} bars '
                     f'(peak |xcorr| = {abs(best_cc):.3f}). '
                     f'Use {leader} moves as an early entry signal for {lagger} trades.')
    else:
        ll_html = _p(f'{_badge("NO SIGNAL", _CLR_MUTED)} No reliable lead/lag detected '
                     f'(lag={best_lag}, xcorr={best_cc:.3f}). Treat as contemporaneous.')
    sections.append(_section("⏱️", "LEAD / LAG", ll_html))

    # ── 5. Expected Return Model ──
    last_resid = float(residual.iloc[-1]) if not residual.empty else np.nan
    resid_std  = float(residual.std(ddof=1))
    q_lbl      = "STRONG" if r2 > 0.70 else ("MODERATE" if r2 > 0.40 else "WEAK")
    q_clr      = _CLR_GREEN if r2 > 0.70 else (_CLR_YELLOW if r2 > 0.40 else _CLR_RED)

    model_rows = [
        _row("Model fit", f'{_badge(q_lbl, q_clr)} R² = {r2:.3f}'),
        _row("Alpha (α)",  f'<span style="color:{_CLR_GREEN if alpha>1e-5 else _CLR_RED}">{alpha:.6f}</span>'
                           + (f' — {active_name} generates excess return above {tracking_name}.' if alpha > 1e-5
                              else f' — {active_name} underperforms {tracking_name} on average.' if alpha < -1e-5
                              else '')),
        _row("Beta (β)",   f'{beta_model:.3f}'),
    ]
    if not np.isnan(last_resid):
        if last_resid > 1.5 * resid_std:
            r_lbl, r_clr, r_msg = "ABOVE MODEL", _CLR_GREEN, f"{active_name} returning more than predicted — watch for mean reversion down."
        elif last_resid < -1.5 * resid_std:
            r_lbl, r_clr, r_msg = "BELOW MODEL", _CLR_RED,   f"{active_name} returning less than predicted — watch for mean reversion up."
        else:
            r_lbl, r_clr, r_msg = "IN LINE",     _CLR_MUTED, f"{active_name} return broadly in line with model expectations."
        model_rows.append(_row("Latest residual",
                               f'{_badge(r_lbl, r_clr)} {last_resid:.6f} — {r_msg}'))
    sections.append(_section("🧮", f"EXPECTED RETURN MODEL ({active_name} ~ {tracking_name})",
                              f'<table style="border-collapse:collapse">{"".join(model_rows)}</table>'))

    # ── 6. Beta / Market Sensitivity ──
    beta_rows = []
    for name, b in [(name_a, beta_a), (name_b, beta_b)]:
        if np.isnan(b):
            beta_rows.append(_row(name, _badge("N/A", _CLR_MUTED)))
            continue
        if b > 1.3:
            bl, bc = "HIGH β", _CLR_RED
        elif b > 0.8:
            bl, bc = "MID β",  _CLR_YELLOW
        elif b > 0.3:
            bl, bc = "LOW β",  _CLR_GREEN
        else:
            bl, bc = "VERY LOW β", _CLR_ACCENT
        beta_rows.append(_row(name, f'{_badge(bl, bc)} {b:.2f}'))

    extra = ""
    if not (np.isnan(beta_a) or np.isnan(beta_b)) and abs(beta_a - beta_b) > 0.5:
        ratio = abs(beta_b / (beta_a + 1e-9))
        extra = _p(f'{_badge("SIZE ADJUSTMENT", _CLR_ACCENT)} Beta-neutral pair: '
                   f'for every 1 unit of {name_b}, hold <b>{ratio:.2f}</b> units of {name_a}.')
    sections.append(_section("📉", "MARKET SENSITIVITY (Beta vs Benchmark)",
                              f'<table style="border-collapse:collapse">{"".join(beta_rows)}</table>{extra}'))

    # ── 7. Valuation ──
    pe_rows = []
    for name, tr, fw in [(name_a, pe_a_tr, pe_a_fw), (name_b, pe_b_tr, pe_b_fw)]:
        if np.isnan(tr) and np.isnan(fw):
            pe_rows.append(_row(name, _badge("N/A", _CLR_MUTED) + ' <span style="color:' + _CLR_MUTED + ';font-size:11px">ETF / index / no data</span>'))
        else:
            tr_s = f'Trail: <b>{tr:.1f}x</b>' if not np.isnan(tr) else 'Trail: N/A'
            fw_s = f'Fwd: <b>{fw:.1f}x</b>'   if not np.isnan(fw) else 'Fwd: N/A'
            pe_rows.append(_row(name, f'<span style="color:{_CLR_TEXT}">{tr_s} &nbsp; {fw_s}</span>'))

    pe_note = ""
    if not (np.isnan(pe_a_tr) or np.isnan(pe_b_tr)):
        if pe_a_tr < pe_b_tr * 0.85:
            pe_note = _p(f'{_badge("DISCOUNT", _CLR_GREEN)} {name_a} trades at a notable P/E discount '
                         f'({pe_a_tr:.1f}x vs {pe_b_tr:.1f}x) — may be relatively undervalued.')
        elif pe_b_tr < pe_a_tr * 0.85:
            pe_note = _p(f'{_badge("DISCOUNT", _CLR_GREEN)} {name_b} trades at a notable P/E discount '
                         f'({pe_b_tr:.1f}x vs {pe_a_tr:.1f}x) — may be relatively undervalued.')
        else:
            pe_note = _p(f'{_badge("COMPARABLE", _CLR_MUTED)} Valuations are broadly similar.')
    sections.append(_section("💰", "VALUATION (P/E Ratios)",
                              f'<table style="border-collapse:collapse">{"".join(pe_rows)}</table>{pe_note}'))

    # ── 8. Summary Table ──
    trend_a = "Bullish" if met_a["ann_return"] > 0.05 else ("Bearish" if met_a["ann_return"] < -0.05 else "Neutral")
    trend_b = "Bullish" if met_b["ann_return"] > 0.05 else ("Bearish" if met_b["ann_return"] < -0.05 else "Neutral")
    tc_a    = _CLR_GREEN if trend_a == "Bullish" else (_CLR_RED if trend_a == "Bearish" else _CLR_YELLOW)
    tc_b    = _CLR_GREEN if trend_b == "Bullish" else (_CLR_RED if trend_b == "Bearish" else _CLR_YELLOW)

    if not np.isnan(last_z):
        if last_z > 2.0:   ps, pc = f"Short {name_a} / Long {name_b}", _CLR_RED
        elif last_z < -2.0: ps, pc = f"Long {name_a} / Short {name_b}", _CLR_GREEN
        else:               ps, pc = "No signal", _CLR_MUTED
    else:
        ps, pc = "N/A", _CLR_MUTED

    lag_s  = f"{name_a} leads" if best_lag > 0 else (f"{name_b} leads" if best_lag < 0 else "No lead/lag")
    alp_s  = "Positive" if alpha > 1e-5 else ("Negative" if alpha < -1e-5 else "Neutral")
    alp_c  = _CLR_GREEN if alpha > 1e-5 else (_CLR_RED if alpha < -1e-5 else _CLR_MUTED)

    sum_rows = [
        _row(f"{name_a} Trend",       _badge(trend_a, tc_a)),
        _row(f"{name_b} Trend",       _badge(trend_b, tc_b)),
        _row("Pair Spread",           _badge(ps, pc)),
        _row("Lead / Lag",            f'<span style="color:{_CLR_TEXT}">{lag_s}</span>'),
        _row(f"{name_a} Beta",        f'<span style="color:{_CLR_TEXT}">{beta_a:.2f}</span>' if not np.isnan(beta_a) else _badge("N/A", _CLR_MUTED)),
        _row(f"{name_b} Beta",        f'<span style="color:{_CLR_TEXT}">{beta_b:.2f}</span>' if not np.isnan(beta_b) else _badge("N/A", _CLR_MUTED)),
        _row(f"Alpha ({active_name})", _badge(alp_s, alp_c)),
    ]
    sections.append(_section("📋", "SUMMARY",
                              f'<table style="border-collapse:collapse">{"".join(sum_rows)}</table>'))

    # ── Assemble ──
    disclaimer = (
        f'<div style="margin-top:18px;padding:10px 14px;background:{_CLR_PANEL};'
        f'border-left:3px solid {_CLR_YELLOW};border-radius:0 4px 4px 0;'
        f'font-size:11px;color:{_CLR_MUTED};line-height:1.6">'
        f'⚠ For informational purposes only. All signals are derived from historical data. '
        f'Past performance does not guarantee future results. '
        f'Always apply your own risk management and consult a qualified financial adviser.</div>'
    )

    html = (
        f'<div style="font-family:\'IBM Plex Mono\',\'Courier New\',monospace;'
        f'background:{_CLR_BG};color:{_CLR_TEXT};padding:24px 28px;'
        f'border:1px solid {_CLR_BORDER};border-radius:8px;margin-top:16px">'
        f'<div style="font-size:15px;font-weight:700;letter-spacing:0.1em;'
        f'color:{_CLR_TEXT};margin-bottom:6px">TRADING RECOMMENDATIONS</div>'
        f'<div style="font-size:11px;color:{_CLR_MUTED};margin-bottom:20px;'
        f'padding-bottom:14px;border-bottom:1px solid {_CLR_BORDER}">'
        f'{name_a} / {name_b} pair analysis signal summary</div>'
        + "".join(sections)
        + disclaimer
        + '</div>'
    )
    return html


# ── Core analysis ──────────────────────────────────────────────────────────────
def run_analysis(ticker_a, ticker_b, benchmark, start_date, end_date,
                 interval, returns_kind, max_lag, roll_window, rf_annual,
                 tracking_choice, active_choice):

    pa = _download_prices(ticker_a, start_date, end_date, interval)
    pb = _download_prices(ticker_b, start_date, end_date, interval)
    prices = pd.concat([pa, pb], axis=1).dropna()
    name_a, name_b = prices.columns[0], prices.columns[1]

    ra = _returns(prices[name_a], returns_kind)
    rb = _returns(prices[name_b], returns_kind)
    rets = pd.concat([ra, rb], axis=1).dropna()
    ra, rb = rets[name_a], rets[name_b]

    tracking = ra if tracking_choice == name_a else rb
    active   = ra if active_choice   == name_a else rb

    bm_p  = _download_prices(benchmark, start_date, end_date, interval)
    bm_r  = _returns(bm_p, returns_kind).rename(benchmark.strip().upper())
    beta_a = _beta(ra, bm_r)
    beta_b = _beta(rb, bm_r)

    corr_same    = float(ra.corr(rb))
    rolling_corr = rets[name_a].rolling(int(roll_window)).corr(rets[name_b])

    xcorr    = _cross_corr_lags(ra, rb, int(max_lag))
    best_lag = int(xcorr.abs().idxmax())
    best_cc  = float(xcorr.loc[best_lag])

    alpha, beta_model, expected, residual, r2 = _ols_expected(active, tracking)

    norm          = prices / prices.iloc[0] * 100.0
    mdd_a, dd_a   = _max_drawdown(prices[name_a])
    mdd_b, dd_b   = _max_drawdown(prices[name_b])

    ann_factor = _annualization_factor(interval)
    met_a = _risk_metrics(ra, ann_factor, rf_annual)
    met_b = _risk_metrics(rb, ann_factor, rf_annual)

    pe_a_tr, pe_a_fw = _get_pe(ticker_a)
    pe_b_tr, pe_b_fw = _get_pe(ticker_b)

    summary = pd.DataFrame([
        {"ticker": name_a, "trailing_P/E": pe_a_tr, "forward_P/E": pe_a_fw,
         "ann_return_est": met_a["ann_return"], "ann_vol": met_a["ann_vol"],
         "sharpe": met_a["sharpe"], "sortino": met_a["sortino"],
         "max_drawdown": mdd_a, f"beta_vs_{benchmark.strip().upper()}": beta_a},
        {"ticker": name_b, "trailing_P/E": pe_b_tr, "forward_P/E": pe_b_fw,
         "ann_return_est": met_b["ann_return"], "ann_vol": met_b["ann_vol"],
         "sharpe": met_b["sharpe"], "sortino": met_b["sortino"],
         "max_drawdown": mdd_b, f"beta_vs_{benchmark.strip().upper()}": beta_b},
    ]).set_index("ticker")

    reco_html = _generate_recommendations_html(
        name_a=name_a, name_b=name_b, met_a=met_a, met_b=met_b,
        mdd_a=mdd_a, mdd_b=mdd_b, corr=corr_same,
        best_lag=best_lag, best_cc=best_cc,
        alpha=alpha, beta_model=beta_model, r2=r2,
        beta_a=beta_a, beta_b=beta_b,
        prices=prices, ra=ra, rb=rb, roll_window=roll_window,
        pe_a_tr=pe_a_tr, pe_a_fw=pe_a_fw,
        pe_b_tr=pe_b_tr, pe_b_fw=pe_b_fw,
        active_name=active.name, tracking_name=tracking.name,
        residual=residual,
    )

    # ── Build figure ────────────────────────────────────────────────────────
    subplot_titles = (
        f"Normalized Price (base=100)  ·  {_CHART_EXPLANATIONS['price']}",
        f"Returns  ·  {_CHART_EXPLANATIONS['returns']}",
        f"Rolling Correlation (window={roll_window})  ·  {_CHART_EXPLANATIONS['rollcorr']}",
        f"Cross-Correlation vs Lag  ·  {_CHART_EXPLANATIONS['xcorr']}",
        f"Lag Scatter (A vs B+{best_lag})  ·  {_CHART_EXPLANATIONS['scatter']}",
        f"Drawdown  ·  {_CHART_EXPLANATIONS['drawdown']}",
        f"Expected Active Return  (R²={r2:.3f})  ·  {_CHART_EXPLANATIONS['expected']}",
        f"Residual  ·  {_CHART_EXPLANATIONS['residual']}",
    )

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # Colour palette for traces
    ca, cb = _CLR_ACCENT, _CLR_GREEN

    # Row 1 — Price & Returns
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name_a], name=name_a,
                             line=dict(color=ca, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=norm.index, y=norm[name_b], name=name_b,
                             line=dict(color=cb, width=1.8)), row=1, col=1)

    fig.add_trace(go.Scatter(x=rets.index, y=ra, name=f"{name_a} ret",
                             line=dict(color=ca, width=1), opacity=0.85), row=1, col=2)
    fig.add_trace(go.Scatter(x=rets.index, y=rb, name=f"{name_b} ret",
                             line=dict(color=cb, width=1), opacity=0.85), row=1, col=2)

    # Row 2 — Rolling corr & Cross-corr
    rc_colors = [_CLR_GREEN if v >= 0 else _CLR_RED for v in rolling_corr.dropna()]
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name="rolling corr",
                             line=dict(color=_CLR_ACCENT, width=1.5),
                             fill="tozeroy", fillcolor=_rgba(_CLR_ACCENT, 0.09)), row=2, col=1)
    fig.add_hline(y=0, row=2, col=1, line_width=1, line_color=_CLR_BORDER)
    fig.add_hline(y=0.8,  row=2, col=1, line_width=1, line_dash="dot",
                  line_color=_CLR_GREEN, annotation_text="0.8", annotation_font_size=9,
                  annotation_font_color=_CLR_GREEN)
    fig.add_hline(y=-0.8, row=2, col=1, line_width=1, line_dash="dot",
                  line_color=_CLR_RED, annotation_text="-0.8", annotation_font_size=9,
                  annotation_font_color=_CLR_RED)

    bar_colors = [_CLR_GREEN if v >= 0 else _CLR_RED for v in xcorr.values]
    fig.add_trace(go.Bar(x=xcorr.index.astype(int), y=xcorr.values, name="xcorr",
                         marker_color=bar_colors, opacity=0.8), row=2, col=2)
    fig.add_vline(x=best_lag, row=2, col=2, line_width=2,
                  line_color=_CLR_YELLOW, line_dash="dot",
                  annotation_text=f"lag={best_lag}", annotation_font_size=9,
                  annotation_font_color=_CLR_YELLOW)

    # Row 3 — Lag scatter & Drawdown
    b_shift = rb.shift(-best_lag)
    scat = pd.concat([ra.rename("A"), b_shift.rename("B_shift")], axis=1).dropna()
    fig.add_trace(go.Scatter(x=scat["A"], y=scat["B_shift"], mode="markers",
                             name="lagged scatter",
                             marker=dict(color=_CLR_ACCENT, size=4, opacity=0.5)), row=3, col=1)

    dd_df = pd.concat([dd_a.rename(name_a), dd_b.rename(name_b)], axis=1).dropna()
    fig.add_trace(go.Scatter(x=dd_df.index, y=dd_df[name_a], name=f"{name_a} DD",
                             line=dict(color=ca, width=1.5),
                             fill="tozeroy", fillcolor=_rgba(ca, 0.09)), row=3, col=2)
    fig.add_trace(go.Scatter(x=dd_df.index, y=dd_df[name_b], name=f"{name_b} DD",
                             line=dict(color=cb, width=1.5),
                             fill="tozeroy", fillcolor=_rgba(cb, 0.09)), row=3, col=2)
    fig.add_hline(y=-0.20, row=3, col=2, line_width=1, line_dash="dot",
                  line_color=_CLR_RED, annotation_text="-20%", annotation_font_size=9,
                  annotation_font_color=_CLR_RED)

    # Row 4 — Expected return & Residual
    model_df = pd.concat([active.rename("active"), expected.rename("expected")], axis=1).dropna()
    fig.add_trace(go.Scatter(x=model_df.index, y=model_df["active"],
                             name=f"Actual ({active.name})",
                             line=dict(color=_CLR_ACCENT, width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=model_df.index, y=model_df["expected"],
                             name="Model expected",
                             line=dict(color=_CLR_RED, width=1.5, dash="dot")), row=4, col=1)

    fig.add_trace(go.Scatter(x=residual.index, y=residual, name="residual",
                             line=dict(color=_CLR_YELLOW, width=1.2),
                             fill="tozeroy", fillcolor=_rgba(_CLR_YELLOW, 0.08)), row=4, col=2)
    fig.add_hline(y=0, row=4, col=2, line_width=1, line_color=_CLR_BORDER)

    # Global layout
    fig.update_layout(
        height=1380,
        showlegend=True,
        hovermode="x unified",
        title_text=(
            f"<b>{name_a} vs {name_b}</b>  ·  interval={interval}  ·  "
            f"corr={corr_same:.3f}  ·  best lag={best_lag} (xcorr={best_cc:.3f})  ·  "
            f"α={alpha:.5f}  β={beta_model:.3f}  R²={r2:.3f}"
        ),
        **_PLOTLY_THEME,
    )

    # Apply axis styling to all subplots
    for i in range(1, 9):
        fig.update_xaxes(**_AXIS_STYLE, row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
        fig.update_yaxes(**_AXIS_STYLE, row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)

    # Style the subplot title annotations (smaller, muted colour)
    for ann in fig.layout.annotations:
        ann.font.size  = 11
        ann.font.color = _CLR_MUTED

    return summary, fig, reco_html


# ── Dashboard UI ──────────────────────────────────────────────────────────────
# Inject Google Font for the widget labels
display(HTML(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet">'
    '<style>'
    '  .widget-label { font-family: "IBM Plex Mono", monospace !important; font-size: 12px !important; }'
    '  .widget-text input, .widget-dropdown select, .widget-datepicker input {'
    '    font-family: "IBM Plex Mono", monospace !important; font-size: 12px !important;'
    '    background: #1a1d27 !important; color: #e2e6f0 !important; border-color: #2e3248 !important; }'
    '  .widget-button button { font-family: "IBM Plex Mono", monospace !important; font-size: 12px !important; }'
    '</style>'
))

out = widgets.Output()

ticker_a_w = widgets.Text(value="AAPL", description="Stock A:",    layout=widgets.Layout(width="220px"))
ticker_b_w = widgets.Text(value="MSFT", description="Stock B:",    layout=widgets.Layout(width="220px"))
bench_w    = widgets.Text(value="SPY",  description="Benchmark:",  layout=widgets.Layout(width="220px"))

start_w = widgets.DatePicker(description="Start:", value=(pd.Timestamp.today() - pd.Timedelta(days=365*3)).date())
end_w   = widgets.DatePicker(description="End:",   value=pd.Timestamp.today().date())

interval_options = ["1wk", "1d", "1h", "30m", "15m", "5m"]
interval_slider  = widgets.IntSlider(value=0, min=0, max=len(interval_options)-1, step=1, description="Timeframe:")
interval_label   = widgets.Label(value=f"→ {interval_options[0]}")

def _sync_interval_label(_=None):
    interval_label.value = f"→ {interval_options[interval_slider.value]}"
interval_slider.observe(_sync_interval_label, names="value")

returns_w  = widgets.Dropdown(options=[("Log returns", "log"), ("Pct returns", "pct")], value="log", description="Returns:")
max_lag_w  = widgets.IntSlider(value=30, min=5, max=240, step=5,    description="Max lag:")
roll_w     = widgets.IntSlider(value=60, min=10, max=252, step=5,   description="Roll win:")
rf_w       = widgets.FloatSlider(value=0.02, min=0.0, max=0.10, step=0.0025,
                                  description="Rf (ann):", readout_format=".3f")

tracking_w = widgets.Dropdown(options=[ticker_a_w.value.upper(), ticker_b_w.value.upper()], description="Tracking:")
active_w   = widgets.Dropdown(options=[ticker_a_w.value.upper(), ticker_b_w.value.upper()], description="Active:")

def _refresh_pair_options(_=None):
    a, b = ticker_a_w.value.strip().upper(), ticker_b_w.value.strip().upper()
    opts = [a, b]
    tracking_w.options = opts
    active_w.options   = opts
    if tracking_w.value not in opts: tracking_w.value = a
    if active_w.value   not in opts: active_w.value   = b

ticker_a_w.observe(_refresh_pair_options, names="value")
ticker_b_w.observe(_refresh_pair_options, names="value")
_refresh_pair_options()

calc_btn  = widgets.Button(description="▶  Calculate", button_style="primary",
                            layout=widgets.Layout(width="140px"))
reset_btn = widgets.Button(description="↺  Reset",
                            layout=widgets.Layout(width="110px"))

def _reset(_):
    ticker_a_w.value     = "AAPL"
    ticker_b_w.value     = "MSFT"
    bench_w.value        = "SPY"
    start_w.value        = (pd.Timestamp.today() - pd.Timedelta(days=365*3)).date()
    end_w.value          = pd.Timestamp.today().date()
    interval_slider.value = 0
    returns_w.value      = "log"
    max_lag_w.value      = 30
    roll_w.value         = 60
    rf_w.value           = 0.02
    _refresh_pair_options()
    with out:
        clear_output(wait=True)
        display(HTML(f'<p style="font-family:IBM Plex Mono,monospace;color:{_CLR_MUTED};font-size:12px">'
                     f'Reset complete. Click ▶ Calculate to run.</p>'))

def _calculate(_):
    with out:
        clear_output(wait=True)
        display(HTML(f'<p style="font-family:IBM Plex Mono,monospace;color:{_CLR_ACCENT};font-size:12px">'
                     f'⏳ Downloading data and computing…</p>'))
        try:
            interval = interval_options[interval_slider.value]
            summary, fig, reco_html = run_analysis(
                ticker_a=ticker_a_w.value, ticker_b=ticker_b_w.value,
                benchmark=bench_w.value,
                start_date=start_w.value, end_date=end_w.value,
                interval=interval, returns_kind=returns_w.value,
                max_lag=int(max_lag_w.value), roll_window=int(roll_w.value),
                rf_annual=float(rf_w.value),
                tracking_choice=tracking_w.value, active_choice=active_w.value,
            )
            clear_output(wait=True)

            # Styled summary table header
            display(HTML(
                f'<div style="font-family:IBM Plex Mono,monospace;font-size:13px;'
                f'font-weight:700;letter-spacing:0.08em;color:{_CLR_TEXT};'
                f'margin:8px 0 4px">METRICS SUMMARY</div>'
            ))
            display(summary.style
                .set_properties(**{"font-family": "IBM Plex Mono, monospace",
                                   "font-size": "12px", "border": "none"})
                .set_table_styles([
                    {"selector": "th", "props": [("font-family", "IBM Plex Mono, monospace"),
                                                  ("font-size", "11px"),
                                                  ("color", _CLR_MUTED),
                                                  ("border-bottom", f"1px solid {_CLR_BORDER}"),
                                                  ("padding", "4px 10px 4px 0")]},
                    {"selector": "td", "props": [("padding", "4px 10px 4px 0")]},
                ])
                .format({
                    "trailing_P/E":   "{:.2f}",
                    "forward_P/E":    "{:.2f}",
                    "ann_return_est": "{:.2%}",
                    "ann_vol":        "{:.2%}",
                    "sharpe":         "{:.2f}",
                    "sortino":        "{:.2f}",
                    "max_drawdown":   "{:.2%}",
                })
            )

            display(HTML(reco_html))
            fig.show()

        except Exception as e:
            clear_output(wait=True)
            display(HTML(
                f'<div style="font-family:IBM Plex Mono,monospace;color:{_CLR_RED};'
                f'font-size:12px;padding:10px;border-left:3px solid {_CLR_RED}">'
                f'<b>Error:</b> {e}<br><br>'
                f'<span style="color:{_CLR_MUTED}">Tip: Intraday intervals (5m/15m/30m/1h) '
                f'often require a shorter date range due to Yahoo Finance limits.</span></div>'
            ))

calc_btn.on_click(_calculate)
reset_btn.on_click(_reset)

display(widgets.VBox([
    widgets.HBox([ticker_a_w, ticker_b_w, bench_w]),
    widgets.HBox([start_w, end_w]),
    widgets.HBox([interval_slider, interval_label, returns_w]),
    widgets.HBox([tracking_w, active_w]),
    widgets.HBox([max_lag_w, roll_w, rf_w]),
    widgets.HBox([calc_btn, reset_btn]),
    out,
]))

with out:
    display(HTML(
        f'<p style="font-family:IBM Plex Mono,monospace;color:{_CLR_MUTED};font-size:12px;margin:8px 0">'
        f'Set tickers and timeframe, choose Tracking / Active, then click ▶ Calculate.<br>'
        f'Weekly (1wk) works for any date range · Intraday may need a shorter window.</p>'
    ))


# In[ ]:




