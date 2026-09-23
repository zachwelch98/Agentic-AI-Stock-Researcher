"""Thin wrapper around the `ta` library for the technical-analysis researcher.

Takes OHLCV bars (oldest -> newest) and returns the latest reading of each
indicator, only computing indicators for which there's enough history.
"""

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD, SMAIndicator


def compute_technical_indicators(bars: list[dict]) -> dict:
    if len(bars) < 2:
        return {"error": "insufficient price history for technical indicators"}

    df = pd.DataFrame(bars).sort_values("date").reset_index(drop=True)
    close = df["close"]
    volume = df["volume"]

    result: dict = {"latest_close": float(close.iloc[-1])}

    if len(close) >= 20:
        result["sma_20"] = float(SMAIndicator(close, window=20).sma_indicator().iloc[-1])
    if len(close) >= 50:
        result["sma_50"] = float(SMAIndicator(close, window=50).sma_indicator().iloc[-1])
    if len(close) >= 12:
        result["ema_12"] = float(EMAIndicator(close, window=12).ema_indicator().iloc[-1])
    if len(close) >= 14:
        result["rsi_14"] = float(RSIIndicator(close, window=14).rsi().iloc[-1])
    if len(close) >= 26:
        macd = MACD(close)
        result["macd"] = float(macd.macd().iloc[-1])
        result["macd_signal"] = float(macd.macd_signal().iloc[-1])
        result["macd_diff"] = float(macd.macd_diff().iloc[-1])
    if len(volume) >= 20:
        recent_avg = volume.iloc[-10:].mean()
        prior_avg = volume.iloc[-20:-10].mean()
        if prior_avg > 0:
            result["volume_trend_pct"] = float((recent_avg - prior_avg) / prior_avg * 100)

    result["signals"] = _signals(result, df)
    return result


def _signals(ind: dict, df: pd.DataFrame) -> dict:
    """Plain-language readings of the indicators, computed in code so the LLM
    quotes them instead of re-deriving (and occasionally misreading) them."""
    signals: dict = {}
    close = ind["latest_close"]

    above = [name for name in ("sma_20", "sma_50") if name in ind and close > ind[name]]
    below = [name for name in ("sma_20", "sma_50") if name in ind and close < ind[name]]
    if above and not below:
        signals["trend"] = f"uptrend: close is above {' and '.join(above)}"
    elif below and not above:
        signals["trend"] = f"downtrend: close is below {' and '.join(below)}"
    elif above and below:
        signals["trend"] = f"mixed: close is above {' and '.join(above)} but below {' and '.join(below)}"

    if "rsi_14" in ind:
        rsi = ind["rsi_14"]
        signals["rsi_state"] = "overbought (>70)" if rsi > 70 else "oversold (<30)" if rsi < 30 else "neutral (30-70)"

    if "macd_diff" in ind:
        diff, macd = ind["macd_diff"], ind["macd"]
        side = "MACD above signal line" if diff > 0 else "MACD below signal line"
        zero = "above zero" if macd > 0 else "below zero"
        if diff > 0 and macd < 0:
            signals["macd_state"] = f"bullish crossover forming: {side}, but still {zero}"
        elif diff < 0 and macd > 0:
            signals["macd_state"] = f"bearish crossover forming: {side}, but still {zero}"
        else:
            signals["macd_state"] = f"{'bullish' if diff > 0 else 'bearish'}: {side}, {zero}"

    if len(df) >= 20:
        signals["support_20d"] = float(df["low"].iloc[-20:].min())
        signals["resistance_20d"] = float(df["high"].iloc[-20:].max())
    if len(df) >= 50:
        signals["support_50d"] = float(df["low"].iloc[-50:].min())
        signals["resistance_50d"] = float(df["high"].iloc[-50:].max())

    if "volume_trend_pct" in ind:
        signals["volume_trend"] = (
            f"average volume of the last 10 sessions is {ind['volume_trend_pct']:+.1f}% vs. the prior 10 sessions"
        )
    return signals
