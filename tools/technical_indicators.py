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

    return result
