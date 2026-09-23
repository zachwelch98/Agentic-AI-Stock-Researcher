from tools.technical_indicators import compute_technical_indicators


def _bars(closes, volumes=None):
    volumes = volumes or [1000] * len(closes)
    return [
        {"date": f"2026-01-{i + 1:02d}" if i < 31 else f"2026-02-{i - 30:02d}" if i < 59 else f"2026-03-{i - 58:02d}",
         "open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": v}
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


def test_downtrend_with_bullish_macd_crossover_is_not_called_bearish():
    # long slide, then a small bounce: price below SMAs, MACD < 0 but above its signal line
    closes = [10 - 0.1 * i for i in range(55)] + [4.4, 4.3, 4.2, 4.3, 4.5]
    out = compute_technical_indicators(_bars(closes))

    sig = out["signals"]
    assert sig["trend"].startswith("downtrend")
    assert out["macd"] < 0 and out["macd_diff"] > 0
    assert sig["macd_state"].startswith("bullish crossover forming")


def test_support_resistance_and_volume_window_reported():
    closes = [5.0 + (i % 5) * 0.1 for i in range(60)]
    volumes = [1000] * 50 + [500] * 10
    sig = compute_technical_indicators(_bars(closes, volumes))["signals"]

    assert sig["support_20d"] < sig["resistance_20d"]
    assert "support_50d" in sig
    assert "last 10 sessions" in sig["volume_trend"] and "-50.0%" in sig["volume_trend"]


def test_short_history_has_no_unsupported_signals():
    out = compute_technical_indicators(_bars([1.0, 1.1, 1.2]))
    assert "macd_state" not in out["signals"] and "support_20d" not in out["signals"]
