from agents.screener import (
    FinalSelectionOutput,
    accept_user_ticker,
    compute_composite_score,
    llm_judgment_screen,
    screen_candidates,
)
from app.graph.state import make_initial_state


def test_composite_score_all_criteria_beat_median():
    score = compute_composite_score(
        pe_ratio=8.0,
        pb_ratio=0.9,
        debt_to_equity=0.5,
        dividend_yield=0.02,
        pe_median=10.0,
        pb_median=1.0,
        de_median=0.8,
    )
    assert score == 3.5  # +1 P/E, +1 P/B, +1 D/E, +0.5 dividend bonus


def test_composite_score_negative_pe_is_never_rewarded():
    score = compute_composite_score(
        pe_ratio=-5.0,
        pb_ratio=1.5,
        debt_to_equity=1.5,
        dividend_yield=None,
        pe_median=10.0,
        pb_median=1.0,
        de_median=0.8,
    )
    assert score == 0.0


def test_composite_score_missing_fields_score_zero_for_that_criterion():
    score = compute_composite_score(
        pe_ratio=None,
        pb_ratio=None,
        debt_to_equity=None,
        dividend_yield=None,
        pe_median=10.0,
        pb_median=1.0,
        de_median=0.8,
    )
    assert score == 0.0


def test_composite_score_dividend_bonus_is_not_a_filter():
    # REIT-shaped candidate: weak on P/E-style metrics but pays a dividend.
    no_dividend = compute_composite_score(
        pe_ratio=None,
        pb_ratio=2.0,
        debt_to_equity=2.0,
        dividend_yield=None,
        pe_median=10.0,
        pb_median=1.0,
        de_median=0.8,
    )
    with_dividend = compute_composite_score(
        pe_ratio=None,
        pb_ratio=2.0,
        debt_to_equity=2.0,
        dividend_yield=0.06,
        pe_median=10.0,
        pb_median=1.0,
        de_median=0.8,
    )
    assert with_dividend == no_dividend + 0.5


def _candidate(ticker, market_cap=5e9, avg_dollar_volume=2e7, **overrides):
    base = {
        "ticker": ticker,
        "company_name": f"{ticker} Inc.",
        "market_cap": market_cap,
        "avg_dollar_volume": avg_dollar_volume,
        "pe_ratio": 10.0,
        "pb_ratio": 1.0,
        "debt_to_equity": 1.0,
        "dividend_yield": 0.0,
    }
    base.update(overrides)
    return base


def test_screen_candidates_filters_below_market_cap_floor():
    raw = [_candidate("TOO_SMALL", market_cap=1e8), _candidate("BIG_ENOUGH", market_cap=5e8)]
    result = screen_candidates(raw)
    assert [c["ticker"] for c in result] == ["BIG_ENOUGH"]


def test_screen_candidates_filters_below_liquidity_floor():
    raw = [_candidate("ILLIQUID", avg_dollar_volume=1e5), _candidate("LIQUID", avg_dollar_volume=5e6)]
    result = screen_candidates(raw)
    assert [c["ticker"] for c in result] == ["LIQUID"]


def test_screen_candidates_ranks_by_score_desc_tiebreak_market_cap():
    raw = [
        _candidate("A", pe_ratio=5.0, pb_ratio=0.5, debt_to_equity=0.5, dividend_yield=0.01, market_cap=1e9),
        _candidate("B", pe_ratio=20.0, pb_ratio=2.0, debt_to_equity=2.0, dividend_yield=0.0, market_cap=2e9),
        _candidate("C", pe_ratio=5.0, pb_ratio=0.5, debt_to_equity=0.5, dividend_yield=0.01, market_cap=3e9),
    ]
    result = screen_candidates(raw)
    tickers = [c["ticker"] for c in result]
    # A and C tie on score; C has the larger market cap so it ranks above A. B scores worst.
    assert tickers == ["C", "A", "B"]


def test_screen_candidates_empty_input_returns_empty():
    assert screen_candidates([]) == []


def test_screen_candidates_all_filtered_out_returns_empty():
    raw = [_candidate("TOO_SMALL", market_cap=1e6)]
    assert screen_candidates(raw) == []


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, _messages):
        return self._result


class _FakeChat:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _schema):
        return _FakeStructuredLLM(self._result)


def _screened_candidates():
    return [
        {"ticker": "RB1", "composite_score": 3.5},
        {"ticker": "RB2", "composite_score": 1.0},
    ]


async def test_llm_judgment_screen_honors_valid_selection(monkeypatch):
    monkeypatch.setattr(
        "agents.screener.get_sonnet",
        lambda: _FakeChat(FinalSelectionOutput(selected_ticker="RB2", justification="qualitative override")),
    )
    state = make_initial_state("Regional Banks", "job")
    state["screened_candidates"] = _screened_candidates()

    result = await llm_judgment_screen(state)

    assert result["final_candidates"] == ["RB2"]
    assert result["screener_justification"] == "qualitative override"


async def test_llm_judgment_screen_falls_back_on_hallucinated_ticker(monkeypatch):
    monkeypatch.setattr(
        "agents.screener.get_sonnet",
        lambda: _FakeChat(FinalSelectionOutput(selected_ticker="NOT_A_CANDIDATE", justification="oops")),
    )
    state = make_initial_state("Regional Banks", "job")
    state["screened_candidates"] = _screened_candidates()

    result = await llm_judgment_screen(state)

    assert result["final_candidates"] == ["RB1"]  # highest composite_score wins the fallback


async def test_accept_user_ticker_bypasses_screening():
    state = make_initial_state("", "job", user_supplied_ticker="JPM")

    result = await accept_user_ticker(state)

    assert result["final_candidates"] == ["JPM"]
    assert result["used_llm_fallback_tickers"] is False
    assert result["status"] == "researching"


class _FakeMcpTool:
    def __init__(self, payloads: dict[str, dict | Exception]):
        self._payloads = payloads

    async def ainvoke(self, args: dict):
        payload = self._payloads[args["ticker"]]
        if isinstance(payload, Exception):
            raise payload
        return payload


async def test_resolve_identity_pins_ticker_to_company(monkeypatch):
    from agents.screener import resolve_identity

    fake = _FakeMcpTool({"TE": {"ticker": "TE", "company_name": "T1 Energy Inc", "website": "https://t1energy.com/", "market_cap": 1.28e9}})
    monkeypatch.setattr("agents.screener.get_mcp_tool", lambda name: fake)
    state = make_initial_state("", "job", user_supplied_ticker="TE")
    state["final_candidates"] = ["TE"]

    result = await resolve_identity(state)

    assert result["company_profiles"]["TE"]["company_name"] == "T1 Energy Inc"
    assert result["company_profiles"]["TE"]["website"] == "https://t1energy.com/"
    assert "status" not in result


async def test_resolve_identity_fails_on_unresolvable_ticker(monkeypatch):
    from agents.screener import resolve_identity

    fake = _FakeMcpTool({"ZZZZ": {"ticker": "ZZZZ", "company_name": None}})
    monkeypatch.setattr("agents.screener.get_mcp_tool", lambda name: fake)
    state = make_initial_state("", "job", user_supplied_ticker="ZZZZ")
    state["final_candidates"] = ["ZZZZ"]

    result = await resolve_identity(state)

    assert result["status"] == "failed"
    assert "ZZZZ" in result["error"]


async def test_resolve_identity_fails_when_lookup_raises(monkeypatch):
    from agents.screener import resolve_identity

    monkeypatch.setattr("agents.screener.get_mcp_tool", lambda name: _FakeMcpTool({"TE": RuntimeError("boom")}))
    state = make_initial_state("", "job", user_supplied_ticker="TE")
    state["final_candidates"] = ["TE"]

    assert (await resolve_identity(state))["status"] == "failed"
