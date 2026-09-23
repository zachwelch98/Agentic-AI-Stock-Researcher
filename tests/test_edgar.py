from tools.edgar import html_to_text, read_sec_filing, select_filings

_RECENT = {
    "form": ["4", "8-K", "4", "DEF 14A", "10-Q"],
    "filingDate": ["2026-09-14", "2026-08-28", "2026-08-01", "2026-05-01", "2026-04-01"],
    "accessionNumber": ["0001-26-000001", "0001-26-000002", "0001-26-000003", "0001-26-000004", "0001-26-000005"],
    "primaryDocument": ["a.xml", "b.htm", "c.xml", "d.htm", "e.htm"],
    "_cik": ["123"] * 5,
}


def test_select_filings_default_keeps_order_and_builds_urls():
    filings = select_filings(_RECENT)
    assert [f["form"] for f in filings] == ["4", "8-K", "4", "DEF 14A", "10-Q"]
    assert filings[1]["document_url"] == "https://www.sec.gov/Archives/edgar/data/123/000126000002/b.htm"


def test_select_filings_filters_before_limit():
    assert [f["form"] for f in select_filings(_RECENT, forms=["def 14a"], limit=1)] == ["DEF 14A"]
    assert len(select_filings(_RECENT, forms=["4"], limit=1)) == 1


def test_html_to_text_strips_markup_and_scripts():
    assert html_to_text("<html><script>x=1</script><p>Net&nbsp;income <b>3</b></p></html>").endswith("3")
    assert "x=1" not in html_to_text("<script>x=1</script>hi")


async def test_read_sec_filing_only_reads_sec_archives():
    out = await read_sec_filing.ainvoke({"url": "https://example.com/x"})
    assert out.startswith("refused")
