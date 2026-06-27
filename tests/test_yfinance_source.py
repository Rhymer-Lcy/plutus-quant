"""The rate-limit-resilient batched downloader (yfinance_source). yf.download is monkeypatched, so
these never touch the network; they lock the request-pacing contract that keeps scheduled pulls
under Yahoo's per-IP rate limit (the cause of the 2026-06-26 YFRateLimitError run failure)."""
import pandas as pd

from plutus.data.sources import yfinance_source as yfs


def _fake_download_factory(calls):
    """A yf.download stand-in that records each call and returns a yfinance-shaped frame:
    flat columns for a single ticker, MultiIndex (field, ticker) for several."""
    idx = pd.to_datetime(["2026-01-02", "2026-01-05"])

    def fake_download(chunk, **kwargs):
        chunk = list(chunk)
        calls.append({"chunk": chunk, "threads": kwargs.get("threads"),
                      "auto_adjust": kwargs.get("auto_adjust")})
        if len(chunk) == 1:
            return pd.DataFrame({"Open": [1.0, 2.0], "Close": [10.0, 11.0]}, index=idx)
        data = {(field, t): [base + j, base + j + 1]
                for field, base in (("Open", 1.0), ("Close", 10.0))
                for j, t in enumerate(chunk)}
        return pd.DataFrame(data, index=idx)

    return fake_download


def test_download_batches_serially_with_inter_batch_sleep(monkeypatch):
    monkeypatch.setattr(yfs, "_YF_BATCH", 2)
    calls, sleeps = [], []
    monkeypatch.setattr(yfs.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(yfs.yf, "download", _fake_download_factory(calls))

    out = yfs.adjusted_close_panel(["A", "B", "C", "D", "E"], "2026-01-01", "2026-02-01")

    assert [c["chunk"] for c in calls] == [["A", "B"], ["C", "D"], ["E"]]   # 5 names, batch 2 -> 3 chunks
    assert all(c["threads"] is False for c in calls)                       # serialized, no burst
    assert all(c["auto_adjust"] is True for c in calls)
    assert sleeps == [yfs._YF_SLEEP_SEC, yfs._YF_SLEEP_SEC]                # between chunks, not after the last
    assert sorted(out.columns) == ["A", "B", "C", "D", "E"]               # every chunk's Close concatenated
    assert list(out.index) == [pd.Timestamp("2026-01-02"), pd.Timestamp("2026-01-05")]
    assert out.index.name == "date"


def test_raw_close_panel_requests_unadjusted(monkeypatch):
    calls = []
    monkeypatch.setattr(yfs.time, "sleep", lambda s: None)
    monkeypatch.setattr(yfs.yf, "download", _fake_download_factory(calls))

    out = yfs.raw_close_panel(["A"], "2026-01-01", "2026-02-01")

    assert all(c["auto_adjust"] is False for c in calls)                  # unadjusted -> as-traded price
    assert list(out.columns) == ["A"]


def test_empty_tickers_short_circuits_without_download(monkeypatch):
    called = []
    monkeypatch.setattr(yfs.yf, "download", lambda *a, **k: called.append(1))

    assert yfs.adjusted_close_panel([], "2026-01-01", "2026-02-01").empty
    assert yfs.raw_close_panel([], "2026-01-01", "2026-02-01").empty
    assert called == []                                                   # no network call for an empty book
