"""Catalyst gap events: is the move over by the time the news is public? (issue #3)

A biotech catalyst (trial data, FDA action, M&A) is released outside trading hours by design,
so it arrives as an overnight GAP -- nobody can trade into it. This module measures what is
left AFTER the gap, for an entry that is strictly later than the public release.

The daily total return is decomposed split- and dividend-immune:

    intraday[t]  = close[t] / open[t] - 1          same-day ratio, so an overnight split or
                                                   dividend cannot contaminate it
    overnight[t] = (1 + DlyRet[t]) / (1 + intraday[t]) - 1

`overnight` is the gap (the part you missed); `intraday` is the part an open-entry can still
earn on the event day. Both entries EXIT at the same close, so they differ by exactly the
entry-day intraday move:

    CLOSE entry, horizon h: hold close[t] -> close[t+h]      (h daily returns)
    OPEN  entry, horizon h: hold open[t]  -> close[t+h]      (entry-day intraday + h dailies)

Returns are ABNORMAL (the name minus the same-day equal-weight mean of the universe), so a
sector-wide move is not credited to the event. A name that delists inside the window
contributes the days it actually traded -- CRSP's DlyRet carries the delisting return on the
final day, which is the position being liquidated, not a gap in the data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decompose_overnight(dlyret: pd.DataFrame, open_raw: pd.DataFrame,
                        close_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(overnight, intraday) panels from the total return and the raw open/close.

    Raw same-day open and close are used for `intraday` precisely because their ratio is
    immune to splits and dividends; `overnight` is then backed out of the adjusted total
    return, so the gap is adjusted too."""
    intraday = close_raw / open_raw - 1.0
    overnight = (1.0 + dlyret) / (1.0 + intraday) - 1.0
    return overnight, intraday


def find_events(overnight: pd.DataFrame, close_raw: pd.DataFrame, threshold: float = 0.20,
                min_history: int = 20, eligible: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every (name, date) whose overnight gap is at least `threshold`.

    `eligible` is the TRADABILITY mask (price and market-cap floors) applied to the EVENT DAY
    only: it decides whether the event could have been traded when it happened. It must NOT be
    used to truncate the holding period -- a name that gaps up and then craters below the price
    floor still hands its losses to whoever bought, and dropping those days would bias the
    measured drift upward.

    `min_history` prior traded days are required so a freshly listed name's first noisy prints
    are not read as catalysts (an implementation choice fixed BEFORE any return was computed;
    the frozen design left it open). Returns a long frame [permno, date, gap], sorted."""
    history_ok = close_raw.notna().cumsum().shift(1) >= min_history
    hit = (overnight >= threshold) & history_ok & overnight.notna()
    if eligible is not None:
        hit &= eligible.reindex(index=hit.index, columns=hit.columns).fillna(False)
    ev = (overnight.where(hit).stack().rename("gap").reset_index()
          .rename(columns={"level_0": "date", "level_1": "permno"}))
    ev.columns = ["date", "permno", "gap"]
    return ev.sort_values(["date", "permno"]).reset_index(drop=True)


def _sum_available(series: np.ndarray) -> tuple[float, int]:
    """Sum the non-NaN entries and report how many there were (a delisted name simply has
    fewer -- the position ended when the name did)."""
    ok = ~np.isnan(series)
    return float(series[ok].sum()), int(ok.sum())


def event_cars(events: pd.DataFrame, abn_cc: pd.DataFrame, abn_intra: pd.DataFrame,
               halfspread: pd.DataFrame, horizons: tuple[int, ...],
               runup_days: int = 10) -> pd.DataFrame:
    """Per-event cumulative ABNORMAL returns for both entries, gross and net of the name's own
    round-trip half-spread.

    One row per event with, for each horizon h: `close_h` / `open_h` (gross) and
    `close_h_net` / `open_h_net`. Also `runup` (abnormal CAR over the `runup_days` before the
    event -- anticipation/leakage) and `n_days_h` (days the position actually survived).

    COST: entry half-spread on the event day + exit half-spread on the exit day (the last day
    the name traded, if it delisted first). CRSP quotes the CLOSING bid/ask, so the OPEN entry
    is charged a closing spread it would not really get -- opening spreads on a catalyst day are
    wider, so every open-entry NET figure here is OPTIMISTIC. Disclosed, not corrected."""
    idx = abn_cc.index
    rows = []
    for date, permno, gap in zip(events["date"], events["permno"], events["gap"]):
        if permno not in abn_cc.columns:
            continue
        pos = idx.get_loc(date)
        cc = abn_cc[permno].to_numpy()
        hs = halfspread[permno].to_numpy()
        entry_hs = hs[pos] if pos < len(hs) and not np.isnan(hs[pos]) else np.nan
        intra = abn_intra[permno].to_numpy()[pos]

        rup_lo = max(pos - runup_days, 0)
        runup, _ = _sum_available(cc[rup_lo:pos])

        row = {"date": date, "permno": permno, "gap": gap, "runup": runup,
               "entry_halfspread": entry_hs}
        for h in horizons:
            seg = cc[pos + 1:pos + 1 + h]
            car_close, n_days = _sum_available(seg)
            car_open = car_close + (0.0 if np.isnan(intra) else float(intra))

            # exit spread: the last day the position actually existed
            exit_pos = pos + n_days if n_days else pos
            exit_hs = hs[exit_pos] if exit_pos < len(hs) and not np.isnan(hs[exit_pos]) else np.nan
            rt = np.nansum([entry_hs, exit_hs])          # one side in, one side out
            row[f"close_{h}"] = car_close
            row[f"open_{h}"] = car_open
            row[f"close_{h}_net"] = car_close - rt
            row[f"open_{h}_net"] = car_open - rt
            row[f"n_days_{h}"] = n_days
        rows.append(row)
    return pd.DataFrame(rows)
