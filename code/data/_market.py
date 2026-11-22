"""Shared Korean market-data loader: KRX prints under FnGuide's adjusted close.

Reads the marcap dataset's daily quantity panel and puts FnGuide's total-return
adjusted close on it, so the full (2020+) and pre-period (2015-2019) pipelines
call one loader rather than repeating the paths and the join.

Two prices leave this module, because two different questions are asked of a
price and only one of them is a return.

``close`` is FnGuide's total-return adjusted close (item ``S410007700``,
``adj_close_tr`` in ``data._fnguide_price``): capital changes removed and
cash dividends reinvested at the ex-date price. It is the only column the
forward returns are computed from, so a dividend is a reinvestment rather than
a price drop the return records as a loss.

``close_raw`` is the closing price as KRX published it. ``market_cap`` is that
print times the share count, and ``amount`` and ``volume`` are the session's
realised value and quantity, so a ratio taken between any of those and a price
needs the print itself: a back-adjusted series would divide each stock by its
own event history instead. ``vwap_formation_robustness`` and the share-space
block of ``flow_persistence_decomposition`` are the two call sites, and both
now read Korea on the ``close_raw`` basis.

``market_cap``, ``amount`` and ``volume`` are marcap's ``Marcap``, ``Amount``
and ``Volume`` as published; no adjustment is applied to any of them.

Which rows are in the panel is decided by the point-in-time common-stock
universe ``data.kr_universe`` builds from the same marcap snapshots, with the
security type read off the sessions the FnGuide export prices. Every row the
universe admits is therefore one the export prices.
"""

from pathlib import Path

import pandas as pd

from data._fnguide_price import load_price_panel
from paths import KR_UNIVERSE_PANEL, MARCAP_DIR

# marcap's per-day KRX snapshot, restricted to the keys and the four published
# quantities the panel carries. Open / high / low are not read: nothing
# downstream forms an intraday range.
_MARCAP_COLS = ("Date", "Code", "Close", "Volume", "Amount", "Marcap")


def _common_stock_spells(panel_path):
    """Point-in-time common-stock membership, as ``(code, first_date, last_date)``.

    Built by ``code/data/kr_universe.py`` from the marcap dataset; ``make data``
    refreshes it before either panel is assembled.
    """
    panel_path = Path(panel_path)
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} not found — build it with "
            f"`python code/data/kr_universe.py` (or `make data-universe`), "
            f"which classifies the marcap dataset into point-in-time "
            f"security-type spells.")
    panel = pd.read_parquet(panel_path)
    panel = panel[panel["kind"] == "common"][["code", "first_date", "last_date"]].copy()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    for col in ("first_date", "last_date"):
        panel[col] = pd.to_datetime(panel[col])
    return panel


def load_krx_panel(tickers, start_date, end_date):
    """marcap's daily quantity panel over the point-in-time common-stock universe.

    Returns ``date, ticker, close_raw, volume, amount, market_cap``, every
    column as KRX published it. Zero-volume sessions are dropped, so each
    remaining row is a session with a realised print. ``tickers`` is an
    optional ``"A"``-prefixed whitelist; ``None`` returns the whole universe.
    """
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    marcap_dir = Path(MARCAP_DIR)

    frames = []
    for year in range(start.year, end.year + 1):
        year_file = marcap_dir / f"marcap-{year}.parquet"
        if not year_file.exists():
            raise FileNotFoundError(f"marcap year file missing: {year_file}")
        frames.append(pd.read_parquet(year_file, columns=list(_MARCAP_COLS)))
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= start) & (df["Date"] <= end) & (df["Volume"] > 0)]
    df["Code"] = df["Code"].astype(str).str.zfill(6)

    spells = _common_stock_spells(KR_UNIVERSE_PANEL)
    df = df.merge(spells, left_on="Code", right_on="code", how="inner")
    df = df[(df["Date"] >= df["first_date"]) & (df["Date"] <= df["last_date"])]

    out = pd.DataFrame({
        "date": df["Date"].values,
        "ticker": ("A" + df["Code"]).values,
        "close_raw": df["Close"].astype("float64").values,
        "volume": df["Volume"].astype("float64").values,
        "amount": df["Amount"].astype("float64").values,
        "market_cap": df["Marcap"].astype("float64").values,
    })
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)
    if tickers is not None:
        out = out[out["ticker"].isin(set(tickers))].reset_index(drop=True)
    return out


def _attach_fnguide_close(df):
    """Attach FnGuide's total-return close as ``close``.

    Joins on the 6-digit code (marcap keys ``"A005930"``, FnGuide ``"005930"``)
    and on the date. The universe admits only sessions the export prices, so
    every row must find its close; a row that does not means the universe was
    built from another pull of the export than the one cached here.
    """
    px = load_price_panel()[['date', 'ticker', 'adj_close_tr', 'segment']]
    out = df.copy()
    out['_t6'] = out['ticker'].str.lstrip('A')
    out = out.merge(px, left_on=['date', '_t6'], right_on=['date', 'ticker'],
                    how='left', suffixes=('', '_fg'))

    # A 6-digit code is reissued once its first occupant delists, and FnGuide
    # keys a series by code, so one column can carry two companies. ``segment``
    # numbers the spells; a window containing a handover would let the
    # groupby-ticker forward returns difference across it. Neither window does,
    # and this is the assertion that keeps that true if a window moves.
    spells = out.dropna(subset=['segment']).groupby('ticker')['segment'].nunique()
    straddling = sorted(spells[spells > 1].index)
    if straddling:
        raise ValueError(
            f"{len(straddling)} tickers span a FnGuide code re-issue inside this "
            f"window ({straddling[:5]}...): forward returns grouped on ticker "
            f"alone would difference one company's price against another's.")

    unpriced = out['adj_close_tr'].isna()
    if unpriced.any():
        raise RuntimeError(
            f"FnGuide 수정주가 does not serve {int(unpriced.sum()):,} of the "
            f"{len(out):,} universe rows in this window, across "
            f"{out.loc[unpriced, '_t6'].nunique()} codes "
            f"({sorted(out.loc[unpriced, '_t6'].unique())[:5]}). The universe "
            f"admits only sessions the export prices, so it was built from "
            f"another pull; rebuild it with `make data-universe`.")

    out = out.rename(columns={'adj_close_tr': 'close'})
    return out.drop(columns=['_t6', 'ticker_fg', 'segment']).reset_index(drop=True)


def load_market_data(tickers, start_date, end_date):
    return _attach_fnguide_close(load_krx_panel(tickers, start_date, end_date))
