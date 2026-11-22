"""FnGuide adjusted-price loader — 수정주가 export → tidy parquet.

``raw/fnguide_price_adjclose_20260813.xlsx`` is a 220 MB four-sheet DataGuide
export carrying both adjusted-price conventions FnGuide publishes, KOSPI and
KOSDAQ split into separate sheets:

| Sheet | Item code | Column written |
|---|---|---|
| ``수정주가_KOSPI`` / ``수정주가_KOSDAQ`` | ``S410000700`` | ``adj_close_pr`` — price return, capital changes only |
| ``수정주가(현금배당포함)__KOSPI`` / ``수정주가(현금배당포함)_KOSDAQ`` | ``S410007700`` | ``adj_close_tr`` — total return, cash dividends reinvested |

(The doubled underscore in ``수정주가(현금배당포함)__KOSPI`` is the vendor's,
not a typo here.)

Both conventions cover the same 3,590 tickers over 2005-01-03 – 2026-08-12 (the
columns that carry a price in the window: 1,025 KOSPI, 2,551 KOSDAQ, 14 in both), and
the export was pulled with the "all codes" (전체 / 상폐 포함) filter, so delisted
names carry prices through their delisting date. It is the vendor's own
series, and the reference any independently built adjusted Korean close is
measured against rather than mixed with.

The xlsx is read with python-calamine (~4 s a sheet); melting the four wide
sheets to long still costs ~50 s and ~4.4 GB of peak RSS, so it is done once
here and every consumer reads the parquet, which is built on first use and
cached under ``results/_cache/``.

A code is not a company. KRX reissues a 6-digit code once its first occupant is
delisted, and FnGuide keys a series by code, so both occupants arrive in one
column separated by a NaN gap — 59 codes in this export. ``segment`` numbers
each code's listing spells so the handover cannot be differenced across; see
``_mark_segments``. **Difference within ``['ticker', 'segment']``, never within
``ticker`` alone**, or one bar of the result compares two different companies.

Market transfers are the one structural wrinkle. 14 tickers (신세계푸드, KTF,
신세계건설, …) moved KOSDAQ → KOSPI and so appear as a column in *both* market
sheets. DataGuide serves each one its **whole** history under both sheets, at
identical prices — the sheet says which markets a name ever traded on, not which
one it was on during a given session — so these carry ``market == 'BOTH'`` and
their duplicate rows are collapsed. That the two copies agree is checked, not
assumed: a (date, ticker) served by both sheets at *different* prices would be a
contradiction in the vendor export and raises. Point-in-time market membership
is not available from this file at all.

Degrees of freedom
------------------
CHOSEN    the two item codes, ``S410000700`` (price return) and ``S410007700``
          (total return). Both are the vendor's own conventions rather than a
          construction here, and the panels form their returns on the second
          (subsec:data). Each is asserted against its sheet's header, so a
          re-pull that swaps an item is caught at build time rather than
          benchmarked against the wrong series downstream.
CHOSEN    the segment rule: a listing spell ends where the delisting calendar
          records a genuine delisting inside a gap in that code's prices.
          Neither half is a tuned threshold. A delisting with no gap is a code
          that went on trading, and a gap with no delisting is a trading halt,
          whose resumption return belongs to the same company.
MEASURED  every price, as the vendor serves it. Nothing here re-derives one.
FITTED    nothing.

BASELINE: FnGuide's own adjusted closes as the Korean price basis | obvious
          alternative: a referee says the exchange's published prices, back-
          adjusted from its own event record, are the primary source and a
          vendor's series is a convenience | discharge: subsec:data, which
          states the convention the returns are formed on, and
          sec:appendix_data, which names the export.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import python_calamine

# The export, its parsed cache, and the delisting calendar that separates a
# reissued code's two occupants. All three are resolved at use rather than at
# import: a caller that only wants the parsed panel must not need the 220 MB
# export on disk, and a checker importing this module for one pure function
# must not need either.


def _raw_path() -> Path:
    from paths import FNGUIDE_ADJCLOSE_FILE
    return Path(FNGUIDE_ADJCLOSE_FILE)


def _cache_path() -> Path:
    from paths import FNGUIDE_PRICE_CACHE
    return Path(FNGUIDE_PRICE_CACHE)


def _calendar_path() -> Path:
    from paths import KR_DELISTING_CALENDAR
    return Path(KR_DELISTING_CALENDAR)


# Sheet → (value column, market). Item codes are asserted against row 12 of each
# sheet's header so a re-pull that swaps an item is caught at build time rather
# than silently benchmarked against the wrong series.
SHEETS = {
    '수정주가_KOSPI':                 ('adj_close_pr', 'KOSPI',  'S410000700'),
    '수정주가_KOSDAQ':                ('adj_close_pr', 'KOSDAQ', 'S410000700'),
    '수정주가(현금배당포함)__KOSPI':  ('adj_close_tr', 'KOSPI',  'S410007700'),
    '수정주가(현금배당포함)_KOSDAQ':  ('adj_close_tr', 'KOSDAQ', 'S410007700'),
}

# DataGuide's fixed export layout: metadata rows 1-14, data from row 15.
# Row 9 is the ticker code (A-prefixed), row 12 the FnGuide item code.
#
# These indices are into ``CalamineWorkbook.to_python()``, which trims a leading
# blank row. That is safe here only because this module reads one file and the
# Korean-locale exports open on a non-blank ``Refresh`` row. The English-locale
# ``fnguide_investor_*`` exports do start blank, so the same indices would land
# one row off on those — see README.md, "Common File Structure". The item-code
# assertion below is what would catch it.
_CODE_ROW = 9
_ITEM_ROW = 12
_FIRST_DATA_ROW = 15


def _read_sheet(rows: list[list], sheet: str, value_col: str,
                market: str, item_code: str) -> pd.DataFrame:
    """Turn one DataGuide sheet's cells into a long (date, ticker, market, value) frame."""
    codes = rows[_CODE_ROW - 1][1:]
    items = rows[_ITEM_ROW - 1][1:]

    # Trailing blanks are '' from calamine, which is falsy — the ticker block ends
    # at the first empty code cell.
    n_cols = sum(1 for c in codes if c)
    if n_cols == 0:
        raise ValueError(f"{sheet}: no ticker codes on row {_CODE_ROW}")
    bad = {i for i in items[:n_cols] if i != item_code}
    if bad:
        raise ValueError(
            f"{sheet}: expected item code {item_code} on row {_ITEM_ROW}, "
            f"found {sorted(bad)} — the export's sheet/item mapping changed"
        )

    # 'A005930' → '005930': the repo keys on 6-digit zero-padded codes throughout.
    tickers = [c[1:] for c in codes[:n_cols]]
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"{sheet}: duplicate ticker columns on row {_CODE_ROW}")

    body = [r for r in rows[_FIRST_DATA_ROW - 1:] if r[0] != '']
    dates = [r[0] for r in body]

    # calamine writes a blank cell as '', a genuine price as float. Map only ''
    # to NaN so any *other* non-numeric cell raises in the astype below rather
    # than being coerced away — a string in a price column is a broken export,
    # not a missing observation.
    cells = np.array([r[1:n_cols + 1] for r in body], dtype=object)
    cells[cells == ''] = np.nan
    wide = pd.DataFrame(cells.astype('float64'),
                        index=pd.to_datetime(dates), columns=tickers)
    long = (wide.reset_index(names='date')
                .melt(id_vars='date', var_name='ticker', value_name=value_col)
                .dropna(subset=[value_col]))
    long['market'] = market
    return long


def _mark_segments(panel: pd.DataFrame, calendar_path: Path | None = None) -> pd.DataFrame:
    """Number each code's listing spells in a ``segment`` column.

    A 6-digit KRX code freed by a delisting is reissued, and FnGuide serves one
    column per *code*, so both occupants arrive as one series with a NaN gap
    between them. Nothing in the column marks the handover: a consumer that
    drops NaNs and differences gets one return comparing two different
    companies, and those returns are enormous rather than merely wrong.

    The break is where the calendar records a **genuine delisting inside a gap**
    in that code's prices. Both halves are needed and neither is a tuned
    threshold: a delisting with no gap means the code kept trading (a holdco
    conversion or market transfer, ``is_genuine == 'N'``), and a gap with no
    delisting is a trading halt, where the resumption return is real and belongs
    to the same company. On the 2026-08-13 export the rule fires on 59 codes and
    no ``is_genuine == 'N'`` row falls inside a gap at all, so the genuineness
    filter never has to arbitrate.

    Consumers difference within ``['ticker', 'segment']``, never within
    ``ticker`` alone.
    """
    calendar_path = Path(calendar_path) if calendar_path else _calendar_path()
    if not calendar_path.exists():
        raise FileNotFoundError(
            f"{calendar_path} not found — the segment column needs the KRX "
            f"delisting calendar to tell a reissued code from a halted one; it "
            f"is committed under data/registry/, so restore it from git"
        )
    cal = pd.read_csv(calendar_path, dtype={'ticker': str})
    genuine = cal[cal['is_genuine'] == 'Y']
    events = pd.DataFrame({
        'ticker': genuine['ticker'].str.zfill(6),
        'event': pd.to_datetime(genuine['delisting_date']),
    })

    # Count the gap in *sessions*, not calendar days, so a weekend or a holiday
    # run is not mistaken for one.
    sessions = pd.Index(panel['date'].drop_duplicates().sort_values())
    sess_no = panel['date'].map(pd.Series(range(len(sessions)), index=sessions))
    prev_sess = sess_no.groupby(panel['ticker'], sort=False).shift(1)
    prev_date = panel['date'].groupby(panel['ticker'], sort=False).shift(1)
    gapped = (sess_no - prev_sess) > 1

    # Each gap is a candidate; it becomes a break only if a genuine delisting for
    # that code falls between the session before it and the session after it.
    cand = pd.DataFrame({'ticker': panel.loc[gapped, 'ticker'],
                         'date': panel.loc[gapped, 'date'],
                         'prev': prev_date[gapped]})
    hit = cand.merge(events, on='ticker')
    hit = hit[hit['event'].between(hit['prev'], hit['date'])]

    key = pd.MultiIndex.from_arrays([panel['ticker'], panel['date']])
    is_break = pd.Series(key.isin(pd.MultiIndex.from_frame(hit[['ticker', 'date']])),
                         index=panel.index)
    panel['segment'] = (is_break.groupby(panel['ticker'], sort=False)
                                .cumsum().astype('int16'))

    print(f"[price_cache] {int(is_break.sum())} reissued-code handovers across "
          f"{panel.loc[is_break, 'ticker'].nunique()} codes → segment > 0")
    return panel


def build_price_cache(raw_path: Path | None = None,
                      out_path: Path | None = None) -> pd.DataFrame:
    """Parse the adjusted-price export into ``results/_cache/fnguide_price.parquet``.

    Output columns: ``date, ticker, market, adj_close_pr, adj_close_tr,
    segment`` — one row per (session, ticker) the vendor served a price for,
    both conventions on the same row, and the listing spell that (ticker, date)
    belongs to.
    """
    raw_path = Path(raw_path) if raw_path else _raw_path()
    out_path = Path(out_path) if out_path else _cache_path()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found — re-pull 수정주가 / "
            f"수정주가(현금배당포함) from DataGuide with term_start=20050101; a "
            f"re-pull lands under its own date, which paths.py names"
        )

    wb = python_calamine.CalamineWorkbook.from_path(raw_path)
    missing = set(SHEETS) - set(wb.sheet_names)
    if missing:
        raise ValueError(f"{raw_path.name}: missing sheets {sorted(missing)}")

    per_convention: dict[str, list[pd.DataFrame]] = {}
    for sheet, (value_col, market, item_code) in SHEETS.items():
        rows = wb.get_sheet_by_name(sheet).to_python()
        long = _read_sheet(rows, sheet, value_col, market, item_code)
        print(f"[price_cache] {sheet}: {len(long):,} obs, "
              f"{long.ticker.nunique():,} tickers, "
              f"{long.date.min():%Y-%m-%d}–{long.date.max():%Y-%m-%d}")
        per_convention.setdefault(value_col, []).append(long)

    frames = {}
    for value_col, parts in per_convention.items():
        panel = pd.concat(parts, ignore_index=True)
        dup = panel.duplicated(subset=['date', 'ticker'], keep=False)
        if dup.any():
            clash = panel[dup].groupby(['date', 'ticker'])[value_col].nunique()
            clash = clash[clash > 1]
            if len(clash):
                raise ValueError(
                    f"{value_col}: {len(clash):,} (date, ticker) pairs served by both "
                    f"market sheets at conflicting prices — "
                    f"e.g. {clash.head().to_dict()}"
                )
            both = sorted(panel.loc[dup, 'ticker'].unique())
            print(f"[price_cache] {value_col}: {len(both)} transferred tickers served "
                  f"identically by both market sheets ({dup.sum():,} rows) → market='BOTH'")
            panel.loc[panel.ticker.isin(both), 'market'] = 'BOTH'
            panel = panel.drop_duplicates(subset=['date', 'ticker'])
        frames[value_col] = panel

    pr, tr = frames['adj_close_pr'], frames['adj_close_tr']
    panel = pr.merge(tr, on=['date', 'ticker'], how='outer', suffixes=('', '_tr_market'))
    both = panel.market.notna() & panel.market_tr_market.notna()
    if (panel.market[both] != panel.market_tr_market[both]).any():
        raise ValueError(
            "the two conventions disagree on which market a (date, ticker) traded on"
        )
    panel['market'] = panel.market.fillna(panel.market_tr_market)
    panel = (panel.drop(columns='market_tr_market')
                  .sort_values(['ticker', 'date'])
                  .reset_index(drop=True))
    panel = _mark_segments(panel)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)

    only_pr = panel.adj_close_tr.isna().sum()
    only_tr = panel.adj_close_pr.isna().sum()
    print(f"[price_cache] wrote {out_path} — {len(panel):,} rows, "
          f"{panel.ticker.nunique():,} tickers, "
          f"{panel.date.min():%Y-%m-%d}–{panel.date.max():%Y-%m-%d}; "
          f"{only_pr:,} rows price-return only, {only_tr:,} total-return only")
    return panel


def load_price_panel(path: Path | None = None) -> pd.DataFrame:
    """Load the cached FnGuide adjusted-price panel, building it if absent."""
    path = Path(path) if path else _cache_path()
    if not path.exists():
        return build_price_cache(out_path=path)
    return pd.read_parquet(path)


if __name__ == '__main__':
    build_price_cache()
