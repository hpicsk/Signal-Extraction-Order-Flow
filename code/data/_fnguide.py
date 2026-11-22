"""FnGuide DataGuide sheet readers for the Korean investor-flow panels.

This module holds the three readers the Korean pipelines call, so the panels
are built from the vendor's sheets inside this repository rather than through
a loader kept beside the data.

Every DataGuide export shares a 14-row metadata header, with symbol codes on
row 9, symbol names on row 10 and item codes on row 12. Data start at row 15,
column A is the date, and each further column is one ticker. Every read goes
through ``engine="calamine"``: on exports this size the default openpyxl reader
costs about ten times the time and several times the memory.

Two flow panels are read. ``load_investor_flow`` returns the three KRX
categories (기관 / 개인 / 외국인), with foreign flow defined as 등록외국인 alone
unless ``foreign_definition='외국인계'`` adds 기타외국인, which is null before
2003-12-01 and then contributes zero. ``load_subinvestor_flow`` returns the
eight sub-types KRX discloses under the 기관계 aggregate. All values are KRW.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Literal

import pandas as pd


# --------------------------------------------------------------------------
# Sheet I/O
# --------------------------------------------------------------------------

def load_fnguide_sheet(
    filepath: str,
    sheet_name: str,
    *,
    return_names: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, str]]:
    """Load a DataGuide sheet as a date-indexed wide frame of ticker columns.

    Values are left as read. With ``return_names`` a ``{ticker: name}`` map
    drawn from row 10 of the header is returned as well.
    """
    meta = pd.read_excel(filepath, sheet_name=sheet_name, header=None, nrows=14,
                         engine="calamine")
    tickers = list(meta.iloc[8, 1:].values)

    df = pd.read_excel(filepath, sheet_name=sheet_name, header=None, skiprows=14,
                       engine="calamine")
    df.columns = ['date'] + tickers
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()

    if not return_names:
        return df

    names = meta.iloc[9, 1:].values
    name_map = {
        code: name
        for code, name in zip(tickers, names)
        if pd.notna(code) and pd.notna(name)
    }
    return df, name_map


def melt_fnguide_wide(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert a wide (date x ticker) frame to long (date, ticker, value)."""
    long = df.reset_index().melt(id_vars='date', var_name='ticker', value_name=value_name)
    long = long.dropna(subset=[value_name])
    return long


def _cache_path(name: str, cache_dir) -> str:
    """Resolve a cache filename under *cache_dir*, creating the directory."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / name)


def _cache_is_fresh(cache: str, sources: Iterable[str | os.PathLike]) -> bool:
    """True iff *cache* exists and is newer than every source file.

    A missing source makes the cache stale, so the caller reads the source
    directly and fails on it with its own error.
    """
    if not os.path.exists(cache):
        return False
    cache_mtime = os.path.getmtime(cache)
    for s in sources:
        s = str(s)
        if not os.path.exists(s):
            return False
        if os.path.getmtime(s) > cache_mtime:
            return False
    return True


# --------------------------------------------------------------------------
# The three KRX categories
# --------------------------------------------------------------------------

# The trailing date on each export is the pull it came from. A re-pull arrives
# under its own date and these constants move to it.
_INST_BUY_FILE  = 'fnguide_investor_inst-buy_20260214.xlsx'
_INST_SELL_FILE = 'fnguide_investor_inst-sell-fin-ins-trust_20260214.xlsx'
_RETAIL_FILE    = 'fnguide_investor_pension-corp-retail_20260219.xlsx'
_FOREIGN_FILE   = 'fnguide_investor_foreign-pe_20260219.xlsx'
_BANK_FILE      = 'fnguide_investor_bank-otherfin_20260219.xlsx'
_GOVT_FILE      = 'fnguide_investor_govt-total_20260219.xlsx'

# (file, sheet) sources per investor type and side. Multiple sources are summed.
_BASE_SOURCES: dict[str, dict[str, list[tuple[str, str]]]] = {
    '기관': {
        'buy':  [(_INST_BUY_FILE,  '매수대금(기관)')],
        'sell': [(_INST_SELL_FILE, '매도대금(기관계)')],
    },
    '개인': {
        'buy':  [(_RETAIL_FILE, '매수대금(개인)')],
        'sell': [(_RETAIL_FILE, '매도대금(개인)')],
    },
}

_FOREIGN_SOURCES: dict[str, dict[str, list[tuple[str, str]]]] = {
    '등록외국인': {
        'buy':  [(_FOREIGN_FILE, '매수대금(등록외국인)')],
        'sell': [(_FOREIGN_FILE, '매도대금(등록외국인)')],
    },
    '외국인계': {
        'buy':  [(_FOREIGN_FILE, '매수대금(등록외국인)'),
                 (_FOREIGN_FILE, '매수대금(기타외국인)')],
        'sell': [(_FOREIGN_FILE, '매도대금(등록외국인)'),
                 (_FOREIGN_FILE, '매도대금(기타외국인)')],
    },
}

_KR_TO_EN = {'기관': 'institutional', '개인': 'individual', '외국인': 'foreign'}


def _load_sum_long(raw_dir: Path,
                   sources: list[tuple[str, str]],
                   start: pd.Timestamp,
                   end: pd.Timestamp,
                   value_name: str) -> pd.DataFrame:
    """Load one or more sheets, filter to [start, end], and sum across sheets."""
    parts = []
    for fname, sheet in sources:
        print(f"  loading {fname} :: {sheet}", flush=True)
        wide = load_fnguide_sheet(str(raw_dir / fname), sheet)
        wide = wide.loc[(wide.index >= start) & (wide.index <= end)]
        long = melt_fnguide_wide(wide, value_name)
        long = long[long[value_name] != 0]  # drop zero-trade rows
        parts.append(long)
    if len(parts) == 1:
        return parts[0]
    return (pd.concat(parts, ignore_index=True)
              .groupby(['date', 'ticker'], as_index=False)[value_name].sum())


def load_investor_flow(
    start_date: str,
    end_date: str,
    *,
    raw_dir: str | Path,
    investor_types: tuple[str, ...] = ('기관', '개인', '외국인'),
    foreign_definition: Literal['등록외국인', '외국인계'] = '등록외국인',
    cache_dir: str | Path | None = None,
    cache_filename: str | None = None,
) -> pd.DataFrame:
    """Return a long ``(date, ticker, investor_type, buy/sell/net)`` panel.

    ``start_date`` and ``end_date`` are inclusive ``YYYY-MM-DD`` bounds and
    ``raw_dir`` is the ``fnguide_data/raw/`` directory. With ``cache_dir`` the
    panel is cached as parquet under a name derived from the query unless
    ``cache_filename`` is given. Columns are ``date``, ``ticker``,
    ``investor_type`` (기관/개인/외국인) and ``buy_value``, ``sell_value``,
    ``net_buy_value`` in KRW, sorted by ``(date, ticker, investor_type)``.
    """
    raw_dir = Path(raw_dir).expanduser()

    valid_types = ('기관', '개인', '외국인')
    bad = [t for t in investor_types if t not in valid_types]
    if bad:
        raise ValueError(f"Unknown investor_types {bad!r}; valid: {valid_types}")

    if foreign_definition not in _FOREIGN_SOURCES:
        raise ValueError(
            f"foreign_definition must be one of {list(_FOREIGN_SOURCES)}; "
            f"got {foreign_definition!r}.")

    sources_by_type: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for t in investor_types:
        sources_by_type[t] = (_FOREIGN_SOURCES[foreign_definition]
                              if t == '외국인' else _BASE_SOURCES[t])

    all_files = sorted({
        str(raw_dir / fname)
        for sources in sources_by_type.values()
        for side in sources.values()
        for fname, _ in side
    })

    cache = None
    if cache_dir is not None:
        if cache_filename is None:
            tag = '_'.join(sorted(investor_types))
            cache_filename = (
                f"investor_flow_{start_date}_{end_date}_"
                f"{foreign_definition}_{tag}.parquet"
            )
        cache = _cache_path(cache_filename, cache_dir)
        if _cache_is_fresh(cache, all_files):
            print(f"  [cache] loading investor flow from {cache}", flush=True)
            return pd.read_parquet(cache)

    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)

    print(f"\n{'='*70}\n"
          f"LOADING INVESTOR FLOW ({start_date}..{end_date}, "
          f"foreign={foreign_definition})\n{'='*70}", flush=True)

    pieces: list[pd.DataFrame] = []
    for t, sides in sources_by_type.items():
        buy  = _load_sum_long(raw_dir, sides['buy'],  start, end, 'buy_value')
        sell = _load_sum_long(raw_dir, sides['sell'], start, end, 'sell_value')
        merged = buy.merge(sell, on=['date', 'ticker'], how='outer').fillna(0)
        merged['net_buy_value'] = merged['buy_value'] - merged['sell_value']
        merged['investor_type'] = t
        pieces.append(merged[['date', 'ticker', 'investor_type',
                              'buy_value', 'sell_value', 'net_buy_value']])

    flow = pd.concat(pieces, ignore_index=True)
    flow = (flow.sort_values(['date', 'ticker', 'investor_type'])
                .reset_index(drop=True))

    if cache:
        flow.to_parquet(cache, index=False)
        print(f"  [cache] wrote {cache}  ({len(flow):,} rows)", flush=True)

    return flow


def to_wide_net(flow: pd.DataFrame) -> pd.DataFrame:
    """Pivot long flow to wide net buying, one English-named column per type.

    Returns ``date, ticker, institutional, individual, foreign`` with missing
    combinations filled with 0.
    """
    wide = flow.pivot_table(
        index=['date', 'ticker'],
        columns='investor_type',
        values='net_buy_value',
        fill_value=0,
    ).reset_index()
    wide.columns.name = None
    return wide.rename(columns=_KR_TO_EN)


# --------------------------------------------------------------------------
# The eight sub-types under the 기관계 aggregate
# --------------------------------------------------------------------------

SUBINVESTOR_SOURCES: dict[str, tuple[str, str, str]] = {
    'pension':         (_RETAIL_FILE,   '매수대금(연기금등)', '매도대금(연기금등)'),
    'insurance':       (_INST_SELL_FILE, '매수대금(보험)',     '매도대금(보험)'),
    'investment_trust':(_INST_SELL_FILE, '매수대금(투신)',     '매도대금(투신)'),
    'pe_funds':        (_FOREIGN_FILE,  '매수대금(사모펀드)', '매도대금(사모펀드)'),
    'financial_inv':   (_INST_SELL_FILE, '매수대금(금융투자)', '매도대금(금융투자)'),
    'other_financial': (_BANK_FILE,     '매수대금(기타금융)', '매도대금(기타금융)'),
    'bank':            (_BANK_FILE,     '매수대금(은행)',     '매도대금(은행)'),
    'government':      (_GOVT_FILE,     '매수대금(국가)',     '매도대금(국가)'),
}
"""The canonical 8-type -> (file, buy sheet, sell sheet) map. A renamed or
missing sheet raises inside ``load_fnguide_sheet``; a sub-type that loads but
has no nonzero flow in the window is absent from the returned panel, so a
caller that needs all eight checks for the missing ones."""


def _load_one_sub(raw_dir: Path,
                  fname: str,
                  sheet: str,
                  start: pd.Timestamp,
                  end: pd.Timestamp,
                  value_name: str) -> pd.DataFrame:
    print(f"  loading {fname} :: {sheet}", flush=True)
    wide = load_fnguide_sheet(str(raw_dir / fname), sheet)
    wide = wide.loc[(wide.index >= start) & (wide.index <= end)]
    long = melt_fnguide_wide(wide, value_name)
    return long[long[value_name] != 0]


def load_subinvestor_flow(
    start_date: str,
    end_date: str,
    *,
    raw_dir: str | Path,
    subinvestor_types: tuple[str, ...] = tuple(SUBINVESTOR_SOURCES),
    cache_dir: str | Path | None = None,
    cache_filename: str | None = None,
) -> pd.DataFrame:
    """Return a long ``(date, ticker, subinvestor_type, buy/sell/net)`` panel.

    Bounds, ``raw_dir`` and the cache arguments are as in
    ``load_investor_flow``. ``subinvestor_types`` is a subset of
    ``SUBINVESTOR_SOURCES``, all eight by default. ``ticker`` is A-prefixed and
    ``subinvestor_type`` carries the English key.
    """
    raw_dir = Path(raw_dir).expanduser()

    unknown = [t for t in subinvestor_types if t not in SUBINVESTOR_SOURCES]
    if unknown:
        raise ValueError(
            f"Unknown subinvestor_types {unknown!r}; valid: "
            f"{list(SUBINVESTOR_SOURCES)}")

    all_files = sorted({
        str(raw_dir / SUBINVESTOR_SOURCES[t][0])
        for t in subinvestor_types
    })

    cache = None
    if cache_dir is not None:
        if cache_filename is None:
            tag = '_'.join(sorted(subinvestor_types))
            cache_filename = f"subinvestor_flow_{start_date}_{end_date}_{tag}.parquet"
        cache = _cache_path(cache_filename, cache_dir)
        if _cache_is_fresh(cache, all_files):
            print(f"  [cache] loading sub-investor flow from {cache}", flush=True)
            return pd.read_parquet(cache)

    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)

    print(f"\n{'='*70}\n"
          f"LOADING SUB-INVESTOR FLOW ({start_date}..{end_date}, "
          f"{len(subinvestor_types)} types)\n{'='*70}", flush=True)

    pieces: list[pd.DataFrame] = []
    for t in subinvestor_types:
        fname, buy_sheet, sell_sheet = SUBINVESTOR_SOURCES[t]
        buy  = _load_one_sub(raw_dir, fname, buy_sheet,  start, end, 'buy_value')
        sell = _load_one_sub(raw_dir, fname, sell_sheet, start, end, 'sell_value')
        if buy.empty and sell.empty:
            print(f"  [{t}] no nonzero flows in window; skipping", flush=True)
            continue
        merged = buy.merge(sell, on=['date', 'ticker'], how='outer').fillna(0)
        merged['net_buy_value'] = merged['buy_value'] - merged['sell_value']
        merged['subinvestor_type'] = t
        pieces.append(merged[['date', 'ticker', 'subinvestor_type',
                              'buy_value', 'sell_value', 'net_buy_value']])

    if not pieces:
        return pd.DataFrame(columns=['date', 'ticker', 'subinvestor_type',
                                     'buy_value', 'sell_value', 'net_buy_value'])

    flow = pd.concat(pieces, ignore_index=True)
    flow = (flow.sort_values(['date', 'ticker', 'subinvestor_type'])
                .reset_index(drop=True))

    if cache:
        flow.to_parquet(cache, index=False)
        print(f"  [cache] wrote {cache}  ({len(flow):,} rows)", flush=True)

    return flow


def to_wide_net_sub(flow: pd.DataFrame) -> pd.DataFrame:
    """Pivot long sub-investor flow to wide net buying, one column per sub-type."""
    wide = flow.pivot_table(
        index=['date', 'ticker'],
        columns='subinvestor_type',
        values='net_buy_value',
        fill_value=0,
    ).reset_index()
    wide.columns.name = None
    return wide
