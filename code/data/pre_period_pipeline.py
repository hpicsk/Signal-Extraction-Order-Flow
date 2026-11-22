#!/usr/bin/env python3
"""QF Paper - Pre-Period Daily Data Pipeline (2015-2019 holdout).

Builds the full multi-investor (Foreign, Institutional, Individual) panel
plus the 8-way KRX 기관계 sub-investor decomposition for the 2015-01-01 to
2019-12-31 holdout window, used to re-measure the return-free scaling
diagnostics and the return prediction on a window that does not overlap the
main sample. The scaling regimes themselves are assigned from each
class's mandate or execution convention, not from either window.

Canonical D is D_capacity_bound = D_pension + D_insurance (matches
the main panel). The KRX 기관계 aggregate is preserved as
``D_institutional_aggregate`` for the subsec:appendix_aggregate_alignment paragraph.

Investor flow is loaded via ``data._fnguide.load_investor_flow`` from the
FnGuide DataGuide xlsx exports under ``data/fnguide/raw/`` (the
same unified loader and ``foreign_definition='등록외국인'`` used for the 2020+
window — only the date range differs). Market data (volume, traded value,
market cap and the closing price) is loaded via
``data._market.load_market_data``, which reads the marcap dataset over the
point-in-time common-stock universe and puts FnGuide's total-return 수정주가 on
it as ``close``.

Investor-flow definition:
  D_institutional = 매수대금(기관계)     - 매도대금(기관계)
  D_foreign       = 매수대금(등록외국인) - 매도대금(등록외국인)
  D_individual    = 매수대금(개인)       - 매도대금(개인)

  Foreign is the registered-foreigner ("Smart Money") flow, matching the
  post-2020 main panel; 기타외국인 (Other Foreigners, unregistered) is
  excluded by ``foreign_definition='등록외국인'`` so the holdout pins the same
  foreign population the subsec:regime_assignment pre-period ordering
  claim refers to.

  ``D_institutional`` here is the raw KRX 기관계 aggregate column; it is
  aliased downstream to ``D_institutional_aggregate``, and the canonical
  ``D`` is overwritten to ``D_capacity_bound = D_pension + D_insurance``
  to match the post-2020 main panel.

Caveats the DataGuide export carries:
  - 사모펀드 (private funds, NULL pre-2008-06-23) is implicitly included in
    the 기관계 aggregate; we do not load it separately.

Degrees of freedom
------------------
MEASURED        every column the holdout panel carries, from the same two
                sources the main panel uses. Nothing is fitted, and no
                parameter crosses over from the main window.
window          CHOSEN — 2015-2019, the five years immediately preceding the
                return-prediction sample and disjoint from it, fixed by
                START_DATE_PRE / END_DATE_PRE in shared_utils.config.
loader settings CHOSEN — identical to the main pipeline in every respect
                except the date range, so a difference between the two windows
                cannot come from a difference in how they were built. The
                cache name encodes foreign_definition for the same reason.
SPARSE_SUBINVESTOR_TYPES  CHOSEN — bank and government, mirroring the main
                pipeline, so both windows drop the same sub-types on the same
                rule.

BASELINE: the same registered-foreigner definition the main panel uses |
obvious alternative: 외국인계, the 등록외국인 plus 기타외국인 total |
discharge: subsec:data, via code/data/full_daily_data_pipeline.py, which carries the
same choice for the published window, and sec:appendix_data, which names the
arm that sizes what it excludes.

Output:
  results/full_daily_data_all_investors_pre2020.parquet
  results/_cache/investor_flow_<window>_<foreign_def>_<investors>.parquet
    (loader-derived cache name, encodes foreign_definition so a definition
     change cannot silently reuse a stale cache)
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from data._fnguide import (
    SUBINVESTOR_SOURCES,
    load_investor_flow,
    load_subinvestor_flow,
    to_wide_net,
    to_wide_net_sub,
)
from data._market import load_market_data
from shared_utils import (
    END_DATE_PRE,
    START_DATE_PRE,
    add_investor_signals,
    apply_pipeline_filters,
    compute_forward_returns,
    merge_subinvestor_panel,
    set_capacity_bound_canonical,
    save_parquet,
)
from paths import (
    CACHE_DIR,
    DAILY_DATA_PRE2020,
    FNGUIDE_RAW_DIR,
)

SUBINVESTOR_TYPE_LIST = list(SUBINVESTOR_SOURCES)

# Bank (eunhaeng) and government (gukga) trade individual equities too rarely for
# stable inference; the manuscript reports them with em-dashes and excludes them
# from the contributing set (subsec:subtype_dose_response and
# the "six same-signed sub-types" partition in subsec:appendix_aggregate_alignment).
# Either may be entirely absent from a given FnGuide snapshot — gukga has zero
# non-zero stock-days in this window — in which case its D_ column is zero-filled
# so the decomposition still emits the documented em-dash row. The other six legs
# feed the aggregate and must be present; a missing one is a renamed/dropped sheet.
SPARSE_SUBINVESTOR_TYPES = ('bank', 'government')


OUTPUT_FILE = str(DAILY_DATA_PRE2020)

START_DATE = START_DATE_PRE
END_DATE = END_DATE_PRE

def main() -> pd.DataFrame:
    t0 = datetime.now()
    print(f"\n{'#'*70}\nQF PAPER - PRE-PERIOD PIPELINE (2015-2019)\n{'#'*70}\n"
          f"Start: {t0:%Y-%m-%d %H:%M:%S}\nWindow: {START_DATE} to {END_DATE}\n"
          f"Output: {OUTPUT_FILE}\n{'#'*70}\n", flush=True)

    # Step 1: investor flow (Foreign = 등록외국인, registered-only "Smart Money",
    # matching the main panel; see code/data/full_daily_data_pipeline.py).
    # No cache_filename override: let the loader derive a name that encodes
    # foreign_definition (and the date range / investor set), as the main panel
    # does. A fixed name that omits foreign_definition would let a future
    # definition change silently reuse a cache built under the other definition
    # (cache_is_fresh only checks mtime vs the source xlsx).
    flow_long = load_investor_flow(
        START_DATE, END_DATE,
        raw_dir=FNGUIDE_RAW_DIR,
        foreign_definition='등록외국인',
        cache_dir=CACHE_DIR,
    )
    flow_wide = to_wide_net(flow_long).rename(columns={
        'foreign':       'D_foreign',
        'institutional': 'D_institutional',
        'individual':    'D_individual',
    })
    print(f"Investor flow: {len(flow_wide):,} (date, ticker) rows; "
          f"{flow_wide['ticker'].nunique():,} tickers, "
          f"{flow_wide['date'].nunique():,} dates", flush=True)

    # Step 1b: sub-investor flow (8 KRX 기관계 components)
    sub_long = load_subinvestor_flow(
        START_DATE, END_DATE,
        raw_dir=FNGUIDE_RAW_DIR,
        cache_dir=CACHE_DIR,
        cache_filename='subinvestor_flow_pre2020.parquet',
    )
    sub_wide = to_wide_net_sub(sub_long)
    sub_wide = sub_wide.rename(columns={t: f'D_{t}' for t in SUBINVESTOR_TYPE_LIST
                                        if t in sub_wide.columns})
    missing = [t for t in SUBINVESTOR_TYPE_LIST if f'D_{t}' not in sub_wide.columns]
    unexpected = [t for t in missing if t not in SPARSE_SUBINVESTOR_TYPES]
    if unexpected:
        contributing = [t for t in SUBINVESTOR_TYPE_LIST
                        if t not in SPARSE_SUBINVESTOR_TYPES]
        raise ValueError(
            f"Contributing sub-investor types absent after FnGuide load: {unexpected}. "
            f"Expected all of {contributing} (Capacity-Bound = pension + insurance "
            f"depends on them). A missing contributing type means a renamed/empty "
            f"FnGuide sheet — fix the source rather than zero-filling a leg of the "
            f"aggregate. (Sparse legs {list(SPARSE_SUBINVESTOR_TYPES)} may be absent.)"
        )
    # Documented-sparse legs (bank, government) that are legitimately absent are
    # zero-filled so the decomposition still emits their em-dash rows.
    for t in missing:
        sub_wide[f'D_{t}'] = 0.0
    print(f"Sub-investor flow: {len(sub_wide):,} (date, ticker) rows; "
          f"{sub_wide['ticker'].nunique():,} tickers, "
          f"{sub_wide['date'].nunique():,} dates", flush=True)

    flow_wide = merge_subinvestor_panel(
        flow_wide, sub_wide,
        top_types=('foreign', 'institutional', 'individual'),
        sub_types=SUBINVESTOR_TYPE_LIST,
    )

    # Step 2: market data
    tickers = sorted(flow_wide['ticker'].unique())
    market = load_market_data(tickers, START_DATE, END_DATE)

    # Step 3: merge & compute signals
    df = flow_wide.merge(market, on=['date', 'ticker'], how='inner')
    df['M'] = df['market_cap']
    # V = daily traded value (KRW) from marcap's Amount field (realised
    # Sum(execution price x quantity)), matching the main pipeline and the
    # manuscript's "Daily Traded Value" definition. See the comment in
    # full_daily_data_pipeline.merge_and_calculate_variables for rationale.
    df['V'] = df['amount']
    df['turnover'] = df['V'] / df['M']

    add_investor_signals(df, ('foreign', 'institutional', 'individual'))
    add_investor_signals(df, SUBINVESTOR_TYPE_LIST)

    # Capacity-Bound = pension + insurance becomes canonical D; the KRX 기관계
    # aggregate is preserved as the D_institutional_aggregate alias.
    set_capacity_bound_canonical(df)
    df['year'] = df['date'].dt.year

    # Forward returns (only cross-sectional moments are used in scaling-validation,
    # but we keep them for parity with the main panel schema)
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    compute_forward_returns(df)

    # Filtering (mirror full_daily_data_pipeline.filter_and_clean)
    n0 = len(df)
    df = apply_pipeline_filters(df)

    print(f"\nPanel: {len(df):,} obs (from {n0:,}); {df['ticker'].nunique():,} tickers, "
          f"{df['date'].nunique():,} days", flush=True)

    # Save
    cols = ['date', 'ticker', 'year', 'close', 'close_raw', 'volume',
            'D', 'M', 'V', 'S_MC', 'S_TV', 'turnover',
            'R_t', 'R_t1', 'R_t5', 'R_t20', 'size_quintile',
            'D_foreign', 'S_MC_foreign', 'S_TV_foreign',
            'D_institutional', 'S_MC_institutional', 'S_TV_institutional',
            'D_individual', 'S_MC_individual', 'S_TV_individual',
            'D_capacity_bound', 'S_MC_capacity_bound', 'S_TV_capacity_bound',
            'D_institutional_aggregate', 'S_MC_institutional_aggregate',
            'S_TV_institutional_aggregate']
    for sub in SUBINVESTOR_TYPE_LIST:
        cols.extend([f'D_{sub}', f'S_MC_{sub}', f'S_TV_{sub}'])
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    save_parquet(df[cols], OUTPUT_FILE, index=False, announce=False)

    dt = (datetime.now() - t0).total_seconds()
    size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\n{'#'*70}\nDONE in {dt/60:.1f} min  ({size_mb:.1f} MB)\n{'#'*70}\n", flush=True)
    return df


if __name__ == '__main__':
    main()
