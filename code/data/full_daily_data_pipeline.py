#!/usr/bin/env python3
"""
QF Paper - Full Daily Data Pipeline

Loads comprehensive daily order flow data for ALL THREE investor types
(Foreign, Institutional, Individual) over the main window and merges with market data
from ``data._market.load_market_data``: marcap's published volume, traded
value, market cap and closing price over the point-in-time common-stock
universe, on FnGuide's total-return adjusted close (수정주가(현금배당포함)),
which removes splits, rights offerings and capital reductions *and* reinvests
the cash dividend.

Both sources avoid look-ahead survivorship bias, and they have to, because the
merge is an inner join and either one could impose it: marcap retains names that
traded historically and later delisted, and the FnGuide export is pulled with
the "all codes" (상폐 포함) filter, so a delisted name carries prices through its
delisting date rather than vanishing at the export date.

Main analysis uses the Capacity-Bound Institutional flow
(D_capacity_bound = D_pension + D_insurance), the matched-filter-relevant
subset of the KRX 기관계 aggregate. The aggregate result is preserved as
D_institutional_aggregate for the aggregate-alignment paragraph in
subsec:appendix_aggregate_alignment and the 8-way sub-investor decomposition in
sec:appendix_subinvestor_decomposition.

Degrees of freedom
------------------
MEASURED        every column the panel carries. Flow comes from the KRX
                disclosure as reported, market data from marcap, and neither
                is fitted or calibrated here.
MIN_MARKET_CAP  CHOSEN — 0. sec:appendix_data imposes no minimum size, and
                pinning the constant to zero keeps that decision visible in
                the pipeline rather than implicit in its absence.
                code/empirical/robustness/threshold_sensitivity.py re-runs the headline across a
                six-point size grid.
window          CHOSEN — START_DATE_MAIN to END_DATE_MAIN in shared_utils.config.
                The window was extended from 2020-2024 to 2020-2025 after the
                criteria of code/empirical/robustness/window_2025_criteria.py were
                committed.
capacity_bound  CHOSEN — pension plus insurance, the two KRX sub-types whose
                mandates bound position size against holdings. The definition
                lives in shared_utils as CAPACITY_BOUND_COMPONENTS, so no
                script can adopt a different subset, and
                sec:appendix_subinvestor_decomposition reports all eight legs
                separately.
SPARSE_SUBINVESTOR_TYPES  CHOSEN — bank and government are zero-filled when a
                snapshot omits them, so the decomposition still emits the
                em-dash row the manuscript prints. The other six legs must be
                present, and a missing one is a renamed sheet rather than a
                sparse class.

BASELINE: the registered-foreigner definition of foreign flow
(``foreign_definition='등록외국인'``) | obvious alternative: 외국인계, which the
same loader builds by adding the unregistered 기타외국인 leg, and which is what a
reader who has only seen KRX headline foreign flow will assume was used |
discharge: subsec:data, which states the restriction, and sec:appendix_data,
which sizes the excluded leg
by the gross and sign-flip shares that
``code/empirical/robustness/foreign_definition_share.py`` measures and
``check_prose_foreign_definition`` asserts.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

from data._fnguide import (
    SUBINVESTOR_SOURCES,
    load_investor_flow,
    load_subinvestor_flow,
    to_wide_net_sub,
)
from data._market import load_market_data
from paths import (
    CACHE_DIR,
    DAILY_DATA_ALL_INVESTORS,
    FNGUIDE_RAW_DIR,
)
from shared_utils import (
    END_DATE_MAIN,
    MC_SEED,
    START_DATE_MAIN,
    add_investor_signals,
    apply_pipeline_filters,
    compute_forward_returns,
    merge_subinvestor_panel,
    set_capacity_bound_canonical,
    save_parquet,
)

OUTPUT_FILE_ALL_INVESTORS = str(DAILY_DATA_ALL_INVESTORS)

# Date range
START_DATE = START_DATE_MAIN
END_DATE = END_DATE_MAIN

# Investor type mapping (Korean to English)
INVESTOR_TYPES = {
    '외국인': 'foreign',      # Foreign investors
    '기관': 'institutional',   # Domestic institutional aggregate (KRX 기관계)
    '개인': 'individual'       # Individual/retail investors
}
INVESTOR_TYPE_LIST = ['foreign', 'institutional', 'individual']

# Sub-investor decomposition of 기관계. Drives
# sec:appendix_subinvestor_decomposition and the
# definition of capacity_bound = pension + insurance (the matched-filter
# headline actor in the main paper); that subset lives in shared_utils as
# CAPACITY_BOUND_COMPONENTS, the single source of truth applied by
# set_capacity_bound_canonical.
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

# Filtering parameters
MIN_MARKET_CAP = 0  # No market cap filter

np.random.seed(MC_SEED)


def load_investor_data(start_date=START_DATE, end_date=END_DATE):
    """Load all three top-level investor types (Foreign / Institutional /
    Individual) in long format via data._fnguide.load_investor_flow."""
    print(f"\n{'='*70}\nLOADING ALL INVESTOR (Foreign, Institutional, Individual) "
          f"TRADING DATA\n{'='*70}")
    print(f"Source: {FNGUIDE_RAW_DIR} (fnguide_investor_*.xlsx)")
    print(f"Date range: {start_date} to {end_date}")

    df = load_investor_flow(
        start_date, end_date,
        raw_dir=FNGUIDE_RAW_DIR,
        investor_types=tuple(INVESTOR_TYPES.keys()),
        foreign_definition='등록외국인',  # registered foreign ("smart money") definition
        cache_dir=CACHE_DIR,
    )
    df = df.drop(columns=['buy_value', 'sell_value'])
    df['investor_type'] = df['investor_type'].map(INVESTOR_TYPES)
    df = df[['date', 'ticker', 'net_buy_value', 'investor_type']]
    df = df.sort_values(['date', 'ticker', 'investor_type']).reset_index(drop=True)

    print(f"\n{'='*70}\nAll data loaded: {len(df):,} observations")
    print(f"Unique dates: {df['date'].nunique():,}")
    print(f"Unique tickers: {df['ticker'].nunique():,}")
    print("\nObservations by investor type:")
    for inv_type in INVESTOR_TYPE_LIST:
        count = len(df[df['investor_type'] == inv_type])
        print(f"  {inv_type}: {count:,} ({100*count/len(df):.1f}%)")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}\n{'='*70}\n")
    return df


def load_subinvestor_data(start_date=START_DATE, end_date=END_DATE):
    """Load 8 KRX sub-investor types and return a wide ``(date, ticker, ...)`` frame.

    Returns columns ``date, ticker, D_pension, D_insurance, D_investment_trust,
    D_pe_funds, D_financial_inv, D_other_financial, D_bank, D_government``.
    Used by the multi-investor pipeline to populate the aggregate-alignment
    decomposition and the sec:appendix_subinvestor_decomposition 8-way
    Fama-MacBeth table.
    """
    print(f"\n{'='*70}\nLOADING SUB-INVESTOR (KRX 기관계 DECOMPOSITION)\n{'='*70}",
          flush=True)
    flow = load_subinvestor_flow(
        start_date, end_date,
        raw_dir=FNGUIDE_RAW_DIR,
        cache_dir=CACHE_DIR,
    )
    wide = to_wide_net_sub(flow)
    rename = {t: f'D_{t}' for t in SUBINVESTOR_TYPE_LIST if t in wide.columns}
    wide = wide.rename(columns=rename)
    missing = [t for t in SUBINVESTOR_TYPE_LIST if f'D_{t}' not in wide.columns]
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
        wide[f'D_{t}'] = 0.0
    print(f"  Sub-investor wide: {len(wide):,} (date, ticker) rows; "
          f"{wide['ticker'].nunique():,} tickers, "
          f"{wide['date'].nunique():,} dates", flush=True)
    return wide


def pivot_investor_data(df):
    """
    Pivot investor data from long to wide format.

    Converts from: date, ticker, investor_type, net_buy_value
    To: date, ticker, D_foreign, D_institutional, D_individual

    ``D_institutional`` at this raw-pivot stage is the KRX 기관계
    aggregate. Downstream it is aliased to ``D_institutional_aggregate``
    and the canonical ``D`` is overwritten to ``D_capacity_bound =
    D_pension + D_insurance`` (see ``merge_and_calculate_variables``).

    Args:
        df: Long-format DataFrame from load_investor_data()

    Returns:
        Wide-format DataFrame with one column per investor type
    """
    print("Pivoting investor data to wide format...")

    # Pivot table: index=(date, ticker), columns=investor_type, values=net_buy_value
    df_wide = df.pivot_table(
        index=['date', 'ticker'],
        columns='investor_type',
        values='net_buy_value',
        aggfunc='sum'  # Sum in case of duplicates
    ).reset_index()

    # Rename columns to D_<investor_type>
    rename_dict = {inv: f'D_{inv}' for inv in INVESTOR_TYPE_LIST}
    df_wide = df_wide.rename(columns=rename_dict)

    # Fill NaN with 0 (if investor didn't trade that stock on that day)
    for inv in INVESTOR_TYPE_LIST:
        col = f'D_{inv}'
        if col in df_wide.columns:
            df_wide[col] = df_wide[col].fillna(0)

    print(f"  Wide format: {len(df_wide):,} (date, ticker) combinations")

    return df_wide


def _compute_multi_investor_signals(df):
    """Populate D/S_MC/S_TV for all three top-level types, the 8 KRX
    sub-investors, the capacity-bound aggregate, and the
    institutional-aggregate alias.

    The canonical ``D`` / ``S_MC`` / ``S_TV`` are pointed at
    ``D_capacity_bound = D_pension + D_insurance`` (matched-filter actor in
    the main paper); the raw KRX 기관계 aggregate is kept under the
    explicit ``D_institutional_aggregate`` alias for the
    subsec:appendix_aggregate_alignment paragraph.
    """
    print("\nComputing signals for all investor types...")
    add_investor_signals(df, INVESTOR_TYPE_LIST)
    for inv in INVESTOR_TYPE_LIST:
        smc = df[f'S_MC_{inv}']
        print(f"  {inv}: S_MC mean={smc.mean():.6f}, std={smc.std():.6f}")
    add_investor_signals(df, SUBINVESTOR_TYPE_LIST)
    set_capacity_bound_canonical(df)


def merge_and_calculate_variables(investor_df, market_df):
    """Merge the investor and market panels and compute analysis variables.

    The merge is inner-join on ``(date, ticker)``. After merge the function
    adds ``M`` / ``V`` / ``turnover``, computes the multi-investor signal
    block, sets ``year``, and appends forward returns (``R_t``, ``R_t1``,
    ``R_t5``, ``R_t20``) via :func:`compute_forward_returns`.
    """
    print(f"\n{'='*70}\nMERGING DATA AND CALCULATING VARIABLES\n{'='*70}")
    df = pd.merge(investor_df, market_df, on=['date', 'ticker'], how='inner')
    print(f"After merge: {len(df):,} observations")
    print(f"Unique dates: {df['date'].nunique():,}")
    print(f"Unique tickers: {df['ticker'].nunique():,}")

    print("\nCalculating normalised variables...")
    df['M'] = df['market_cap']
    # V = daily traded value (KRW), the realised Sum(execution price x quantity)
    # reported in marcap's Amount field. This is the manuscript's "Daily Traded
    # Value" (Table: Variable Definitions). Using the realised value rather than
    # volume*close keeps S_TV = D/V a true participation rate, since D itself is
    # the realised net buy value (buy amount - sell amount). A KRW total is
    # invariant to any restatement of price against share count, so it is the
    # one column no adjustment convention could have moved.
    df['V'] = df['amount']
    df['turnover'] = df['V'] / df['M']

    _compute_multi_investor_signals(df)

    df['year'] = df['date'].dt.year
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    compute_forward_returns(df)

    print("\nCommon statistics:")
    print(f"  M (market cap): mean={df['M'].mean():.2e}, std={df['M'].std():.2e}")
    print(f"  V (traded value): mean={df['V'].mean():.2e}, std={df['V'].std():.2e}")
    print(f"  turnover: mean={df['turnover'].mean():.4f}, std={df['turnover'].std():.4f}")

    return df


def filter_and_clean(df):
    """Apply :func:`shared_utils.data.apply_pipeline_filters` to the merged panel
    and report the post-filter row counts / quintile breakdown.

    ``MIN_MARKET_CAP`` is honoured before delegation and is pinned to zero,
    which is how sec:appendix_data's "No minimum market capitalization is
    imposed" reaches the pipeline as a value rather than as an absence. The
    branch it guards therefore does not run on ``main``. Deleting it would
    leave a replicator unable to tell a size floor the design removed from one
    it never considered.
    """
    print(f"\n{'='*70}\nFILTERING AND CLEANING DATA\n{'='*70}")
    initial_count = len(df)
    print(f"Initial observations: {initial_count:,}")

    if MIN_MARKET_CAP > 0:
        print(f"Pre-filter: minimum market cap ({MIN_MARKET_CAP:.0e} KRW)")
        df = df[df['M'] >= MIN_MARKET_CAP]

    df = apply_pipeline_filters(df)
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)

    print(f"After pipeline filter: {len(df):,} ({100*len(df)/initial_count:.1f}%)")
    quintile_counts = df['size_quintile'].value_counts().sort_index()
    for quintile, count in quintile_counts.items():
        print(f"   {quintile}: {count:,} observations ({100*count/len(df):.1f}%)")

    print(f"\n{'='*70}")
    print(f"Final dataset: {len(df):,} observations ({100*len(df)/initial_count:.1f}% of initial)")
    print(f"Unique dates: {df['date'].nunique():,}")
    print(f"Unique tickers: {df['ticker'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Years covered: {sorted(df['year'].unique())}")
    print(f"{'='*70}\n")

    return df


def save_output(df, output_file):
    """
    Save the final multi-investor dataset to Parquet.

    Args:
        df: Final clean DataFrame
        output_file: Path to output Parquet file
    """
    print(f"\n{'='*70}")
    print("SAVING OUTPUT")
    print(f"{'='*70}")
    print(f"Output file: {output_file}")

    # Select columns in logical order
    output_cols = [
        'date', 'ticker', 'year',
        'close', 'close_raw', 'volume',
        'D', 'M', 'V',
        'S_MC', 'S_TV', 'turnover',
        'R_t', 'R_t1', 'R_t5', 'R_t20',
        'size_quintile'
    ]

    for inv_type in INVESTOR_TYPE_LIST:
        d_col = f'D_{inv_type}'
        smc_col = f'S_MC_{inv_type}'
        stv_col = f'S_TV_{inv_type}'
        if d_col in df.columns:
            output_cols.extend([d_col, smc_col, stv_col])
    # Capacity-Bound (the new canonical institutional actor) and the explicit
    # aggregate alias kept for the subsec:appendix_aggregate_alignment paragraph.
    for grp in ('capacity_bound', 'institutional_aggregate'):
        output_cols.extend([f'D_{grp}', f'S_MC_{grp}', f'S_TV_{grp}'])
    # Sub-investor decomposition (sec:appendix_subinvestor_decomposition)
    for sub in SUBINVESTOR_TYPE_LIST:
        output_cols.extend([f'D_{sub}', f'S_MC_{sub}', f'S_TV_{sub}'])

    # Filter to columns that exist
    output_cols = [c for c in output_cols if c in df.columns]

    save_parquet(df[output_cols], output_file, index=False, announce=False)

    # Get file size
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)

    print("File saved successfully!")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"Observations: {len(df):,}")
    print(f"Columns: {len(output_cols)}")
    print(f"{'='*70}\n")


def main():
    """
    Pipeline execution for ALL THREE investor types.

    This generates the dataset needed for the multi-investor return-prediction
    and robustness tests (subsec:return_prediction and sec:empirical through
    sec:robustness).
    """
    print(f"\n{'#'*70}")
    print("QF PAPER - MULTI-INVESTOR DATA PIPELINE")
    print("For multi-investor return prediction and robustness "
          "(subsec:appendix_aggregate_alignment, sec:robustness)")
    print(f"{'#'*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Investor types: {INVESTOR_TYPE_LIST}")
    print(f"Output: {OUTPUT_FILE_ALL_INVESTORS}")
    print(f"{'#'*70}\n")

    start_time = datetime.now()

    # Step 1: Load ALL investor trading data (Foreign, Institutional, Individual)
    investor_df_long = load_investor_data(START_DATE, END_DATE)

    # Step 2: Pivot to wide format (D_foreign, D_institutional, D_individual)
    investor_df_wide = pivot_investor_data(investor_df_long)

    # Step 2b: Load 8 KRX sub-investor types and outer-merge onto the wide panel.
    subinvestor_df_wide = load_subinvestor_data(START_DATE, END_DATE)
    investor_df_wide = merge_subinvestor_panel(
        investor_df_wide, subinvestor_df_wide,
        top_types=INVESTOR_TYPE_LIST, sub_types=SUBINVESTOR_TYPE_LIST,
    )

    # Step 3: Get list of tickers from investor data
    tickers = sorted(investor_df_wide['ticker'].unique())
    print(f"Tickers found in investor data: {len(tickers):,}")

    # Step 4: Load market data for these tickers
    market_df = load_market_data(tickers, START_DATE, END_DATE)

    # The merge below is an inner join, so a flow export that stops before the
    # market data do would end the panel early without an error. Both sources
    # must reach the last session the window holds.
    flow_last = investor_df_wide['date'].max()
    market_last = market_df['date'].max()
    if flow_last != market_last:
        raise ValueError(
            f"investor flow ends {flow_last.date()} and market data end "
            f"{market_last.date()}; re-pull the export that stops early "
            f"rather than building a panel that ends before {END_DATE}")

    # Step 5: Merge and calculate variables for ALL investor types
    df_merged = merge_and_calculate_variables(investor_df_wide, market_df)

    # Step 6: Filter and clean. No winsorization here — the published panel is
    # untrimmed, and outlier robustness is a separate arm
    # (code/empirical/robustness/robustness_tests_extended.py, 1%/99% within-day).
    df_final = filter_and_clean(df_merged)

    # Step 7: Save output with multi-investor columns
    save_output(df_final, OUTPUT_FILE_ALL_INVESTORS)

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'#'*70}")
    print("MULTI-INVESTOR PIPELINE COMPLETED SUCCESSFULLY")
    print(f"{'#'*70}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Final dataset: {len(df_final):,} observations")
    print(f"Output saved to: {OUTPUT_FILE_ALL_INVESTORS}")
    print("\nColumns per top-level investor type:")
    for inv_type in INVESTOR_TYPE_LIST:
        print(f"  {inv_type}: D_{inv_type}, S_MC_{inv_type}, S_TV_{inv_type}")
    print("\nCapacity-Bound + aggregate alias:")
    for grp in ('capacity_bound', 'institutional_aggregate'):
        print(f"  {grp}: D_{grp}, S_MC_{grp}, S_TV_{grp}")
    print("\nKRX sub-investors (sec:appendix_subinvestor_decomposition):")
    for sub in SUBINVESTOR_TYPE_LIST:
        print(f"  {sub}: D_{sub}, S_MC_{sub}, S_TV_{sub}")
    print(f"{'#'*70}\n")

    return df_final


if __name__ == '__main__':
    main()
