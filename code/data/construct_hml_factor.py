#!/usr/bin/env python3
"""
Construct daily HML (High-Minus-Low) value factor returns for the Korean equity market.

Follows the standard Fama-French (1993) June-rebalancing convention:
  - Book equity (BE) from FnGuide IFRS consolidated annual financials
  - Market equity (ME) at end of December
  - B/M = BE(fiscal year t-1) / ME(December t-1)
  - Portfolios formed at end of June year t, held through June year t+1
  - HML = equal-weighted High B/M - equal-weighted Low B/M (median split)

Degrees of freedom
------------------
MEASURED        book equity and market equity, from the FnGuide annual
                financials and the December market-cap snapshot as reported.
formation rule  CHOSEN — the June-rebalancing, median-split convention of
                Fama and French (1993), taken whole rather than assembled from
                parts, so no element of the construction was picked after
                seeing the alpha it feeds.
BE_SHEETS       CHOSEN — common stock, capital surplus, retained earnings, and
                treasury stock, the four IFRS line items that sum to book
                equity in the FnGuide consolidated export.
BE_YEARS        CHOSEN — derived from the sample window in shared_utils.config:
                the fiscal years its formation dates need under a June
                rebalance with a one-year BE lag. Not a window choice in its
                own right; it follows from the sample.
MIN_STOCKS_PER_HML_TAIL  CHOSEN — 10 names per tail and 20 before a formation
MIN_STOCKS_FOR_BM_SPLIT  year is split at all. Safety guardrails rather than
                tuned cutpoints, binding on no formation year the sample
                window reaches, which check_hml_floors_do_not_bind reads off
                the factor series' coverage. The factor feeds the factor-
                neutrality alpha check alone (subsec:appendix_j_factor), not
                any headline result.

BASELINE: Fama-French (1993) HML as the value control | obvious alternative: a
referee asks whether a Korean value factor should follow the market's own
convention rather than the U.S. one | discharge: subsec:appendix_j_factor
claims only that the matched-filter alpha survives this control, which is
bounded by its own evidence. A sharper control could overturn it, and the
manuscript claims no more than the arm it ran.

Output: results/hml_factor_daily.csv (columns: date, HML)
"""

import pandas as pd

from data._fnguide import load_fnguide_sheet
from data._market import load_krx_panel
from shared_utils import END_DATE_MAIN, START_DATE_MAIN, print_header, save_csv
from paths import (
    DAILY_DATA_ALL_INVESTORS,
    FINANCIALS_FILE as _FINANCIALS_FILE,
    HML_FACTOR_FILE,
)

DAILY_DATA_FILE = str(DAILY_DATA_ALL_INVESTORS)
OUTPUT_FILE = str(HML_FACTOR_FILE)

# Book equity component sheets in fnguide_financials_annual_20260219.xlsx
BE_SHEETS = ['보통주자본금', '자본잉여금', '이익잉여금', '자기주식']
FINANCIALS_FILE = str(_FINANCIALS_FILE)

# Formation years the sample window needs: January-June of its first year holds
# the portfolio formed the June before, and July-December of its last year the
# one formed that June. Formation year t reads BE(t-1) and ME(December t-1).
FIRST_SAMPLE_YEAR = pd.Timestamp(START_DATE_MAIN).year
FORMATION_YEARS = list(range(FIRST_SAMPLE_YEAR - 1,
                             pd.Timestamp(END_DATE_MAIN).year + 1))
BE_YEARS = [year - 1 for year in FORMATION_YEARS]  # fiscal year-end Dec

# Per-tail floor for daily HML. Distinct from the panel-wide
# MIN_STOCKS_PER_DAY_DEFAULT used in Fama-MacBeth aggregation: that
# threshold guards a full cross-section, this one guards a single tail of
# a median split, which is naturally thinner.
#
# Justification: both floors are safety
# guardrails, not tuned cutpoints. A median-split tail return is
# statistically meaningless below a handful of names, and a formation year
# with too few valid BE/ME pairs cannot define a median split at all. Neither
# binds on the sample window, which check_hml_floors_do_not_bind reads off the
# factor series' coverage, and the HML factor here
# feeds only the factor-neutrality alpha robustness check (subsec:appendix_j_factor,
# Fama-French 1993 median-split convention), not any headline result.
MIN_STOCKS_PER_HML_TAIL = 10

# Minimum cross-section size to attempt a B/M median split at a formation
# year. A formation year with fewer valid BE/ME pairs is skipped rather than
# split — coarser than MIN_STOCKS_PER_HML_TAIL, which guards each tail after
# the split.
MIN_STOCKS_FOR_BM_SPLIT = 20


def load_book_equity():
    """
    Load annual book equity from FnGuide IFRS consolidated financials.
    BE = 보통주자본금 + 자본잉여금 + 이익잉여금 + 자기주식
    (자기주식 is already stored as negative values in FnGuide)
    Returns: dict mapping (ticker, year) -> BE in KRW
    """
    print("Loading book equity components from FnGuide...")

    components = {}
    for sheet in BE_SHEETS:
        df = load_fnguide_sheet(FINANCIALS_FILE, sheet)
        components[sheet] = df
        print(f"  {sheet}: {df.shape[0]} years, {df.shape[1]} stocks")

    # Sum components to get book equity
    be_dict = {}
    for year in BE_YEARS:
        # Find the row closest to Dec 31 of this year
        values = {}
        for sheet, df in components.items():
            mask = (df.index >= pd.Timestamp(f'{year}-12-01')) & (df.index <= pd.Timestamp(f'{year+1}-02-28'))
            if mask.sum() == 0:
                print(f"  WARNING: No data for {sheet} in {year}")
                continue
            row = df.loc[mask].iloc[0]
            values[sheet] = row

        if len(values) != len(BE_SHEETS):
            print(f"  Skipping year {year}: missing components")
            continue

        # Compute BE for each ticker. Use vectorised pd.to_numeric so any
        # non-numeric cells become NaN and propagate through the sum, rather
        # than silently swallowing per-ticker conversion errors.
        be_components = pd.DataFrame(
            {sheet: pd.to_numeric(values[sheet], errors='coerce')
             for sheet in BE_SHEETS}
        )
        n_nonnumeric = int((be_components.isna()
                            & values[BE_SHEETS[0]].notna()).any(axis=1).sum())
        be_series = be_components.sum(axis=1, skipna=False)
        be_series = be_series[be_series.notna() & (be_series > 0)]

        for ticker, be in be_series.items():
            be_dict[(ticker, year)] = be

        msg = f"  Year {year}: {len(be_series)} stocks with valid BE > 0"
        if n_nonnumeric:
            msg += f" ({n_nonnumeric} dropped for non-numeric components)"
        print(msg)

    return be_dict


def load_december_market_cap():
    """
    Load end-of-December market equity for the BE_YEARS.
    - Years inside the sample window: from daily data panel (column M, in KRW)
    - Years before it: from marcap (market_cap == marcap Marcap, in KRW)
    Both legs are the same marcap source. Returns: dict mapping (ticker, year) -> ME in KRW
    """
    me_dict = {}

    # --- December of each BE year inside the sample window, from daily data ---
    panel_years = [year for year in BE_YEARS if year >= FIRST_SAMPLE_YEAR]
    print(f"Loading December market caps from daily data {panel_years}...")
    df_daily = pd.read_parquet(DAILY_DATA_FILE, columns=['date', 'ticker', 'M'])
    df_daily['date'] = pd.to_datetime(df_daily['date'])

    for year in panel_years:
        dec_mask = (df_daily['date'].dt.year == year) & (df_daily['date'].dt.month == 12)
        dec_data = df_daily[dec_mask]
        if len(dec_data) == 0:
            print(f"  WARNING: No December data for {year}")
            continue

        last_dec_date = dec_data['date'].max()
        last_day = dec_data[dec_data['date'] == last_dec_date][['ticker', 'M']].dropna()

        for _, row in last_day.iterrows():
            me_dict[(row['ticker'], year)] = row['M']

        print(f"  Dec {year} ({last_dec_date.date()}): {len(last_day)} stocks")

    # --- December of each BE year before the window, from marcap (common-stock
    # universe, KRW) ---
    # Same source and convention as the daily-panel leg above (last December
    # trading day, market_cap == marcap Marcap in KRW). Mirrors the per-year
    # loop above; no millions->KRW conversion needed.
    pre_years = [year for year in BE_YEARS if year < FIRST_SAMPLE_YEAR]
    print(f"Loading December market caps from marcap {pre_years}...")
    for year in pre_years:
        mdf = load_krx_panel(None, f'{year}-12-01', f'{year}-12-31')
        if len(mdf) == 0:
            print(f"  WARNING: No December {year} marcap data")
            continue

        last_dec_date = mdf['date'].max()
        last_day = mdf[mdf['date'] == last_dec_date][['ticker', 'market_cap']].dropna()

        for _, row in last_day.iterrows():
            me_dict[(row['ticker'], year)] = row['market_cap']

        print(f"  Dec {year} ({last_dec_date.date()}): {len(last_day)} stocks")

    return me_dict


def compute_bm_assignments(be_dict, me_dict):
    """
    Compute B/M ratios and assign stocks to High/Low groups for each formation year.
    Formation year t uses BE(t-1) / ME(Dec t-1).
    Returns: dict mapping formation_year -> {'high': set(tickers), 'low': set(tickers)}
    """
    print("\nComputing B/M ratios and portfolio assignments...")
    assignments = {}

    for formation_year in FORMATION_YEARS:
        be_year = formation_year - 1

        # Collect B/M for stocks with both BE and ME
        bm_list = []
        for (ticker, yr), be in be_dict.items():
            if yr != be_year:
                continue
            me = me_dict.get((ticker, be_year))
            if me is None or me <= 0:
                continue
            bm_list.append((ticker, be / me))

        if len(bm_list) < MIN_STOCKS_FOR_BM_SPLIT:
            print(f"  Formation {formation_year}: only {len(bm_list)} stocks, skipping")
            continue

        bm_df = pd.DataFrame(bm_list, columns=['ticker', 'bm'])
        median_bm = bm_df['bm'].median()

        high_tickers = set(bm_df[bm_df['bm'] >= median_bm]['ticker'])
        low_tickers = set(bm_df[bm_df['bm'] < median_bm]['ticker'])

        assignments[formation_year] = {'high': high_tickers, 'low': low_tickers}
        print(f"  Formation {formation_year} (BE/ME from {be_year}): "
              f"{len(high_tickers)} High, {len(low_tickers)} Low, "
              f"median B/M = {median_bm:.3f}")

    return assignments


def compute_daily_hml(assignments):
    """
    Compute daily HML factor returns.
    For each trading day, determine which formation year's assignments apply
    (Jan-Jun: previous formation, Jul-Dec: current formation),
    then compute equal-weighted High - Low return.
    """
    print("\nComputing daily HML returns...")

    df_daily = pd.read_parquet(DAILY_DATA_FILE, columns=['date', 'ticker', 'R_t1'])
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    df_daily = df_daily.dropna(subset=['R_t1'])

    dates = sorted(df_daily['date'].unique())
    hml_records = []

    for date in dates:
        dt = pd.Timestamp(date)

        # Determine formation year: Jul-Dec year t uses formation t, Jan-Jun year t uses formation t-1
        if dt.month >= 7:
            formation_year = dt.year
        else:
            formation_year = dt.year - 1

        if formation_year not in assignments:
            # Formation year skipped for too few stocks: no HML on this date.
            continue

        high_tickers = assignments[formation_year]['high']
        low_tickers = assignments[formation_year]['low']

        day_data = df_daily[df_daily['date'] == date]

        high_rets = day_data[day_data['ticker'].isin(high_tickers)]['R_t1']
        low_rets = day_data[day_data['ticker'].isin(low_tickers)]['R_t1']

        if len(high_rets) < MIN_STOCKS_PER_HML_TAIL or len(low_rets) < MIN_STOCKS_PER_HML_TAIL:
            continue

        hml = high_rets.mean() - low_rets.mean()
        hml_records.append({'date': date, 'HML': hml})

    hml_df = pd.DataFrame(hml_records)
    print(f"  Generated HML for {len(hml_df)} trading days")
    print(f"  Mean HML: {hml_df['HML'].mean()*100:.4f}%")
    print(f"  Std HML:  {hml_df['HML'].std()*100:.4f}%")
    print(f"  Date range: {hml_df['date'].min()} to {hml_df['date'].max()}")

    return hml_df


def main():
    print_header("HML FACTOR CONSTRUCTION")

    be_dict = load_book_equity()
    me_dict = load_december_market_cap()
    assignments = compute_bm_assignments(be_dict, me_dict)
    hml_df = compute_daily_hml(assignments)

    # Save
    save_csv(hml_df, OUTPUT_FILE, index=False)
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
