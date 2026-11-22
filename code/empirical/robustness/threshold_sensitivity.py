#!/usr/bin/env python3
"""
Market Cap Threshold Sensitivity Analysis

Verifies the headline results (Fama-MacBeth horse-race t-stats, long-short
Sharpe ratios, rank Information Coefficients) are insensitive to re-imposing a
market-cap floor. Panel-construction filtering is governed by MIN_MARKET_CAP,
which the published pipeline leaves at its no-filter setting (see
load_domestic_panel), so the panel enters as this sweep's own baseline column
and the sweep re-applies M >= threshold per stock-day on top of it.

Sweeps: 0 (no filter), 100B, 250B, 500B, 1T, 2T KRW

Degrees of freedom
------------------
MEASURED        the three headline statistics at each grid point, on the
                published panel. Nothing here is fitted, and the 0-threshold
                column must reproduce the published values exactly.
THRESHOLDS      CHOSEN — a grid rather than a preferred value, which is what
                makes this a sensitivity check. The cutpoints are read off the
                panel's own capitalisation quantiles rather than picked as
                round absolute figures, so each column removes a stated share
                of the universe and the grid's coverage is a measurement
                instead of an assertion. No grid point is adopted downstream;
                the published pipeline leaves MIN_MARKET_CAP at its
                no-filter setting.
MIN_DAYS_PORTFOLIO  CHOSEN — 20 valid days before a long-short series is
MIN_DAYS_IC     summarised and 10 before a rank-IC series is. Safety
                guardrails on a summary statistic rather than tuned cutpoints,
                binding at no point of the grid, which
                check_threshold_grid_summary_floors_do_not_bind reads off the
                summaries each point reports.
MIN_STOCKS_PER_DAY  inherited from shared_utils.config, the floor every
                Fama-MacBeth specification in the paper uses.

BASELINE: the unfiltered panel as the published specification | obvious
alternative: a referee asks whether the result is carried by micro-caps that a
size filter would remove | discharge: this sweep, reported in
subsec:appendix_j_marketcap.

Usage:
    cd code && PYTHONPATH=. python empirical/robustness/threshold_sensitivity.py
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from scipy import stats
from shared_utils import (
    MIN_STOCKS_PER_DAY_DEFAULT,
    daily_stats_pack,
    fama_macbeth_horse_race,
    load_domestic_panel,
    portfolio_stats,
    print_header,
    quintile_spread,
    save_csv,
    save_json,
)
from paths import RESULTS_DIR

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# The grid is anchored to the panel's OWN market-cap distribution rather than
# to round absolute cutoffs, so that each column removes a stated share of the
# universe. The panel's quantiles are p5 = 100 bn, p25 = 239 bn, p50 = 496 bn,
# p70 = 1.0 tn and p82 = 2.0 tn KRW, so the six columns below span no filter
# through the highest floor, removing 0, 5, 26, 50, 69 and 82 percent
# of stock-days. An absolute grid topping out at 100 bn would leave four of six
# columns removing under 0.6% of stock-days, which cannot test the small-cap
# inclusion the appendix (subsec:appendix_j_marketcap) claims to test. The
# realised removal share of each column is recorded per row below and reported
# in the table, so the grid's coverage is read off the table rather than
# asserted in prose.
THRESHOLDS = [0, 100e9, 250e9, 500e9, 1000e9, 2000e9]
THRESHOLD_LABELS = ['0 (none)', '100B', '250B', '500B', '1T', '2T']

MIN_STOCKS_PER_DAY = MIN_STOCKS_PER_DAY_DEFAULT

# Minimum number of valid trading days before a long-short return series
# (Panel B) or rank-IC series (Panel C) is summarised.
MIN_DAYS_PORTFOLIO = 20
MIN_DAYS_IC = 10


# ============================================================================
# DATA LOADING
# ============================================================================

def load_unfiltered_data():
    """Load the canonical Capacity-Bound Institutional panel.

    Reads the same ``results/full_daily_data_all_investors.parquet`` panel the headline
    Fama-MacBeth analysis uses (via :func:`load_domestic_panel`). The
    published pipeline leaves ``MIN_MARKET_CAP`` at its no-filter setting, so
    the panel is unfiltered on market cap; ``apply_pipeline_filters`` (NaN/inf
    drops, size-quintile binning) has been applied at panel-construction time.
    The threshold sweep below then re-applies ``M >= threshold`` per
    stock-day.
    """
    df = load_domestic_panel()
    print(f"\nPanel: {len(df):,} obs, {df['ticker'].nunique()} tickers "
          f"(D = D_capacity_bound != 0)")
    print(f"Market cap range: {df['M'].min()/1e9:.1f}B - {df['M'].max()/1e12:.1f}T KRW")
    return df


# ============================================================================
# FILTERING
# ============================================================================

def apply_threshold(df, threshold):
    """Apply a market cap threshold to the dataset."""
    if threshold > 0:
        df_filtered = df[df['M'] >= threshold].copy()
    else:
        df_filtered = df.copy()

    return df_filtered


# ============================================================================
# METRICS: Long-short quintile Sharpe ratio
# ============================================================================

def long_short_sharpe(df, signal_col, return_col='R_t1'):
    """
    Compute annualized Sharpe ratio for Q5-Q1 long-short portfolio.
    """
    dates = sorted(df['date'].unique())
    ls_returns = []
    n_skipped_unique = 0

    for date in dates:
        df_date = df[df['date'] == date].dropna(subset=[signal_col, return_col])
        if len(df_date) < MIN_STOCKS_PER_DAY:
            continue

        # qcut needs at least 5 distinct signal values to form 5 quantiles.
        if df_date[signal_col].nunique() < 5:
            n_skipped_unique += 1
            continue

        spread = quintile_spread(df_date, signal_col, return_col)
        if np.isfinite(spread):
            ls_returns.append(spread)

    if n_skipped_unique:
        print(f"  long_short_sharpe({signal_col}): skipped {n_skipped_unique} "
              f"dates with <5 unique signal values")
    if len(ls_returns) < MIN_DAYS_PORTFOLIO:
        return None

    return portfolio_stats(ls_returns)


# ============================================================================
# METRICS: Information Coefficient
# ============================================================================

def information_coefficient(df, signal_col, return_col='R_t1'):
    """
    Compute mean daily cross-sectional rank IC (Spearman correlation).
    """
    dates = sorted(df['date'].unique())
    daily_ics = []

    for date in dates:
        df_date = df[df['date'] == date].dropna(subset=[signal_col, return_col])
        if len(df_date) < MIN_STOCKS_PER_DAY:
            continue

        ic, _ = stats.spearmanr(df_date[signal_col], df_date[return_col])
        if np.isfinite(ic):
            daily_ics.append(ic)

    if len(daily_ics) < MIN_DAYS_IC:
        return None

    ics = np.array(daily_ics)
    s = daily_stats_pack(ics)

    return {
        'mean_ic': s['mean'],
        'ic_t_stat': s['t'],
        'ic_std': float(ics.std()),
        'n_dates': len(daily_ics),
    }


# ============================================================================
# MAIN SWEEP
# ============================================================================

def run_sensitivity(df, thresholds=THRESHOLDS, threshold_labels=THRESHOLD_LABELS):
    """Run the full sensitivity sweep."""
    results = []
    n_unfiltered = len(df)

    for thresh, label in zip(thresholds, threshold_labels):
        print_header(f"THRESHOLD: {label} KRW  (MIN_MARKET_CAP = {thresh:.0e})")

        df_filt = apply_threshold(df, thresh)
        n_obs = len(df_filt)
        n_tickers = df_filt['ticker'].nunique()
        n_dates = df_filt['date'].nunique()

        pct_removed = 100.0 * (1.0 - n_obs / n_unfiltered)

        print(f"  Universe: {n_obs:,} obs | {n_tickers} tickers | {n_dates} dates "
              f"| {pct_removed:.1f}% of stock-days removed")

        if n_obs == 0:
            print("  SKIPPED (no data)")
            continue

        row = {
            'threshold': thresh,
            'threshold_label': label,
            'n_obs': n_obs,
            'n_tickers': n_tickers,
            'n_dates': n_dates,
            'pct_stock_days_removed': pct_removed,
        }

        # --- Fama-MacBeth horse race (R_t+1 on S^MC and S^TV jointly) ---
        # This is what the published Panel A reports — see its table note.
        hr = fama_macbeth_horse_race(
            df_filt,
            signal_specs=[('S_MC', 'S_MC'), ('S_TV', 'S_TV')],
            return_col='R_t1',
            min_stocks_per_day=MIN_STOCKS_PER_DAY,
        )
        if hr:
            for signal in ['S_MC', 'S_TV']:
                row[f'fm_hr_tstat_{signal}_t+1'] = hr[signal]['t_stat']
                row[f'fm_hr_coef_{signal}_t+1'] = hr[signal]['coefficient']
                row[f'fm_hr_se_{signal}_t+1'] = hr[signal]['se']
            print(f"  HR t+1: t(S_MC)={hr['S_MC']['t_stat']:.2f}, "
                  f"t(S_TV)={hr['S_TV']['t_stat']:.2f}")

        # --- Long-short Sharpe ---
        for signal in ['S_MC', 'S_TV']:
            ls = long_short_sharpe(df_filt, signal, 'R_t1')
            if ls:
                row[f'ls_sharpe_{signal}'] = ls['sharpe']
                row[f'ls_mean_bps_{signal}'] = ls['mean_bps']
                row[f'ls_tstat_{signal}'] = ls['t_stat']
                print(f"  L/S {signal}: Sharpe={ls['sharpe']:.2f}, mean={ls['mean_bps']:.1f}bps, t={ls['t_stat']:.2f}")

        # --- IC ---
        for signal in ['S_MC', 'S_TV']:
            ic = information_coefficient(df_filt, signal, 'R_t1')
            if ic:
                row[f'ic_mean_{signal}'] = ic['mean_ic']
                row[f'ic_tstat_{signal}'] = ic['ic_t_stat']
                print(f"  IC {signal}: {ic['mean_ic']:.4f} (t={ic['ic_t_stat']:.2f})")

        results.append(row)

    return results


# ============================================================================
# OUTPUT
# ============================================================================

def print_summary_table(results):
    """Print a clean summary table."""
    print()
    print_header("THRESHOLD SENSITIVITY SUMMARY", level='banner', width=80)

    # --- Universe size ---
    print_header("Panel A: Universe Size")
    print(f"{'Threshold':<12} {'Obs':>12} {'Tickers':>10} {'Dates':>8} {'Removed %':>10}")
    print("-" * 54)
    for r in results:
        print(f"{r['threshold_label']:<12} {r['n_obs']:>12,} {r['n_tickers']:>10} "
              f"{r['n_dates']:>8} {r['pct_stock_days_removed']:>10.1f}")

    # --- Long-short Sharpe ---
    print_header("Panel B: Long-Short Sharpe Ratios (Q5-Q1, R_t+1)")
    print(f"{'Threshold':<12} {'SR(S_MC)':>10} {'SR(S_TV)':>10} {'bps(MC)':>10} {'bps(TV)':>10}")
    print("-" * 52)
    for r in results:
        sr_mc = r.get('ls_sharpe_S_MC', np.nan)
        sr_tv = r.get('ls_sharpe_S_TV', np.nan)
        bp_mc = r.get('ls_mean_bps_S_MC', np.nan)
        bp_tv = r.get('ls_mean_bps_S_TV', np.nan)
        print(f"{r['threshold_label']:<12} {sr_mc:>10.2f} {sr_tv:>10.2f} {bp_mc:>10.1f} {bp_tv:>10.1f}")

    # --- IC ---
    print_header("Panel C: Information Coefficient (Rank IC, R_t+1)")
    print(f"{'Threshold':<12} {'IC(S_MC)':>10} {'t(IC_MC)':>10} {'IC(S_TV)':>10} {'t(IC_TV)':>10}")
    print("-" * 52)
    for r in results:
        ic_mc = r.get('ic_mean_S_MC', np.nan)
        it_mc = r.get('ic_tstat_S_MC', np.nan)
        ic_tv = r.get('ic_mean_S_TV', np.nan)
        it_tv = r.get('ic_tstat_S_TV', np.nan)
        print(f"{r['threshold_label']:<12} {ic_mc:>10.4f} {it_mc:>10.2f} {ic_tv:>10.4f} {it_tv:>10.2f}")


def save_results(results):
    """Save results to JSON and CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    save_json({
        'metadata': {
            'description': 'Market cap threshold sensitivity analysis',
            'thresholds_krw': [float(t) for t in THRESHOLDS],
            'threshold_labels': THRESHOLD_LABELS,
            'generated': datetime.now().isoformat(),
        },
        'results': results,
    }, RESULTS_DIR / 'threshold_sensitivity_results.json')

    # CSV
    save_csv(pd.DataFrame(results),
             RESULTS_DIR / 'threshold_sensitivity_results.csv')


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    print_header("MARKET CAP THRESHOLD SENSITIVITY ANALYSIS", level='banner', width=80)
    print(f"Thresholds: {THRESHOLD_LABELS}")
    print(f"Start:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    start_time = datetime.now()

    df = load_unfiltered_data()
    results = run_sensitivity(df, THRESHOLDS, THRESHOLD_LABELS)

    print_summary_table(results)
    save_results(results)

    duration = (datetime.now() - start_time).total_seconds()
    print(f"\nCompleted in {duration:.0f}s ({duration/60:.1f} min)")


if __name__ == '__main__':
    main()
