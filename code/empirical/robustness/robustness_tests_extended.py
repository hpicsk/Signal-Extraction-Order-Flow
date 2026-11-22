#!/usr/bin/env python3
"""
QF Paper - Extended Robustness Tests (capacity-bound / foreign)

Six robustness tests, parametrized by investor type:
1.  Pre-period hold-out validation (2015-2019 holdout panel, genuine non-overlap)
1b. Yearly subperiod return prediction (subsec:appendix_j_oos)
2.  Long-short portfolio returns (economic significance)
3.  Outlier robustness (winsorization)
4.  Alternative standard errors (Newey-West)
5.  Factor neutrality / Fama-French 3-factor spanning test

Run with --investor institutional (default) or --investor foreign. The
``institutional`` CLI value is shorthand for the Capacity-Bound
Institutional subset (pension + insurance): the script loads
``load_domestic_panel``, whose canonical ``D`` is ``D_capacity_bound``.
The CLI value is kept for backwards compatibility with downstream JSON
filenames; the KRX 기관계 aggregate (which is null at the next-day
horizon) is reported separately in
subsec:appendix_aggregate_alignment / sec:appendix_subinvestor_decomposition.

Degrees of freedom
------------------
MEASURED        every statistic the six tests report, on the published panel
                and on the 2015-2019 holdout. Nothing here is fitted.
MIN_STOCKS_PER_PORTFOLIO_DAY  CHOSEN — 50, stricter than the Fama-MacBeth
                floor of 30, because a quintile sort needs a larger cross-
                section for stable 20% tail breakpoints than a single
                regression needs for a slope. A generic non-degeneracy guard,
                not a cutpoint tuned on the spread.
N_QUINTILES     CHOSEN — 5, the partition width every sort in the paper uses.
                The long-short is the Q5-minus-Q1 leg of it.
winsor limits   CHOSEN — the outlier arm reports the percentiles it trims
                through \\winsorLowerPct / \\winsorUpperPct, so the manuscript
                states the choice rather than hiding it inside the arm.
NW lags         CHOSEN — 5 and 10, reported side by side in
                subsec:appendix_j_outlier, so the serial-correlation arm does
                not rest on one lag length.
MC_SEED         inherited from shared_utils.config. The tests are
                deterministic given the panel; the seed is set so any future
                resampling arm inherits the paper's committed seed.

BASELINE: the published Fama-MacBeth specification as the thing being stressed
| obvious alternative: a referee asks whether the headline survives outliers,
serial correlation, a non-overlapping window, or the three Fama-French factors
| discharge: the six arms of this script, reported across
sec:appendix_robustness_battery.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

from shared_utils import (
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    fit_fama_macbeth,
    load_all_investors_panel,
    load_domestic_panel,
    ols_fit,
    portfolio_stats,
    print_header,
    quintile_spread,
    fit_fm_spec,
    run_fm_triple,
    save_json,
    t_stars,
)
from shared_utils.data import winsorize
from paths import RESULTS_DIR, DAILY_DATA_ALL_INVESTORS, DAILY_DATA_PRE2020

warnings.filterwarnings('ignore')

ALL_INVESTORS_INPUT = DAILY_DATA_ALL_INVESTORS
PRE_PERIOD_INPUT = DAILY_DATA_PRE2020

np.random.seed(MC_SEED)

# Minimum cross-section size for a day to enter the long-short / factor-spanning
# portfolio formation (Tests 2 & 5). Distinct from — and stricter than — the
# Fama-MacBeth floor MIN_STOCKS_PER_DAY_DEFAULT (30): quintile sorts need a
# larger cross-section for stable 20%-tail breakpoints. A code-level formation
# choice, not separately reported in the manuscript (the long-short and
# factor-neutrality results it feeds are in
# subsec:appendix_j_longshort / subsec:appendix_j_factor).
# Justification: a generic non-degeneracy
# guard for the quintile breakpoints, not a cutpoint tuned on the spread.
MIN_STOCKS_PER_PORTFOLIO_DAY = 50

# Number of buckets in the long-short sort (Q5 vs Q1).
N_QUINTILES = 5


# =============================================================================
# INVESTOR-TYPE CONFIG
# =============================================================================

@dataclass(frozen=True)
class InvestorConfig:
    """Per-investor parameters: the signal column names, flow filter, output
    filename, the signal whose Q5-Q1 spread feeds the factor-spanning test, and
    the one portfolio-formation switch that differs between the capacity-bound
    and foreign panels (whether to skip days with fewer than N_QUINTILES
    distinct signal values)."""
    name: str               # 'institutional' or 'foreign'
    s_mc: str               # column name for MC-normalised flow
    s_tv: str               # column name for TV-normalised flow
    flow_col: str | None    # filter where this column != 0; None = pre-filtered
    output_filename: str    # results JSON filename
    spanning_signal: str    # which signal forms the long-short for the FF3 test
    require_qcut_5_unique: bool  # skip dates with < N_QUINTILES unique signal values
    long_short_legs: tuple  # the Q5-Q1 legs the manuscript reports, of 'S_MC' / 'S_TV'
    pre_period_univariate: bool  # whether the pre-period arm reports the univariate fits


CONFIGS = {
    'institutional': InvestorConfig(
        name='institutional',
        s_mc='S_MC',
        s_tv='S_TV',
        flow_col=None,
        output_filename='extended_robustness_results.json',
        spanning_signal='S_MC',
        require_qcut_5_unique=False,
        # subsec:appendix_j_longshort reports both legs.
        long_short_legs=('S_MC', 'S_TV'),
        # tab:robustness_summary reports the pre-period horse race alone.
        pre_period_univariate=False,
    ),
    'foreign': InvestorConfig(
        name='foreign',
        s_mc='S_MC_foreign',
        s_tv='S_TV_foreign',
        flow_col='D_foreign',
        output_filename='extended_robustness_foreign_results.json',
        spanning_signal='S_TV_foreign',
        require_qcut_5_unique=True,
        # tab:robustness_summary reports the matched filter's leg alone.
        long_short_legs=('S_TV',),
        # subsec:appendix_foreign_robustness reports the signs of both
        # specifications on the pre-period.
        pre_period_univariate=True,
    ),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(cfg: InvestorConfig):
    """Load the daily panel and the unfiltered version the factors are built on.

    Returns ``(df_filtered, df_unfiltered)``. The flow restriction that defines
    the regression sample is a property of that sample, not of the equity
    universe the Fama-French analogues span, so MKT and SMB are always built on
    the unfiltered multi-investor panel — the same universe
    ``code/data/construct_hml_factor.py`` builds HML on. This holds for the
    ``institutional`` arm (= capacity-bound, see module docstring) as much as
    for ``foreign``.
    """
    print_header(f"LOADING DATA ({cfg.name.upper()})")

    if cfg.flow_col is None:
        df = load_domestic_panel()
        df_unfiltered = load_all_investors_panel(
            ALL_INVESTORS_INPUT, required_cols=['R_t1', 'M'], verbose=False,
        )
        print(f"Factor universe: {len(df_unfiltered):,} stock-days "
              f"(regression sample {len(df):,} after the D != 0 filter)")
    else:
        required = [cfg.s_mc, cfg.s_tv, cfg.flow_col, 'R_t1', 'M']
        df_unfiltered = load_all_investors_panel(
            ALL_INVESTORS_INPUT, required_cols=required, verbose=False,
        )
        df = df_unfiltered[df_unfiltered[cfg.flow_col] != 0].copy()
        print(f"After {cfg.flow_col} != 0 filter: {len(df):,} observations "
              f"(dropped {len(df_unfiltered) - len(df):,})")

    print(f"Loaded {len(df):,} observations")
    print(f"Unique dates: {df['date'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print()

    return df, df_unfiltered


# =============================================================================
# FAMA-MACBETH REGRESSION CORE
# =============================================================================

def fama_macbeth_regression(df, cfg: InvestorConfig, univariate):
    """Day-standardized Fama-MacBeth for the horse race and, when
    ``univariate`` is set, the MC and TV specs, via
    ``shared_utils.run_fm_triple`` (``shared_utils.fit_fm_spec`` for the horse
    race alone).

    ``S_MC`` and ``S_TV`` are first cleaned on a shared day-set (joint dropna on
    ``[s_mc, s_tv, R_t1]``) so the univariate MC and TV rows of the result
    table stay comparable. On that frame ``run_fm_triple``'s per-spec dropna is
    a no-op, so the three fits (univariate MC, univariate TV, horse race) use
    the intended day-sets; each delegates to ``prepare_fm_panel`` +
    ``linearmodels.FamaMacBeth`` (``cov_type='unadjusted'``, the historical
    simple-FM SE).

    Returns ``{'univariate': {MC, TV, n_days}, 'horse_race': {MC, TV, n_days}}``,
    without ``univariate`` when it is not set, where each ``MC``/``TV`` cell is
    ``{coef, t_stat, se}``.
    """
    common = df.dropna(subset=[cfg.s_mc, cfg.s_tv, 'R_t1'])
    if univariate:
        triple = run_fm_triple(
            common, return_col='R_t1', signal_mc=cfg.s_mc, signal_tv=cfg.s_tv,
            min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT,
        )
        horse_race = triple['hr']
    else:
        horse_race = fit_fm_spec(
            common, [cfg.s_mc, cfg.s_tv], 'R_t1',
            min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT,
        )

    def _cell(spec, col):
        if spec is None:
            return {}
        s = spec[col]
        return {'coef': s['coef'], 't_stat': s['t_stat'], 'se': s['se']}

    def _pack(mc_spec, tv_spec):
        out = {'MC': _cell(mc_spec, cfg.s_mc), 'TV': _cell(tv_spec, cfg.s_tv),
               'n_days': 0}
        # n_days follows the "last successful fit" convention: TV for the
        # univariate pair, the joint fit for the horse race.
        for spec in (mc_spec, tv_spec):
            if spec is not None:
                out['n_days'] = spec['n_dates']
        return out

    out = {'univariate': _pack(triple['mc'], triple['tv'])} if univariate else {}
    out['horse_race'] = _pack(horse_race, horse_race)
    return out


# =============================================================================
# TEST 1: PRE-PERIOD OUT-OF-SAMPLE VALIDATION (2015-2019)
# =============================================================================

def load_pre_period_panel(cfg: InvestorConfig):
    """Load the 2015-2019 pre-period panel filtered for the given investor type."""
    df = pd.read_parquet(PRE_PERIOD_INPUT)
    df['date'] = pd.to_datetime(df['date'])
    if cfg.flow_col is None:
        df = df[df['D'] != 0].copy()
    else:
        df = df[df[cfg.flow_col] != 0].copy()
    return df


def test_pre_period_oos(cfg: InvestorConfig):
    """Genuine OOS test: run FM regressions on the 2015-2019 pre-period panel,
    which does not overlap the main sample used to identify the matched filter."""
    print_header(f"TEST 1: PRE-PERIOD OOS VALIDATION (2015-2019, {cfg.name.upper()})", level='banner')

    df_pre = load_pre_period_panel(cfg)
    n_days = df_pre['date'].nunique()
    print(f"\nPre-period (2015-2019): {len(df_pre):,} obs, {n_days} days "
          f"({df_pre['date'].min().date()} to {df_pre['date'].max().date()})")

    print("\nRunning Fama-MacBeth regressions...")
    fm = fama_macbeth_regression(df_pre, cfg, univariate=cfg.pre_period_univariate)

    results = {
        **fm,
        'n_days': int(n_days),
        'n_obs': int(len(df_pre)),
    }

    print_header(f"PRE-PERIOD OOS RESULTS ({cfg.name.upper()})")
    for spec in ('univariate', 'horse_race'):
        if spec in fm:
            print(f"  {spec:<11} MC t = {fm[spec]['MC']['t_stat']:>7.2f}   "
                  f"TV t = {fm[spec]['TV']['t_stat']:>7.2f}")
    print()

    return results


# =============================================================================
# TEST 1b: YEARLY SUBPERIODS (RETURN PREDICTION)
# =============================================================================

def test_yearly_subperiods_returns(df, cfg: InvestorConfig, exclude_years=()):
    """Run FM regressions year-by-year on the main panel for return prediction.

    Reported in subsec:appendix_j_oos, Panel B.
    """
    print_header(f"TEST 1b: YEARLY RETURN-PREDICTION SUBPERIODS ({cfg.name.upper()})", level='banner')

    results = {}
    years = sorted(int(y) for y in df['date'].dt.year.unique() if int(y) not in exclude_years)
    for year in years:
        df_y = df[df['date'].dt.year == year]
        if len(df_y) == 0:
            continue
        fm = fama_macbeth_regression(df_y, cfg, univariate=False)
        results[str(year)] = {
            **fm,
            'n_days': int(df_y['date'].nunique()),
            'n_obs': int(len(df_y)),
        }

    print_header(f"YEARLY RETURN-PREDICTION RESULTS ({cfg.name.upper()})")
    print(f"{'Year':<6} {'N obs':<10} {'Days':<6} {'MC HR':<10} {'TV HR':<10}")
    print("-" * 46)
    for year, r in results.items():
        print(f"{year:<6} {r['n_obs']:<10,} {r['n_days']:<6} "
              f"{r['horse_race']['MC']['t_stat']:>8.2f}   "
              f"{r['horse_race']['TV']['t_stat']:>8.2f}")
    print()

    return results


# =============================================================================
# TEST 2: LONG-SHORT PORTFOLIO
# =============================================================================

def test_long_short_portfolio(df, cfg: InvestorConfig):
    """Sort daily into quintiles of each leg in ``cfg.long_short_legs`` and
    return its Q5-Q1 portfolio stats (mean bps, Sharpe, t, cumulative).

    A day enters when both normalized flows can be sorted into quintiles,
    whichever legs are formed, so every class's legs share one day-set.
    """
    print_header(f"TEST 2: LONG-SHORT PORTFOLIO RETURNS ({cfg.name.upper()})", level='banner')

    columns = {'S_MC': cfg.s_mc, 'S_TV': cfg.s_tv}
    dates = sorted(df['date'].unique())
    portfolio_returns = {leg: [] for leg in cfg.long_short_legs}
    n_skipped_unique = 0

    for date in dates:
        df_date = df[df['date'] == date].copy()

        if len(df_date) < MIN_STOCKS_PER_PORTFOLIO_DAY:
            continue
        if 'R_t1' not in df_date.columns:
            continue

        df_date = df_date.dropna(subset=[cfg.s_mc, cfg.s_tv, 'R_t1'])
        if len(df_date) < MIN_STOCKS_PER_PORTFOLIO_DAY:
            continue

        if cfg.require_qcut_5_unique:
            if (df_date[cfg.s_tv].nunique() < N_QUINTILES
                    or df_date[cfg.s_mc].nunique() < N_QUINTILES):
                n_skipped_unique += 1
                continue

        for leg in cfg.long_short_legs:
            portfolio_returns[leg].append(
                {'date': date, 'return': quintile_spread(df_date, columns[leg], 'R_t1')})

    if n_skipped_unique:
        print(f"  long_short_quintile: skipped {n_skipped_unique} dates with "
              f"<5 unique values in either signal")

    results = {leg: portfolio_stats(pd.DataFrame(rows)['return'].values)
               for leg, rows in portfolio_returns.items()}

    print_header(f"LONG-SHORT PORTFOLIO RESULTS (Q5 - Q1) — {cfg.name.upper()}")
    for leg, r in results.items():
        print(f"  {leg}: mean {r['mean_bps']:.2f} bps/day, Sharpe {r['sharpe']:.2f}, "
              f"t = {r['t_stat']:.2f}, cumulative {r['cum_ret'] * 100:.1f}%, "
              f"{r['n_days']} days")
    print()

    return results


# =============================================================================
# TEST 3: OUTLIER ROBUSTNESS
# =============================================================================

def test_outlier_robustness(df, cfg: InvestorConfig):
    """Re-run the horse-race FM after winsorizing signals and R_t1 at 1%/99%, returning baseline-vs-winsorized t-stats.

    The cut points are taken within each day, matching the regression the test
    perturbs: ``fama_macbeth_regression`` standardises the cross-section day by
    day, so a pooled cut point would trim whole calm days into the tails of
    turbulent ones rather than trim each day's own outliers.
    """
    print_header(f"TEST 3: OUTLIER ROBUSTNESS (WINSORIZATION) — {cfg.name.upper()}", level='banner')

    df_wins = df.copy()
    for col in (cfg.s_mc, cfg.s_tv, 'R_t1'):
        df_wins[col] = df_wins.groupby('date')[col].transform(winsorize)

    print("\nWinsorized at 1%/99% percentiles within each day")

    print("\nRunning baseline regressions...")
    baseline = fama_macbeth_regression(df, cfg, univariate=False)
    print("Running winsorized regressions...")
    wins = fama_macbeth_regression(df_wins, cfg, univariate=False)

    results = {'baseline': baseline, 'winsorized': wins}

    b_hr, w_hr = baseline['horse_race'], wins['horse_race']
    print_header(f"OUTLIER ROBUSTNESS RESULTS ({cfg.name.upper()})")
    print(f"\n{'Specification':<20} {'Baseline t':<15} {'Winsorized t':<15}")
    print("-" * 50)
    print(f"{'Horse Race MC':<20} {b_hr['MC']['t_stat']:>13.2f}   {w_hr['MC']['t_stat']:>13.2f}")
    print(f"{'Horse Race TV':<20} {b_hr['TV']['t_stat']:>13.2f}   {w_hr['TV']['t_stat']:>13.2f}")
    print()

    return results


# =============================================================================
# TEST 4: ALTERNATIVE STANDARD ERRORS
# =============================================================================

def test_alternative_se(df, cfg: InvestorConfig):
    """Horse-race FM with three SE flavours.

    Reports the same horse-race coefficients with (a) simple FM SE
    (``cov_type='unadjusted'``), (b) Newey-West HAC at 5 lags, (c)
    Newey-West HAC at 10 lags. All three flavours are computed by
    ``linearmodels.FamaMacBeth`` via separate ``cov_type`` passes.
    """
    print_header(f"TEST 4: ALTERNATIVE STANDARD ERRORS ({cfg.name.upper()})", level='banner')

    common = df.dropna(subset=[cfg.s_mc, cfg.s_tv, 'R_t1'])
    fits = {}
    for label, kwargs in [
        ('fm',   {'cov_type': 'unadjusted'}),
        ('nw5',  {'cov_type': 'kernel', 'kernel': 'bartlett', 'bandwidth': 5}),
        ('nw10', {'cov_type': 'kernel', 'kernel': 'bartlett', 'bandwidth': 10}),
    ]:
        res, panel = fit_fama_macbeth(
            common, [cfg.s_mc, cfg.s_tv], 'R_t1',
            min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT,
            **kwargs,
        )
        fits[label] = (res, panel)
    if fits['fm'][0] is None:
        raise RuntimeError("No day passed the FM pre-processing filter "
                           "for the alternative-SE test.")
    panel = fits['fm'][1]
    n_days = int(panel['date'].nunique())

    results = {}
    for var, col in [('MC', cfg.s_mc), ('TV', cfg.s_tv)]:
        coef = float(fits['fm'][0].params[col])
        results[var] = {
            'coef': coef,
            'fm_se':   float(fits['fm'][0].std_errors[col]),
            'fm_t':    float(fits['fm'][0].tstats[col]),
            'nw5_se':  float(fits['nw5'][0].std_errors[col]),
            'nw5_t':   float(fits['nw5'][0].tstats[col]),
            'nw10_se': float(fits['nw10'][0].std_errors[col]),
            'nw10_t':  float(fits['nw10'][0].tstats[col]),
        }
    results['n_days'] = n_days

    print_header(f"ALTERNATIVE STANDARD ERRORS RESULTS ({cfg.name.upper()})")
    print(f"\n{'SE Method':<25} {'MC t-stat':<15} {'TV t-stat':<15}")
    print("-" * 55)
    print(f"{'Fama-MacBeth':<25} {results['MC']['fm_t']:>13.2f}   {results['TV']['fm_t']:>13.2f}")
    print(f"{'Newey-West (5 lags)':<25} {results['MC']['nw5_t']:>13.2f}   {results['TV']['nw5_t']:>13.2f}")
    print(f"{'Newey-West (10 lags)':<25} {results['MC']['nw10_t']:>13.2f}   {results['TV']['nw10_t']:>13.2f}")
    print()

    return results


# =============================================================================
# TEST 5: FACTOR NEUTRALITY (FAMA-FRENCH 3-FACTOR SPANNING)
# =============================================================================

def construct_factors(df_unfiltered):
    """Build daily MKT (value-weighted) and SMB (small-minus-big) factors,
    then merge in pre-computed HML.

    These are analogues of FAMA19933, not replications, and the manuscript
    (subsec:appendix_j_factor) names the two departures the daily frequency
    forces. SMB is re-formed on each day's median market cap rather than
    annually at end-June, and both legs are equal-weighted rather than
    value-weighted within leg. MKT is the only value-weighted series here.
    """
    print("\nConstructing factors...")

    dates = sorted(df_unfiltered['date'].unique())
    rows = []
    for date in dates:
        df_date = df_unfiltered[df_unfiltered['date'] == date].copy()
        if len(df_date) < MIN_STOCKS_PER_PORTFOLIO_DAY:
            continue
        if 'R_t1' not in df_date.columns:
            continue
        df_date = df_date.dropna(subset=['M', 'R_t1'])
        if len(df_date) < MIN_STOCKS_PER_PORTFOLIO_DAY:
            continue

        total_mcap = df_date['M'].sum()
        df_date['weight'] = df_date['M'] / total_mcap
        mkt_ret = (df_date['weight'] * df_date['R_t1']).sum()

        median_mcap = df_date['M'].median()
        small = df_date[df_date['M'] <= median_mcap]
        big = df_date[df_date['M'] > median_mcap]
        smb_ret = (small['R_t1'].mean() if len(small) > 0 else 0) \
                  - (big['R_t1'].mean() if len(big) > 0 else 0)

        rows.append({'date': date, 'MKT': mkt_ret, 'SMB': smb_ret})

    factors_df = pd.DataFrame(rows)

    hml_path = str(RESULTS_DIR / 'hml_factor_daily.csv')
    hml = pd.read_csv(hml_path)
    hml['date'] = pd.to_datetime(hml['date'])
    factors_df = factors_df.merge(hml, on='date', how='left')
    missing = int(factors_df['HML'].isna().sum())
    if missing:
        raise ValueError(
            f"HML factor missing for {missing} dates after merge. "
            f"Re-run construct_hml_factor.py to extend HML coverage."
        )
    print(f"  HML factor loaded: {len(factors_df)} days, all non-null")
    return factors_df


def test_factor_neutrality(df_filtered, df_unfiltered, cfg: InvestorConfig):
    """Regress the cfg.spanning_signal Q5-Q1 portfolio on MKT/SMB/HML (Newey-West, 5 lags) and return alpha, betas, R^2."""
    print_header(f"TEST 5: FACTOR NEUTRALITY (3-FACTOR SPANNING TEST) — {cfg.name.upper()} {cfg.spanning_signal}", level='banner')

    dates = sorted(df_filtered['date'].unique())
    portfolio_returns = []
    n_skipped_unique = 0

    for date in dates:
        df_date = df_filtered[df_filtered['date'] == date].copy()
        if len(df_date) < MIN_STOCKS_PER_PORTFOLIO_DAY:
            continue
        if 'R_t1' not in df_date.columns:
            continue
        df_date = df_date.dropna(subset=[cfg.spanning_signal, 'R_t1', 'M'])
        if len(df_date) < MIN_STOCKS_PER_PORTFOLIO_DAY:
            continue

        if cfg.require_qcut_5_unique:
            if df_date[cfg.spanning_signal].nunique() < N_QUINTILES:
                n_skipped_unique += 1
                continue

        portfolio_returns.append(
            {'date': date,
             'portfolio_ret': quintile_spread(df_date, cfg.spanning_signal, 'R_t1')})

    if n_skipped_unique:
        print(f"  factor_neutrality: skipped {n_skipped_unique} dates with "
              f"<5 unique {cfg.spanning_signal} values")

    df_portfolio = pd.DataFrame(portfolio_returns)
    df_factors = construct_factors(df_unfiltered)
    df_merged = df_portfolio.merge(df_factors, on='date', how='inner')
    print(f"\nMerged data: {len(df_merged)} days")

    y = df_merged['portfolio_ret'].values
    X = df_merged[['MKT', 'SMB', 'HML']].values

    # Newey-West HAC standard errors (Bartlett kernel, 5 lags) via statsmodels.
    res = ols_fit(y, X).get_robustcov_results(cov_type='HAC', maxlags=5)
    beta = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    t_stats = np.asarray(res.tvalues, dtype=float)
    p_values = np.asarray(res.pvalues, dtype=float)

    results = {
        'alpha': {
            'coef': float(beta[0]),
            'se': float(se[0]),
            't_stat': float(t_stats[0]),
            'p_value': float(p_values[0]),
        },
        'MKT': {'coef': float(beta[1]), 'se': float(se[1]), 't_stat': float(t_stats[1])},
        'SMB': {'coef': float(beta[2]), 'se': float(se[2]), 't_stat': float(t_stats[2])},
        'HML': {'coef': float(beta[3]), 'se': float(se[3]), 't_stat': float(t_stats[3])},
        'r_squared': float(res.rsquared),
        'n_obs': int(res.nobs),
    }

    print_header(f"FACTOR SPANNING TEST RESULTS ({cfg.name.upper()} {cfg.spanning_signal}, 3-FACTOR)\n"
                 "Regression: R_portfolio = α + β_MKT × MKT + β_SMB × SMB + β_HML × HML + ε")
    print(f"\n{'Factor':<15} {'Coefficient':<15} {'t-statistic':<15}")
    print("-" * 45)
    sig = t_stars(results['alpha']['t_stat'])
    print(f"{'Alpha (α)':<15} {results['alpha']['coef']*10000:>13.2f} bps {results['alpha']['t_stat']:>13.2f} {sig}")
    print(f"{'MKT':<15} {results['MKT']['coef']:>13.4f}   {results['MKT']['t_stat']:>13.2f}")
    print(f"{'SMB':<15} {results['SMB']['coef']:>13.4f}   {results['SMB']['t_stat']:>13.2f}")
    print(f"{'HML':<15} {results['HML']['coef']:>13.4f}   {results['HML']['t_stat']:>13.2f}")
    print(f"\nR² = {results['r_squared']:.4f}")
    print(f"N = {results['n_obs']}")
    print()

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Extended robustness tests')
    parser.add_argument('--investor', choices=list(CONFIGS.keys()), required=True,
                        help='Investor type (no default — Makefile passes both '
                             '"institutional" and "foreign" in separate invocations).')
    args = parser.parse_args()
    cfg = CONFIGS[args.investor]

    print_header(f"QF PAPER - EXTENDED ROBUSTNESS TESTS ({cfg.name.upper()})", level='banner')
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    df, df_unfiltered = load_data(cfg)

    all_results = {
        'pre_period_oos':      test_pre_period_oos(cfg),
        'yearly_subperiods':   test_yearly_subperiods_returns(df, cfg),
        'long_short':          test_long_short_portfolio(df, cfg),
        'outlier_robustness':  test_outlier_robustness(df, cfg),
        'alternative_se':      test_alternative_se(df, cfg),
        'factor_neutrality':   test_factor_neutrality(df, df_unfiltered, cfg),
    }

    results_to_save = {
        'pre_period_oos':     all_results['pre_period_oos'],
        'yearly_subperiods':  all_results['yearly_subperiods'],
        'long_short':         all_results['long_short'],
        'outlier_robustness': all_results['outlier_robustness'],
        'alternative_se':     all_results['alternative_se'],
        'factor_neutrality':  all_results['factor_neutrality'],
        'timestamp':          datetime.now().isoformat(),
    }
    output_file = RESULTS_DIR / cfg.output_filename
    save_json(results_to_save, output_file, announce=False)

    print_header(f"ALL TESTS COMPLETE ({cfg.name.upper()})", level='banner')
    print(f"Results saved to: {output_file}")
    print()

    return all_results


if __name__ == '__main__':
    main()
