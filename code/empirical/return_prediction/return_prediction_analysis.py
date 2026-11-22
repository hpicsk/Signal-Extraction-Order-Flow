#!/usr/bin/env python3
"""
QF Paper - Return Prediction Analysis (main subsec:return_prediction return prediction)

Paper Reference: subsec:return_prediction ("Return Prediction")

This script tests: R_{t+h} ~ S^MC or S^TV
Purpose: Tests which normalisation better predicts future stock returns

Output:
- results/return_prediction_summary.csv: Fama-MacBeth next-day return prediction

Key Finding: For Capacity-Bound Institutional flow (pension + insurance,
the canonical ``D = D_capacity_bound``), market-capitalisation normalisation
(S^MC) delivers significant univariate next-day return predictability while
traded-value normalisation (S^TV) is weak; in the horse race S^MC
strengthens and S^TV reverses sign, which is the reading subsec:return_prediction
carries. Exact t-statistics live in ``results/return_prediction_results.json``.

Horizon: R_t+1, next-day returns.

Degrees of freedom
------------------
CHOSEN    R_t+1, the next-day horizon subsec:return_prediction reports.
CHOSEN    the (MC-only, TV-only, horse race) triple. The univariate arms are
          the estimand of prop:matched_filter and the horse race is the check
          that the ranking survives conditioning; all three are reported.
CHOSEN    ``D = D_capacity_bound`` (pension + insurance) as the class this
          script scores. subsec:regime_assignment fixes the class from its
          mandate, before any return is consulted.
CHOSEN    ``MIN_STOCKS_PER_DAY_DEFAULT``, inherited from
          ``shared_utils.config``, where it carries its own tag and the
          section that fixes it.
MEASURED  every coefficient, t-statistic, and average R^2 written below, on
          the class's own nonzero-flow stock-days.
FITTED    nothing. No constant here is set to reproduce a target statistic.

BASELINE: Fama-MacBeth over daily cross-sections with time-series standard
          errors | obvious alternative: a referee says next-day return
          predictability of this size is a bid-ask bounce the estimator does
          not remove, so the ranking is a microstructure artefact rather
          than a signal one | discharge: subsec:vwap_formation, where the
          continuation has the sign opposite to a bounce and the signal
          re-formed at the volume-weighted average price of
          subsec:vwap_formation keeps its predictability.
"""

import pandas as pd
import numpy as np
from shared_utils import (
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    pack_fm_spec_flat,
    print_header,
    run_fm_triple,
    save_csv,
    save_json,
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from paths import RESULTS_DIR

# Random seed for reproducibility
np.random.seed(MC_SEED)


def load_data(input_file=None):
    """Load the capacity-bound institutional panel and validate return columns."""
    from shared_utils import load_domestic_panel

    print_header("LOADING DATASET FOR RETURN PREDICTION")
    df = load_domestic_panel(input_file)

    return_cols = ['R_t', 'R_t1']
    missing_cols = [col for col in return_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing return columns: {missing_cols}. "
            f"Run full_daily_data_pipeline.py first.")

    print(f"Loaded {len(df):,} observations  "
          f"({df['date'].nunique():,} dates × {df['ticker'].nunique():,} tickers, "
          f"{df['date'].min()} – {df['date'].max()})")
    for col in return_cols:
        n = df[col].notna().sum()
        print(f"  {col}: {n:,} valid obs, mean={df[col].mean():.6f}, "
              f"std={df[col].std():.6f}")
    return df


def _print_return_spec(spec, cols, label):
    print(f"\n{label} — {' + '.join(cols)}")
    for c in cols:
        s = spec[c]
        stars = ('***' if s['p_value'] < 0.01 else '**' if s['p_value'] < 0.05
                 else '*' if s['p_value'] < 0.10 else '')
        print(f"  {c}: coef={s['coef']:.6f}{stars}  "
              f"se={s['se']:.6f}  t={s['t_stat']:.2f}  p={s['p_value']:.4f}")
    print(f"  Avg R²={spec['r_squared']:.6f}  "
          f"avg N/day={spec['avg_n']:.0f}  total N={spec['total_n']:,}")


def run_all_horizons(df, subset_desc=None):
    """Return prediction at the next-day horizon: S_MC-only, S_TV-only, horse race.

    Returns ``{horizon: {'mc_only': dict, 'tv_only': dict, 'horse_race': dict}}``
    with each leaf dict in the flat-key shape required by
    ``output/_macro_emitters._emit_return_macros``.
    """
    horizon_labels = {'R_t1': 'Next-day'}
    all_results = {}
    for horizon, hlabel in horizon_labels.items():
        print_header(f"HORIZON: {hlabel} ({horizon})", level='banner')
        triple = run_fm_triple(
            df, return_col=horizon, signal_mc='S_MC', signal_tv='S_TV',
            min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT,
        )
        if any(spec is None for spec in triple.values()):
            raise RuntimeError(
                f"No day passed the FM pre-processing filter for {horizon}.")
        tag = f"{subset_desc + ' - ' if subset_desc else ''}{hlabel}"
        _print_return_spec(triple['mc'], ['S_MC'], f"{tag} - S_MC")
        _print_return_spec(triple['tv'], ['S_TV'], f"{tag} - S_TV")
        _print_return_spec(triple['hr'], ['S_MC', 'S_TV'], f"{tag} - Horse Race")
        all_results[horizon] = {
            'mc_only': pack_fm_spec_flat(triple['mc'], ['S_MC'], return_horizon=horizon),
            'tv_only': pack_fm_spec_flat(triple['tv'], ['S_TV'], return_horizon=horizon),
            'horse_race': pack_fm_spec_flat(triple['hr'], ['S_MC', 'S_TV'], return_horizon=horizon),
        }
    return all_results


# Horizon labels for the results/return_prediction_summary.csv rows.
HORIZON_DISPLAY = {'R_t1': 'R_{t+1}'}


def _return_row(horizon_label, spec_label, *, coef_mc='', t_mc='',
                coef_tv='', t_tv='', avg_r2, obs):
    """One return-prediction summary row; unpopulated coef/t cells default to blank."""
    return {
        'Horizon': horizon_label,
        'Specification': spec_label,
        'Coef_MC': coef_mc,
        't_MC': t_mc,
        'Coef_TV': coef_tv,
        't_TV': t_tv,
        'Avg_R2': avg_r2,
        'Obs': obs,
    }


def format_results_table(all_results):
    """Format the per-horizon FM results into the return-prediction summary DataFrame."""
    rows = []
    for horizon, horizon_results in all_results.items():
        hl = HORIZON_DISPLAY[horizon]
        mc = horizon_results['mc_only']
        tv = horizon_results['tv_only']
        hr = horizon_results['horse_race']
        rows.append(_return_row(hl, 'MC Only',
                                coef_mc=mc['coef_S_MC'], t_mc=mc['t_stat_S_MC'],
                                avg_r2=mc['avg_r_squared'], obs=mc['total_n']))
        rows.append(_return_row(hl, 'TV Only',
                                coef_tv=tv['coef_S_TV'], t_tv=tv['t_stat_S_TV'],
                                avg_r2=tv['avg_r_squared'], obs=tv['total_n']))
        rows.append(_return_row(hl, 'Horse Race',
                                coef_mc=hr['coef_S_MC'], t_mc=hr['t_stat_S_MC'],
                                coef_tv=hr['coef_S_TV'], t_tv=hr['t_stat_S_TV'],
                                avg_r2=hr['avg_r_squared'], obs=hr['total_n']))
    return pd.DataFrame(rows)


def main():
    print_header("QF PAPER - RETURN PREDICTION ANALYSIS", level='banner')
    start_time = datetime.now()

    df = load_data()
    all_results = run_all_horizons(df, subset_desc="Full Sample")

    save_json(all_results, RESULTS_DIR / 'return_prediction_results.json')
    df_summary = format_results_table(all_results)
    save_csv(df_summary, RESULTS_DIR / 'return_prediction_summary.csv')

    print_header("RETURN PREDICTION SUMMARY")
    print(df_summary.to_string(index=False))

    duration = (datetime.now() - start_time).total_seconds()
    print_header(f"DONE in {duration:.1f}s — results in {RESULTS_DIR}", level='banner')
    return all_results


if __name__ == '__main__':
    main()
