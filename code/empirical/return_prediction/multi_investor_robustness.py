#!/usr/bin/env python3
"""
QF Paper - Multi-Investor Robustness Analysis

Paper Reference: subsec:return_prediction (return prediction by investor type) and
subsec:boundary_extent (aggregate alignment).

This script validates the matched filter theory by showing:
1. Only Capacity-Bound Institutional flow exhibits an MC normalisation
   advantage for next-day return prediction; foreign flow shows the
   reverse (TV) advantage
2. The KRX 기관계 (institutional aggregate) is itself MC-dominant at the
   next-day horizon, with the dominance pervasive across sub-types and
   sharpest on the capacity-bound subset (subsec:appendix_aggregate_alignment / sec:appendix_subinvestor_decomposition)

Which characteristic carries a class's flow is not decided here.
subsec:regime_assignment assigns it from the class's scaling role, and this
script consumes that assignment and does not recompute it.

Key Hypothesis:
- If matched filter theory is correct:
  - Capacity-Bound Institutional: MC normalisation should dominate TV
  - Foreign: TV normalisation should dominate MC
  - Individual: no shared mandate or execution convention, so no regime is
    assigned and no advantage is predicted (subsec:return_prediction)
  - Institutional Aggregate: itself MC-dominant (pervasive across sub-types)

Output Tables:
- Panel B: Return prediction by investor type

Degrees of freedom
------------------
CHOSEN    ``INVESTOR_TYPE_LIST``. The four classes are the ones KRX discloses
          that subsec:regime_assignment reaches, plus the institutional
          aggregate, which is reported because subsec:appendix_aggregate_alignment
          asks whether the capacity subset drives it.
CHOSEN    the day-standardised (MC-only, TV-only, horse race) triple, the
          same specification subsec:return_prediction fixes for every class.
MEASURED  every coefficient and $t$-statistic, on each class's own
          nonzero-flow stock-days.
FITTED    nothing.

BASELINE: the KRX investor-class definitions as disclosed | obvious
          alternative: a referee says the pension+insurance grouping is the
          analyst's construction rather than a mandate boundary, so the
          contrast is a partition chosen to produce it | discharge:
          sec:appendix_subinvestor_decomposition, which estimates all eight
          disclosed sub-types separately and reports that the dominance is
          pervasive across them rather than confined to the grouping.

"""

import pandas as pd
import numpy as np
from shared_utils import (
    CRIT_5PCT,
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    fm_investor_signal_triple,
    load_all_investors_panel,
    print_header,
    save_csv,
    save_json,
    t_stars,
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from paths import RESULTS_DIR, DAILY_DATA_ALL_INVESTORS

# Configuration
INPUT_FILE = DAILY_DATA_ALL_INVESTORS

# Investor types
# - 'capacity_bound' = pension + insurance (the canonical institutional actor,
#   the headline series)
# - 'institutional_aggregate' = KRX 기관계 aggregate, kept for the subsec:appendix_aggregate_alignment
#   aggregate-alignment paragraph
INVESTOR_TYPE_LIST = ['foreign', 'capacity_bound', 'individual', 'institutional_aggregate']

np.random.seed(MC_SEED)


def load_data(input_file=INPUT_FILE):
    """Load the multi-investor dataset, validating per-investor signal columns."""
    print_header("LOADING MULTI-INVESTOR DATASET")
    print(f"File: {input_file}")
    required = [
        col
        for inv_type in INVESTOR_TYPE_LIST
        for col in (f'D_{inv_type}', f'S_MC_{inv_type}', f'S_TV_{inv_type}')
    ]
    return load_all_investors_panel(input_file, required_cols=required)


def fama_macbeth_return_prediction(df, investor_type, return_horizon='R_t1'):
    """Per-investor FM return prediction: ``R_{t+h} ~ S_MC, S_TV`` (three specs)."""
    out = fm_investor_signal_triple(
        df, investor_type, return_col=return_horizon,
        min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT,
    )
    out['return_horizon'] = return_horizon
    return out


def _print_investor_spec(label, spec, coef_decimals):
    print(f"  {label}: coef={spec['mean']:.{coef_decimals}f}  "
          f"t={spec['t_stat']:.2f}", end="")
    if 'mean_R2' in spec:
        print(f"  R²={spec['mean_R2']:.3f}%")
    else:
        print()


def _run_investor_loop(df, runner, *, header, hypothesis, coef_decimals):
    """Iterate ``runner(df, inv_type)`` over ``INVESTOR_TYPE_LIST`` and print a
    compact per-investor summary. ``runner`` is
    ``fama_macbeth_return_prediction``.
    """
    print_header(header)
    print(hypothesis)
    all_results = {}
    for inv_type in INVESTOR_TYPE_LIST:
        print(f"\n--- {inv_type.upper()} ---")
        res = runner(df, inv_type)
        all_results[inv_type] = res
        print(f"  N days: {res['n_days']}")
        _print_investor_spec("Univariate S_MC", res['univariate_MC'], coef_decimals)
        _print_investor_spec("Univariate S_TV", res['univariate_TV'], coef_decimals)
        _print_investor_spec("Horse race S_MC", res['horse_race_MC'], coef_decimals)
        _print_investor_spec("Horse race S_TV", res['horse_race_TV'], coef_decimals)
    return all_results


def run_all_return_predictions(df):
    """Next-day return prediction FM (R_{t+1} ~ S_MC vs S_TV) for every investor type."""
    return _run_investor_loop(
        df, lambda d, inv: fama_macbeth_return_prediction(d, inv, return_horizon='R_t1'),
        header="FAMA-MACBETH RETURN PREDICTIONS",
        hypothesis="Testing: R_{t+1} ~ S^MC vs S^TV per investor type",
        coef_decimals=6,
    )


# capacity_bound replaces 'institutional' as the matched-S^MC headline actor;
# the aggregate is preserved separately for the subsec:appendix_aggregate_alignment aggregate-alignment paragraph.
# 'matched_arm' picks the FM univariate spec aligned with each investor's
# scaling regime (MC for capacity-bound / aggregate, TV for foreign).
MATCHED_ARMS = [
    ('capacity_bound',          'univariate_MC', 'S_MC'),
    ('foreign',                 'univariate_TV', 'S_TV'),
    ('institutional_aggregate', 'univariate_MC', 'S_MC'),
]

# Investors whose hypothesised matched normalisation is MC — capacity-bound and
# the KRX 기관계 aggregate (Key Hypothesis above, subsec:return_prediction/subsec:appendix_aggregate_alignment). Derived from
# MATCHED_ARMS so there is one source of truth; drives which interpretation
# branch print_summary takes (the MC-advantage check vs the volume-scaling check).
MC_DOMINANT_INVESTORS = {inv for inv, arm, _ in MATCHED_ARMS if arm == 'univariate_MC'}

def _fm_triple_row(inv, res):
    """One Panel B / Panel C summary row from an ``fm_investor_signal_triple``
    result (univariate MC, univariate TV, horse race; R² in percent)."""
    return {
        'Investor Type': inv.capitalize(),
        'MC Coef': res['univariate_MC']['mean'],
        'MC t-stat': res['univariate_MC']['t_stat'],
        'MC R2 (%)': res['univariate_MC']['mean_R2'],
        'TV Coef': res['univariate_TV']['mean'],
        'TV t-stat': res['univariate_TV']['t_stat'],
        'TV R2 (%)': res['univariate_TV']['mean_R2'],
        'HR MC Coef': res['horse_race_MC']['mean'],
        'HR MC t-stat': res['horse_race_MC']['t_stat'],
        'HR TV Coef': res['horse_race_TV']['mean'],
        'HR TV t-stat': res['horse_race_TV']['t_stat'],
        'HR R2 (%)': res['horse_race_R2'],
    }


def _mc_tv_hr_tstats(res):
    """(univariate MC, univariate TV, horse-race MC, horse-race TV) t-stats."""
    return (res['univariate_MC']['t_stat'], res['univariate_TV']['t_stat'],
            res['horse_race_MC']['t_stat'], res['horse_race_TV']['t_stat'])


def save_results(prediction_results, output_dir=RESULTS_DIR):
    """Save combined JSON + the per-panel summary CSV to ``output_dir``."""
    print_header("SAVING RESULTS")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine results
    combined_results = {
        'return_predictions': prediction_results,
        'timestamp': datetime.now().isoformat()
    }

    # Save JSON
    save_json(combined_results, output_dir / 'multi_investor_robustness_results.json')

    # Create summary DataFrame for Panel A (Scaling)
    # Create summary DataFrame for Panel B (Return Prediction)
    prediction_df = pd.DataFrame(
        [_fm_triple_row(inv, prediction_results[inv]) for inv in INVESTOR_TYPE_LIST])
    save_csv(prediction_df, output_dir / 'panel_b_return_prediction.csv')

    return prediction_df


def print_summary(prediction_results):
    """Print a final-summary interpretation of the return-prediction panel."""
    print_header("SUMMARY: INVESTOR HETEROGENEITY ANALYSIS", level='banner')

    print("\n--- PANEL B: RETURN PREDICTION ---")
    print("Hypothesis: MC advantage exists only for M-scaling investors (Institutional)\n")

    for inv in INVESTOR_TYPE_LIST:
        res = prediction_results[inv]
        mc_t, tv_t, hr_mc_t, hr_tv_t = _mc_tv_hr_tstats(res)

        print(f"{inv.upper()}:")
        print(f"  Univariate: MC t={mc_t:.2f}{t_stars(mc_t)}, "
              f"TV t={tv_t:.2f}{t_stars(tv_t)}")
        print(f"  Horse race: MC t={hr_mc_t:.2f}, TV t={hr_tv_t:.2f}")

        # Interpretation
        if inv in MC_DOMINANT_INVESTORS:
            if mc_t > CRIT_5PCT and hr_mc_t > CRIT_5PCT:
                print("  => SUPPORTS matched filter theory (MC advantage confirmed)")
            else:
                print("  => UNEXPECTED: MC advantage not found for MC-scaling investor")
        else:
            if abs(mc_t) < CRIT_5PCT and abs(tv_t) < CRIT_5PCT:
                print("  => SUPPORTS theory (neither normalisation predicts returns)")
            elif abs(mc_t) < CRIT_5PCT:
                print("  => SUPPORTS theory (no MC advantage for volume-scaling investor)")
            else:
                print("  => PARTIAL SUPPORT (some predictability found)")

    print()


def main():
    print_header("QF PAPER - MULTI-INVESTOR ROBUSTNESS (subsec:return_prediction)", level='banner')
    start_time = datetime.now()

    df = load_data()
    prediction_results = run_all_return_predictions(df)
    save_results(prediction_results)
    print_summary(prediction_results)

    duration = (datetime.now() - start_time).total_seconds()
    print_header(f"DONE in {duration:.1f}s", level='banner')
    return prediction_results


if __name__ == '__main__':
    prediction_results = main()
