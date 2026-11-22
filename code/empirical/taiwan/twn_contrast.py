#!/usr/bin/env python3
"""The contrast of eq:reversal_contrast on the Taiwan panel (sec:taiwan).

The paper's claim is the contrast between the two classes' advantages of
$S^{MC}$ over $S^{TV}$ on the stock-days that both classes trade
(subsec:reversal_test). sec:taiwan estimates each Taiwanese class on its own
panel and reads each class's filters apart, which tests each class's ordering
and not the contrast. This script estimates the contrast itself on the Taiwan
panel, with the stacked estimator and the stackings of
code/empirical/return_prediction/reversal_interaction.py, SITC in the place of the capacity-bound class.

Degrees of freedom
------------------
CHOSEN    the estimator, the standardisation within (day, class), the
          cross-section floor and the two stackings of code/empirical/return_prediction/reversal_interaction.py,
          imported rather than re-implemented, so the two markets differ in
          data and not in specification.
CHOSEN    SITC as the class in the capacity regime and foreign flow as the one
          in the volume regime, the assignment of code/empirical/taiwan/twn_analysis.py.
CHOSEN    ``common`` as the stacking the reading rule grades, as in Korea. The
          ``union`` stacking holds each class on its own nonzero-flow
          stock-days, the panels of tab:twn_return_prediction, so its
          univariate advantages are the differences of that table's univariate
          slopes and its point estimate is known before this script runs.
MEASURED  every advantage and contrast below, on the stock-days each stacking
          retains.
FITTED    nothing.

BASELINE: Fama-MacBeth over the per-day contrast series, the estimator of
          code/empirical/return_prediction/reversal_interaction.py | obvious alternative: a referee asks for
          pooled standard errors clustered by date | discharge:
          subsec:reversal_test reports the cluster-by-date t beside the
          Fama-MacBeth one for the Korean horse race, and the univariate
          contrast graded here spans two regressions per day, which no single
          fitted cross-sectional covariance covers (code/empirical/return_prediction/reversal_interaction.py).

Provenance
----------
Post-hoc. The paper's claim was narrowed from the reversal to the contrast on
2026-11-15, after the Korean and Taiwanese results had been read, and this run
tests the narrowed claim in the second market. tab:twn_return_prediction fixes
the union point estimate (SITC 17.28 - 9.43 = 7.85 bp, foreign 9.57 - 7.66 =
1.91 bp, contrast 5.94 bp). No estimate on common support had been computed when
the reading rule below was written; the common-support row count (403,026
stock-days over 1,458 days) had.

Reading rule
------------
Committed before the run. The contrast holds in Taiwan when the univariate
contrast on ``common`` is positive and its Fama-MacBeth t clears the paper's 5%
two-sided bar ``GATE_MIN_T``; otherwise it does not hold there.

Either outcome goes to the "Taiwan." paragraph of subsec:return_prediction, as
the comparison the rule decided, with the estimate and its t, and the advantages
and the contrast on both stackings go to a float of sec:taiwan. The first
sentence of subsec:limitations, which says the contrast is estimated in one
market, is rewritten in the same commit for either outcome.

Inputs
------
    results/full_daily_data_taiwan_all_investors.parquet

Output
------
    results/twn_contrast_results.json
"""

import pandas as pd
from shared_utils import print_header, save_json
from shared_utils.regression import CRIT_5PCT

from empirical.return_prediction.reversal_interaction import (
    RETURN_HORIZON,
    fama_macbeth_univariate,
    standardise_within_day_class,
)
from paths import DAILY_DATA_TAIWAN, RESULTS_DIR

RESULTS_PATH = str(RESULTS_DIR / "twn_contrast_results.json")

# The Taiwanese class in the capacity regime and the one in the volume regime,
# by their column suffixes in the Taiwan panel (code/empirical/taiwan/twn_analysis.py).
CAPACITY_CLASS = "institutional"
VOLUME_CLASS = "foreign"

# The paper's 5% two-sided bar, the one every gate in this repo is read at.
GATE_MIN_T = CRIT_5PCT


def load_stacked_taiwan(sample):
    """Return the long (stock-day, class) frame of the Taiwan panel.

    The frame has the columns that code/empirical/return_prediction/reversal_interaction.py's estimators read:
    ``date``, the return, ``zMC``, ``zTV`` and ``Foreign``, which is one on the
    foreign rows. ``common`` keeps the stock-days on which both classes' flows
    are nonzero, and ``union`` keeps each class on its own nonzero-flow
    stock-days.
    """
    cols = ["date", "stock_id", RETURN_HORIZON]
    for c in (CAPACITY_CLASS, VOLUME_CLASS):
        cols += [f"D_{c}", f"S_MC_{c}", f"S_TV_{c}"]
    df = pd.read_parquet(DAILY_DATA_TAIWAN, columns=cols)
    if sample == "common":
        df = df[(df[f"D_{CAPACITY_CLASS}"] != 0) & (df[f"D_{VOLUME_CLASS}"] != 0)]
    parts = []
    for cls in (CAPACITY_CLASS, VOLUME_CLASS):
        sub = df[df[f"D_{cls}"] != 0]
        part = sub[["date", "stock_id", RETURN_HORIZON]].copy()
        part["zMC"] = sub[f"S_MC_{cls}"].values
        part["zTV"] = sub[f"S_TV_{cls}"].values
        part["Foreign"] = float(cls == VOLUME_CLASS)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run_sample(sample):
    """Estimate the univariate specification on one stacking."""
    panel = standardise_within_day_class(load_stacked_taiwan(sample))
    out = {
        "n_rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "fama_macbeth_univariate": fama_macbeth_univariate(panel),
    }
    print_header(f"Taiwan stacked contrast, {sample}", width=72)
    print(f"Stacked rows: {out['n_rows']:,} over {out['n_dates']:,} dates")
    for name in ("advantage_capacity", "advantage_foreign", "reversal"):
        est = out["fama_macbeth_univariate"][name]
        print(f"  {name:<20} {est['mean'] * 1e4:+7.2f} bp  (t = {est['t']:+.2f})")
    return out


def main():
    results = {
        "config": {"return_horizon": RETURN_HORIZON, "gate_min_t": GATE_MIN_T,
                   "capacity_class": CAPACITY_CLASS, "volume_class": VOLUME_CLASS},
        "common": run_sample("common"),
        "union": run_sample("union"),
    }
    gate = results["common"]["fama_macbeth_univariate"]["reversal"]
    holds = bool(gate["mean"] > 0 and gate["t"] > GATE_MIN_T)
    results["verdict"] = {
        "criterion": "univariate contrast on common support > 0 with t > GATE_MIN_T",
        "contrast": gate["mean"],
        "t": gate["t"],
        "holds": holds,
    }
    print(f"\nCriterion committed before the run: {results['verdict']['criterion']}")
    print(f"  contrast = {gate['mean'] * 1e4:+.2f} bp at t = {gate['t']:+.2f}: "
          f"{'holds' if holds else 'does not hold'}")
    save_json(results, RESULTS_PATH)


if __name__ == "__main__":
    main()
