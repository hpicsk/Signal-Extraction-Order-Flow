#!/usr/bin/env python3
"""Re-estimate every ordering claim on within-day ranks instead of levels.

Both filters are ratios of a signed, heavy-tailed numerator to a positive
denominator, and the paper's regressions enter them as within-day z-scores. A
z-score does not bound leverage: a handful of stock-days sit past ten
cross-sectional standard deviations, and an OLS slope is a leverage-weighted
average, so those days can carry the estimate. The exposure is not symmetric
across the two normalisations, which is what makes it a threat to an ordering
rather than to a level. $S^{TV} = D/V$ is bounded by construction, because $D$
is the net of the buys and sells that $V$ totals, so $|S^{TV}| \\le 1$;
$S^{MC} = D/M$ carries no such bound. The paper already declines to compare
$R^2$ across the two normalisations because the mismatched filter's wider
signal distribution can inflate it mechanically (subsec:reversal_test), and
the slopes on standardised signals are exposed to the same asymmetry.

This script replaces each signal by its within-day van der Waerden normal
score, $\\Phi^{-1}(r_i / (n+1))$ over that day's cross-section, and re-runs
every specification the ordering claims rest on. The transform is the exact
closure of the channel above: within a day the two filters are handed the
same set of scores in a different order, so their marginal distributions are
identical by construction and no difference between their slopes can come
from one having the wider tail. It also caps leverage near three standard
deviations. Feeding the scores through the same day-standardisation the
levels get leaves the estimator untouched and keeps coefficients in basis
points per cross-sectional standard deviation, so the two arms are read side
by side. The return is left in levels, because the transform addresses the
difference between the two filters' tails, and the return is the same variable
in both regressions of a comparison.

The rank arm bounds the advantage from below rather than re-estimating it
without bias, and the bound is measurable. Since $S^{TV} = S^{MC}\\tau^{-1}$
with $\\tau > 0$, a day of constant turnover would leave the two filters with
identical ranks and an advantage of exactly zero, so the transform retains
only the ordinal part of the turnover rescaling. The ``filter_correlation``
block reports how much of the rescaling survives it, as the mean within-day
correlation of the two standardised filters in each arm.

Degrees of freedom
------------------
CHOSEN    the van der Waerden score $\\Phi^{-1}(r/(n+1))$ over the plain
          ranks. Any strictly increasing map of the ranks closes the
          distribution channel equally; normal scores are chosen because they
          leave the regressor Gaussian-shaped, so the Fama-MacBeth slope keeps
          the units and the interpretation it has in the level arm.
CHOSEN    ranks are taken within (day, class), the cross-section each
          specification already standardises over, so the transform changes
          the shape of the regressor and not the set of rows it is fitted on.
BASELINE: the within-day z-score | obvious alternative: a referee asks whether
          a few extreme stock-days carry the ordering | discharge: this
          script, reported in subsec:appendix_j_rank.

Arms
----
``zscore``  the published specification. It must reproduce
            ``results/multi_investor_robustness_results.json`` and
            ``results/reversal_interaction_results.json`` exactly; a mismatch means
            the row filters drifted between this script and those two.
``rank``    the same specification on within-day normal scores.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/rank_transform_robustness.json
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from shared_utils import (
    MC_SEED,
    fm_investor_signal_triple,
    print_header,
    save_json,
)

from empirical.return_prediction.reversal_interaction import (
    BASE_CLASS,
    INPUT_FILE,
    INTERACTED_CLASS,
    MIN_STOCKS_PER_DAY,
    RETURN_HORIZON,
    fama_macbeth_reversal,
    fama_macbeth_univariate,
    load_stacked,
    standardise_within_day_class,
)
from paths import RESULTS_DIR

RESULTS_PATH = str(RESULTS_DIR / "rank_transform_robustness.json")

# The three disclosed classes of tab:investor_heterogeneity, in panel order.
CLASSES = (BASE_CLASS, INTERACTED_CLASS, "individual")

ARMS = ("zscore", "rank")

np.random.seed(MC_SEED)


def normal_scores(s):
    """Van der Waerden scores $\\Phi^{-1}(r/(n+1))$ of one cross-section.

    Ties take their average rank, so tied signal values keep tied scores. The
    $n+1$ denominator is what keeps the extreme ranks off $\\pm\\infty$; it
    puts them at roughly $\\pm 3$ for the cross-section sizes in this panel.
    """
    return norm.ppf(s.rank(method="average") / (len(s) + 1))


def rank_within(frame, cols, keys):
    """Replace ``cols`` by their normal scores within each ``keys`` group."""
    out = frame.copy()
    for col in cols:
        out[col] = out.groupby(keys, sort=False)[col].transform(normal_scores)
    return out


# --------------------------------------------------------------------------
# Per-class triples (tab:investor_heterogeneity)
# --------------------------------------------------------------------------

def class_panel(df, cls):
    """The rows tab:investor_heterogeneity fits for ``cls``.

    ``run_fm_triple`` applies the nonzero-flow filter and the joint dropna
    itself, per specification. Applying both here first makes them no-ops
    downstream and, more to the point, fixes the cross-section the ranks are
    taken over to the one the regression is run on. The ``zscore`` arm's
    agreement with the published triple is what confirms the two coincide.
    """
    cols = [f"S_MC_{cls}", f"S_TV_{cls}"]
    sub = df[df[f"D_{cls}"] != 0]
    return sub.dropna(subset=cols + [RETURN_HORIZON]).copy()


def run_class(df, cls):
    """Fit the univariate/horse-race triple for one class under both arms."""
    panel = class_panel(df, cls)
    panels = {"zscore": panel,
              "rank": rank_within(panel, [f"S_MC_{cls}", f"S_TV_{cls}"], "date")}

    out = {}
    for arm, frame in panels.items():
        out[arm] = fm_investor_signal_triple(
            frame, cls, return_col=RETURN_HORIZON,
            min_stocks_per_day=MIN_STOCKS_PER_DAY,
        )

    print(f"\n  {cls}  ({len(panel):,} rows)")
    print(f"    {'spec':<16} {'z coef (bp)':>12} {'z t':>8} "
          f"{'rank coef (bp)':>15} {'rank t':>8}")
    for spec in ("univariate_MC", "univariate_TV", "horse_race_MC", "horse_race_TV"):
        z, r = out["zscore"][spec], out["rank"][spec]
        print(f"    {spec:<16} {z['mean'] * 1e4:>12.2f} {z['t_stat']:>8.2f} "
              f"{r['mean'] * 1e4:>15.2f} {r['t_stat']:>8.2f}")
    return out


# --------------------------------------------------------------------------
# Stacked reversal (subsec:reversal_test)
# --------------------------------------------------------------------------

def stacked_panels(sample):
    """Both arms' standardised stacked panels for one stacking.

    The dropna runs before the ranks are taken so that the scores are formed
    on the cells ``standardise_within_day_class`` will keep, which is the same
    ordering of operations ``class_panel`` uses. That function's own dropna is
    then a no-op on both arms.
    """
    long = load_stacked(sample).dropna(subset=["zMC", "zTV", RETURN_HORIZON])
    return {
        "zscore": standardise_within_day_class(long),
        "rank": standardise_within_day_class(
            rank_within(long, ["zMC", "zTV"], ["date", "Foreign"])),
    }


def filter_correlation(panel):
    """Mean within-(day, class) correlation of the two standardised filters.

    The share of the turnover rescaling the transform preserves. A correlation
    of one would mean the two filters carry identical ranks every day and the
    rank arm's advantage would be zero by construction, so this is what says
    how far the rank arm understates the level arm.
    """
    cors = panel.groupby(["date", "Foreign"], sort=False).apply(
        lambda cell: cell["zMC"].corr(cell["zTV"]), include_groups=False)
    return {cls: float(cors.xs(float(cls == INTERACTED_CLASS), level="Foreign").mean())
            for cls in (BASE_CLASS, INTERACTED_CLASS)}


def run_stacked(sample):
    """Estimate the reversal on one stacking under both arms."""
    print_header(f"Stacked reversal — {sample} sample", width=72)
    out = {}
    for arm, panel in stacked_panels(sample).items():
        out[arm] = {
            "n_rows": int(len(panel)),
            "n_dates": int(panel["date"].nunique()),
            "fama_macbeth_univariate": fama_macbeth_univariate(panel),
            "fama_macbeth": fama_macbeth_reversal(panel),
            "filter_correlation": filter_correlation(panel),
        }

    for spec in ("fama_macbeth_univariate", "fama_macbeth"):
        print(f"\n  {spec}")
        for key in ("advantage_capacity", "advantage_foreign", "reversal"):
            z, r = out["zscore"][spec][key], out["rank"][spec][key]
            print(f"    {key:<20} z = {z['mean'] * 1e4:+7.2f} (t = {z['t']:+6.2f})"
                  f"    rank = {r['mean'] * 1e4:+7.2f} (t = {r['t']:+6.2f})")
    print("\n  mean within-day corr(zMC, zTV)")
    for cls in (BASE_CLASS, INTERACTED_CLASS):
        print(f"    {cls:<20} z = {out['zscore']['filter_correlation'][cls]:.4f}"
              f"    rank = {out['rank']['filter_correlation'][cls]:.4f}")
    return out


def main():
    """Entry point: re-run the per-class triples and the reversal on ranks."""
    print_header("Ordering claims under a within-day rank transform", level="banner")

    cols = ["date", "ticker", RETURN_HORIZON]
    for cls in CLASSES:
        cols += [f"D_{cls}", f"S_MC_{cls}", f"S_TV_{cls}"]
    df = pd.read_parquet(INPUT_FILE, columns=cols)

    print_header("Per-class return prediction", width=72)
    results = {
        "config": {
            "return_horizon": RETURN_HORIZON,
            "classes": list(CLASSES),
            "arms": list(ARMS),
            "min_stocks_per_day": int(MIN_STOCKS_PER_DAY),
        },
        "per_class": {cls: run_class(df, cls) for cls in CLASSES},
        "stacked": {sample: run_stacked(sample) for sample in ("common", "union")},
    }
    save_json(results, RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()
