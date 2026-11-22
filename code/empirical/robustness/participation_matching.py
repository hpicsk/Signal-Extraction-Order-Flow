#!/usr/bin/env python3
"""Re-estimate the reversal with the two classes' net participation rates matched.

Common support puts the two classes on identical stock-days, so
$M_{i,t}$, $V_{i,t}$, $\\tau_{i,t}$ and $R_{i,t+1}$ are shared row by row and
the only thing that still differs between a capacity-bound row and its foreign
twin is $D_{i,t}$. Those flows are shaped differently. A capacity-bound
institution reaches a stock on a minority of sessions, and on the sessions both
classes trade it takes the smaller share of the day's volume, which is the
opposite of the ordering the objection assumes. The two filters are a
turnover rescaling apart, $S^{TV}_{i,t} = S^{MC}_{i,t}/\\tau_{i,t}$, so a class
whose flow sits at a different point of the participation-rate distribution
loads on them unequally, and the difference-in-differences of
eq:reversal_contrast would then carry the shape of each class's flow rather
than the carrier it scales with.
The exclusion subsec:reversal_test rests on does not reach that channel,
because it holds the firm-day fixed and the channel lives in the flow.

This script matches on the intensity itself. The matching variable is the
absolute participation rate $|S^{TV}_{i,t}| = |D_{i,t}|/V_{i,t}$, the quantity
the objection names, and matching runs inside turnover strata. Because
$|S^{MC}| = |S^{TV}|\\,\\tau$, holding $\\tau$ and $|S^{TV}|$ fixed together
holds $|S^{MC}|$ fixed as well, so the design equalises the magnitude
distribution of both filters at once rather than of the one it conditions on.
That is what keeps it neutral between the two, which a match on either filter
alone would not be.

Matching is coarsened exact matching \\citep{Iacus2012}. Within each date,
$\\log\\tau$ and $\\log|S^{TV}|$ are cut at their pooled ``N_BINS``-quantiles,
and each (date, turnover bin, intensity bin) cell is re-balanced to the
smaller of its two class counts by the weight $\\min(n_c, n_f)/n_{class}$.
Cells holding one class only receive weight zero and leave the sample, which
is the coarsened analogue of the common-support restriction the stacking
already applies at the firm-day level. Weights never exceed one, so no cell is
inflated and the retained population is the overlap of the two flow
distributions. Both the daily fit and the within-cell standardisation are
taken against those weights, since a z-score computed on the unmatched
dispersion would return the classes to different units.

Degrees of freedom
------------------
CHOSEN    $|S^{TV}|$ as the matching variable. It is the participation rate
          whose class difference the objection names, and inside a turnover
          stratum it pins $|S^{MC}|$ with it.
CHOSEN    N_BINS = 5 on each axis, the partition width the size-quintile and
          turnover-quintile stratifications of this paper already use. The
          decile grid is reported as the finer arm, so the headline is not the
          only granularity shown.
CHOSEN    bin edges are cut inside each daily cross-section and pooled over
          the two classes, so a bin is a within-day rank and the cut points
          cannot differ between the classes being matched.
CHOSEN    coarsened exact matching over a propensity-score model. The
          conditioning set is two continuous variables, so the cells can be
          filled directly and no functional form is fitted.
BASELINE: the unmatched stacking of subsec:reversal_test | obvious
          alternative: a referee says the capacity class trades in blocks and
          the foreign class in slices, so the reversal is the shape of the two
          flow distributions rather than the carrier each scales with |
          discharge: this script, reported in
          subsec:appendix_participation_matching.

Criterion
---------
Committed before the run. The reversal survives the matching when the headline
cell keeps the sign Corollary cor:regimes predicts and clears the paper's 5%
two-sided bar, ``GATE_MIN_T``. The headline cell is the ``matched`` arm of the
``common`` stacking under the univariate specification, which is the estimand
of subsec:reversal_test. The verdict is reported as it falls.

Arms
----
``full``      the subsec:reversal_test specification on the unrestricted
              stacking, which must reproduce
              ``results/reversal_interaction_results.json``. A mismatch means the
              stacking or standardisation drifted between the two scripts.
``overlap``   the same specification on the retained rows at unit weights,
              which prices what dropping the one-class cells does on its own.
              It is estimated for the headline cell, the one
              subsec:appendix_participation_matching reads it against, in the
              univariate specification that section reports.
``matched``   the same specification on those rows under the matching
              weights. Read against ``overlap``, the difference is the
              re-balancing; read against ``full``, it is the whole design.

The ``full`` and ``matched`` arms are fitted under both specifications of
subsec:reversal_test, the two that tab:participation_matching prints.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/participation_matching_results.json
"""

import numpy as np
import pandas as pd
from shared_utils import MC_SEED, print_header, save_json
from shared_utils.regression import CRIT_5PCT

from empirical.return_prediction.reversal_interaction import (
    fama_macbeth_reversal,
    fama_macbeth_univariate,
    load_stacked,
    standardise_within_day_class,
)
from paths import RESULTS_DIR

RESULTS_PATH = str(RESULTS_DIR / "participation_matching_results.json")

# Firm-day column the turnover stratum is cut on. The two filters are already
# carried onto the stacked frame by load_stacked as zMC and zTV.
PANEL_COLS = ("turnover",)

# Quantile count on each matching axis. GRIDS names the arms: the headline
# quintile grid and the finer decile grid reported beside it.
N_BINS = 5
GRIDS = {"quintile": 5, "decile": 10}

# Weight column the matched arm standardises and fits against.
WEIGHT_COL = "cemWeight"

# Balance is reported for the matching variable, the stratifier, and the
# filter magnitude they jointly pin.
BALANCE_COLS = ("logIntensity", "logTurnover", "logCapMagnitude")

# Standardised mean difference a matched design is conventionally asked to
# clear \citep{Austin2009}. Diagnostic rather than a gate: the criterion
# below is on the estimate, and this is what the balance it is read
# under is reported against.
BALANCE_BAR = 0.1

# The cell subsec:appendix_participation_matching reports and the criterion it
# is read against: the univariate specification because that is the primary
# estimand of subsec:reversal_test, and common support because that is the
# stacking the headline is estimated on.
GATE_GRID = "quintile"
GATE_SAMPLE = "common"
GATE_ARM = "matched"
GATE_SPEC = "fama_macbeth_univariate"
GATE_MIN_T = CRIT_5PCT

np.random.seed(MC_SEED)


def build_long(sample, extra_cols=()):
    """Stack the two class panels and attach the matching variables.

    ``zTV`` and ``zMC`` arrive as raw ratios and are standardised later, so
    their absolute values here are the participation rate $|D|/V$ and the
    capacity-normalised magnitude $|D|/M$. Both are strictly positive on the
    nonzero-flow stackings, and turnover is strictly positive throughout the
    panel, so the three logs are defined on every row. ``extra_cols`` names
    further firm-day columns to carry onto both class rows, which is how
    subsec:size_restricted attaches the size quintile it re-estimates within.
    """
    long = load_stacked(sample, extra_cols=(*PANEL_COLS, *extra_cols))
    long["logIntensity"] = np.log(long["zTV"].abs())
    long["logTurnover"] = np.log(long["turnover"])
    long["logCapMagnitude"] = np.log(long["zMC"].abs())
    return long


def cem_weights(long, n_bins):
    """Attach coarsened-exact-matching weights and drop the unmatched rows.

    Bin edges are the pooled within-date quantiles of each axis, so the two
    classes are cut at identical points. ``duplicates="drop"`` collapses tied
    edges on the rare date whose cross-section is degenerate on an axis, which
    coarsens that date further rather than dropping it. Each cell is
    re-balanced to ``min(n_capacity, n_foreign)``, so a cell holding one class
    only leaves the sample at weight zero.
    """
    out = long.copy()
    for col, name in (("logTurnover", "binTau"), ("logIntensity", "binInt")):
        out[name] = out.groupby("date", sort=False)[col].transform(
            lambda s: pd.qcut(s, n_bins, labels=False, duplicates="drop"))

    cell = out.groupby(["date", "binTau", "binInt"], sort=False)["Foreign"]
    n_foreign = cell.transform("sum")
    n_cell = cell.transform("size")
    n_capacity = n_cell - n_foreign
    n_own = np.where(out["Foreign"] == 1, n_foreign, n_capacity)
    out[WEIGHT_COL] = np.minimum(n_capacity, n_foreign) / n_own
    return out[out[WEIGHT_COL] > 0].copy()


def balance(long, weight_col=None):
    """Standardised class differences on ``BALANCE_COLS``.

    Each entry is $(\\bar x_f - \\bar x_c)$ over the root mean of the two class
    variances, the conventional standardised mean difference, so it is read
    against the 0.1 a matched design is normally asked to clear. Moments are
    taken against ``weight_col`` when given, which is what makes the matched
    arm's entry a statement about the sample the contrast is fitted on.
    """
    out = {}
    for col in BALANCE_COLS:
        moments = {}
        for foreign, part in long.groupby("Foreign", sort=True):
            x = part[col].to_numpy(dtype=float)
            w = None if weight_col is None else part[weight_col].to_numpy(dtype=float)
            mean = float(np.average(x, weights=w))
            moments[foreign] = (mean, float(np.average((x - mean) ** 2, weights=w)))
        (m_c, v_c), (m_f, v_f) = moments[0.0], moments[1.0]
        out[col] = {
            "capacity_mean": m_c,
            "foreign_mean": m_f,
            "std_diff": (m_f - m_c) / float(np.sqrt((v_c + v_f) / 2.0)),
        }
    return out


def effective_size(long):
    """Kish effective sample size of the weights, by class.

    $(\\sum w)^2 / \\sum w^2$ is the count of unit-weight rows that carry the
    same precision, so it prices what the re-balancing costs in precision
    where the raw row count does not.
    """
    out = {}
    for foreign, part in long.groupby("Foreign", sort=True):
        w = part[WEIGHT_COL].to_numpy(dtype=float)
        label = "foreign" if foreign else "capacity_bound"
        out[label] = {
            "n_rows": int(len(w)),
            "sum_weights": float(w.sum()),
            "effective_n": float(w.sum() ** 2 / np.square(w).sum()),
        }
    return out


def run_arm(panel, weight_col=None, joint=True):
    """Estimate equation eq:reversal_interaction on ``panel``.

    The univariate specification of subsec:reversal_test is fitted as the
    primary estimand, and with ``joint`` the joint one as the conditioning
    check.
    """
    arm = {
        "n_rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "weighted": weight_col is not None,
        "fama_macbeth_univariate": fama_macbeth_univariate(panel, weight_col=weight_col),
    }
    if joint:
        arm["fama_macbeth"] = fama_macbeth_reversal(panel, weight_col=weight_col)
    return arm


def run_grid(long, n_bins, overlap):
    """Match on one bin grid and estimate the matched arm, and with
    ``overlap`` the overlap arm of the module docstring."""
    matched = cem_weights(long, n_bins)
    panel = standardise_within_day_class(matched, weight_col=WEIGHT_COL)

    arms = {}
    if overlap:
        unweighted = standardise_within_day_class(matched)
        arms["overlap"] = run_arm(unweighted, joint=False)
    arms["matched"] = run_arm(panel, weight_col=WEIGHT_COL)
    arms["balance_matched"] = balance(panel, weight_col=WEIGHT_COL)
    if overlap:
        arms["balance_overlap"] = balance(unweighted)
    arms["effective_size"] = effective_size(panel)
    arms["n_bins"] = n_bins
    return arms


def run_sample(sample):
    """Match and re-estimate the reversal on one stacking of the two panels."""
    print_header(f"Reversal under matched flow intensity — {sample}", width=72)
    long = build_long(sample)
    unrestricted = standardise_within_day_class(long)

    results = {"full": run_arm(unrestricted), "balance_full": balance(unrestricted)}
    print(f"Stacked rows: {len(unrestricted):,} unrestricted")
    for col, stats in results["balance_full"].items():
        print(f"  {col:<16} class difference {stats['std_diff']:+.3f} sd")

    for name, n_bins in GRIDS.items():
        grid = run_grid(long, n_bins,
                        overlap=(sample, name) == (GATE_SAMPLE, GATE_GRID))
        results[name] = grid
        ess = grid["effective_size"]
        print(f"\n  {name} grid ({n_bins} x {n_bins} cells per date)")
        print(f"    retained {grid['matched']['n_rows']:,} rows "
              f"({100 * grid['matched']['n_rows'] / len(unrestricted):.1f}%), "
              f"effective {ess['capacity_bound']['effective_n']:,.0f} capacity "
              f"/ {ess['foreign']['effective_n']:,.0f} foreign")
        for col, stats in grid["balance_matched"].items():
            inside = "inside" if abs(stats["std_diff"]) <= BALANCE_BAR else "outside"
            print(f"    {col:<16} class difference {stats['std_diff']:+.3f} sd"
                  f"  ({inside} {BALANCE_BAR})")
        for arm in ("overlap", "matched"):
            if arm not in grid:
                continue
            uni = grid[arm]["fama_macbeth_univariate"]
            extra = ""
            if "fama_macbeth" in grid[arm]:
                joint = grid[arm]["fama_macbeth"]
                extra = (f", joint {joint['reversal']['mean']:+.6f}"
                         f" at t = {joint['reversal']['t']:+.2f}")
            print(f"    {arm:<9} REVERSAL = {uni['reversal']['mean']:+.6f}"
                  f"  (t = {uni['reversal']['t']:+.2f}{extra})"
                  f"  edges {uni['advantage_capacity']['mean']:+.6f}"
                  f" / {uni['advantage_foreign']['mean']:+.6f}")
    return results


def main():
    """Entry point: match the two flow distributions on both stackings."""
    print_header("Reversal with the two classes' flow intensities matched",
                 level="banner")
    results = {
        "config": {
            "matching_variable": "logIntensity",
            "stratifier": "logTurnover",
            "grids": dict(GRIDS),
            "n_bins": N_BINS,
            "balance_bar": BALANCE_BAR,
            "gate_grid": GATE_GRID,
            "gate_sample": GATE_SAMPLE,
            "gate_arm": GATE_ARM,
            "gate_spec": GATE_SPEC,
            "gate_min_t": GATE_MIN_T,
        },
        "common": run_sample("common"),
        "union": run_sample("union"),
    }

    gate = results[GATE_SAMPLE][GATE_GRID][GATE_ARM][GATE_SPEC]["reversal"]
    results["verdict"] = {
        "reversal": gate["mean"],
        "t": gate["t"],
        "passed": bool(gate["mean"] > 0 and gate["t"] > GATE_MIN_T),
        "criterion": f"REVERSAL > 0 and t > {GATE_MIN_T:.2f} in the {GATE_ARM} "
                     f"arm of the {GATE_GRID} grid, {GATE_SAMPLE} sample, "
                     f"{GATE_SPEC} specification",
    }
    print(f"\nCriterion committed before the run: {results['verdict']['criterion']}")
    print(f"  observed REVERSAL = {gate['mean']:+.6f} at t = {gate['t']:+.2f} "
          f"-> {'pass' if results['verdict']['passed'] else 'fail'}")

    save_json(results, RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()
