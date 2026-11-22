#!/usr/bin/env python3
"""
Joint test of the reversal as a single contrast (subsec:reversal_test).

Table~\\ref{tab:investor_heterogeneity} estimates the two trader classes in
separate panels, so the reversal itself — Capacity-Bound Institutional flow
favouring $S^{MC}$ while foreign flow favours $S^{TV}$ — is read off by
comparing two independently estimated $t$-statistics. The reversal is a
class-by-normalisation interaction, and this script estimates it as one
coefficient with one standard error.

The two class panels are stacked into one panel carrying one row per
(stock-day, class). Signals are standardised within day *and* class, matching
the per-class day-standardisation of every other Fama-MacBeth specification
in the paper. Each signal is entered on its own,

    R_{i,t+1} = a_t + c_t Foreign_i + b z(S) + b^F Foreign x z(S) + e,

so that b is the capacity-class loading on that signal and b + b^F the foreign
loading. Writing b1, b2 for the capacity loadings on z(S^MC), z(S^TV) and
b1+b3, b2+b4 for the foreign ones, the S^MC slope minus the S^TV slope, a
class's advantage, is (b1 - b2) for the capacity class and (b1+b3) - (b2+b4)
for foreign, and the contrast, which the code stores under ``reversal``, is the
difference-in-differences between them:

    REVERSAL = (b1 - b2) - [(b1+b3) - (b2+b4)] = b4 - b3.

Corollary~\\ref{cor:regimes} predicts REVERSAL > 0: the capacity class's
matched-filter advantage runs towards $S^{MC}$ and the foreign class's towards
$S^{TV}$, so the two advantages differ in sign, not merely in size.

Two specifications estimate the same four loadings and the same contrast.
``univariate`` fits the equation above once per signal, so each loading is the
signal's own cross-sectional slope. ``joint`` fits both signals together,

    R_{i,t+1} = a_t + c_t Foreign_i + b1 z(S^MC) + b2 z(S^TV)
                + b3 Foreign x z(S^MC) + b4 Foreign x z(S^TV) + e,

so each loading is conditional on the other signal. The univariate arm is the
primary one because prop:matched_filter ranks the correlation of each filter
with returns, which is the univariate slope of a standardised regressor and
not a partial slope holding the other filter fixed. The joint arm asks the
further question of whether the ordering survives conditioning, and is read
here as a robustness check on the univariate one rather than as the estimand.

Both arms form the per-day advantage and REVERSAL series from that day's
loadings and aggregate the series itself, so each standard error comes from
the time-series dispersion of the contrast rather than from differencing two
separately aggregated standard errors. The univariate contrast spans two
regressions per day, which the Fama-MacBeth time series handles and a single
fitted cross-sectional covariance does not, so the cluster-by-date arm below
is estimated for the joint specification only.

Three samples are estimated. ``union`` stacks each class on its own nonzero-flow
stock-days, the joint counterpart of Table~\\ref{tab:investor_heterogeneity}'s
two panels. ``common`` restricts both classes to the stock-days on which
*both* flows are nonzero, so the two classes are compared on one set of firms,
days, prices, and realised returns rather than on two overlapping sets. That
stacking is the primary identification sample of subsec:reversal_test, and it
is what makes the Introduction's "the same firms, days, and prices" literal.

``zero_included`` drops the nonzero-flow requirement itself, so a class's
non-participation days enter at $S^{MC} = S^{TV} = 0$ instead of leaving the
panel. The subsec:data restriction excludes them on the argument of
subsec:appendix_panel_filters that an exact zero is a record of
non-participation on most such rows rather than a measured
balance of buys against sells, a share code/empirical/robustness/foreign_definition_share.py measures,
and that exclusion is one-sided: it removes roughly two thirds of the
capacity-bound rows and under one percent of the foreign ones, which is
exactly the shape of filter that can manufacture a difference-in-differences.
This arm is what turns that argument into evidence. It is a robustness
specification and not the estimand, because a zero row places a class that
did not trade at the same point of the signal distribution as one that traded
a balanced book.

Inference is Fama-MacBeth over the per-day series (primary, matching the
paper's other return-prediction tests) with pooled OLS under cluster-by-date
standard errors reported alongside for the joint arm of the ``union`` and
``common`` stackings. The ``zero_included``, pre-period and second-day arms
estimate the univariate specification alone, the one the paper reports for
them (subsec:contrast_halves, subsec:skip_day). The pooled specification
absorbs day-by-class fixed effects by demeaning the return within each
(date, class) cell; the regressors are already day-by-class demeaned by the
z-scoring, so by Frisch-Waugh-Lovell its slopes equal those of the explicit
dummy specification.

The last two arms re-estimate the ``common`` and ``union`` stackings on the
pre-period panel, a window disjoint from and preceding the one every other
estimate in this script runs on. The extended battery already carries the
capacity-bound horse race there (subsec:appendix_j_oos), so what these arms
add is the contrast itself and the advantage for foreign flow, which subsec:contrast_halves
reports as the fragile one. The estimator, the stackings, and the
standardisation are the same objects the published window uses; only the panel
differs, so a difference between the two windows cannot come from a difference
in specification.

The two ``*_second_day`` arms re-estimate the ``common`` and ``union``
stackings on the one-day return over [t+1, t+2], the return
code/empirical/return_prediction/skip_day_return_prediction.py builds, in the univariate specification alone.
They measure how long the contrast lasts (subsec:skip_day). They were added on
2026-09-13, after the per-class [t+1, t+2] cells of tab:skip_day had been
read.

Degrees of freedom
------------------
CHOSEN    ``common`` as the identification sample, argued four paragraphs
          above: it is what makes "the same firms, days, and prices" literal.
          ``union`` and ``zero_included`` are estimated beside it, not instead
          of it, and subsec:contrast_halves reports all three.
CHOSEN    the univariate arm as primary and the joint arm as its robustness
          check, on prop:matched_filter ranking correlations rather than
          partial slopes. Both are reported.
CHOSEN    within-(day, class) standardisation of both signals, matching every
          other Fama-MacBeth specification in the paper.
CHOSEN    ``BASE_CLASS`` = capacity_bound against ``INTERACTED_CLASS`` =
          foreign. REVERSAL = b4 - b3 changes sign under the swap and nothing
          else, so the pairing is a parameterisation and not a selection.
CHOSEN    ``RETURN_HORIZON`` = R_t1, the horizon subsec:return_prediction
          fixes for every class. The second-day arms take ``SKIP_COL``, the
          return over [t+1, t+2], with the window code/empirical/return_prediction/skip_day_return_prediction.py
          fixes and tags; chosen after that script's cells were read.
CHOSEN    ``MIN_STOCKS_PER_DAY``, inherited from ``shared_utils.config``,
          where it carries its own tag and the section that fixes it.
MEASURED  every loading, per-class advantage, and REVERSAL reported below, on
          the stock-days each stacking retains.
FITTED    nothing. No constant here is set to reproduce a target statistic,
          so the contrast spends no degree of freedom on the number it
          reports.

BASELINE: Fama-MacBeth over the per-day contrast series | obvious
          alternative: a referee says the standard error is a time-series
          dispersion over 1,200-odd days and a pooled panel regression with
          standard errors clustered by date would not survive it |
          discharge: run. The joint arm is estimated both ways on the same
          rows and subsec:reversal_test reports the cluster-by-date t beside
          the Fama-MacBeth one. The univariate contrast spans two regressions
          per day, which no single fitted cross-sectional covariance covers,
          so the cluster arm is estimated for the joint specification only.

Inputs
------
    results/full_daily_data_all_investors.parquet
    results/full_daily_data_all_investors_pre2020.parquet

Output
------
    results/reversal_interaction_results.json
"""

import numpy as np
import pandas as pd
from shared_utils import (
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    daily_stats_pack,
    ols_fit,
    ols_with_cluster_se,
    print_header,
    save_json,
    wls_fit,
)

from empirical.return_prediction.skip_day_return_prediction import (
    SKIP_COL,
    add_skip_return,
)
from paths import DAILY_DATA_ALL_INVESTORS, DAILY_DATA_PRE2020, RESULTS_DIR

INPUT_FILE = str(DAILY_DATA_ALL_INVESTORS)
# Same panel construction over the window preceding the published one, built by
# code/data/pre_period_pipeline.py. Nothing crosses over from the published
# window, so the pre-period arms are out of sample for every estimate here.
PRE_PERIOD_INPUT = str(DAILY_DATA_PRE2020)
RESULTS_PATH = str(RESULTS_DIR / "reversal_interaction_results.json")

RETURN_HORIZON = "R_t1"
MIN_STOCKS_PER_DAY = MIN_STOCKS_PER_DAY_DEFAULT

# Base class first: its loadings are b1 (S^MC) and b2 (S^TV); the second class
# carries the Foreign indicator and the two interaction terms.
BASE_CLASS = "capacity_bound"
INTERACTED_CLASS = "foreign"

# Design-matrix column order, shared by the daily Fama-MacBeth loop and the
# pooled specification. The pooled fit drops the two absorbed columns.
DESIGN_COLS = ("const", "Foreign", "zMC", "zTV", "F_zMC", "F_zTV")
POOLED_COLS = ("zMC", "zTV", "F_zMC", "F_zTV")

# The univariate arm enters one signal at a time, so each day is fitted twice.
SIGNALS = ("zMC", "zTV")

# REVERSAL = b4 - b3, the difference-in-differences defined in the module
# docstring, as a weight vector over POOLED_COLS.
REVERSAL_WEIGHTS = (0.0, 0.0, -1.0, 1.0)

np.random.seed(MC_SEED)


def load_stacked(sample, extra_cols=(), input_file=INPUT_FILE, return_col=RETURN_HORIZON):
    """Return the long (stock-day, class) frame for ``sample``.

    ``union`` keeps each class's own nonzero-flow stock-days; ``common``
    intersects them so both classes are observed on identical rows;
    ``zero_included`` keeps every stock-day for both classes, so the
    non-participation rows the subsec:data restriction excludes enter at
    $S^{MC} = S^{TV} = 0$. ``extra_cols`` names firm-day columns of the panel
    to carry onto both class rows, which is how subsec:size_restricted and
    subsec:appendix_participation_matching attach their sort and matching
    variables to this stacking instead of re-deriving it.
    ``input_file`` selects the panel; it is ``PRE_PERIOD_INPUT`` for the
    out-of-sample arms and the published panel everywhere else.
    ``return_col`` names the dependent variable. ``SKIP_COL`` is not a panel
    column, so it is built from the adjusted close the way
    code/empirical/return_prediction/skip_day_return_prediction.py builds it.
    """
    built = return_col == SKIP_COL
    cols = ["date", "ticker", "close" if built else return_col, *extra_cols]
    for c in (BASE_CLASS, INTERACTED_CLASS):
        cols += [f"D_{c}", f"S_MC_{c}", f"S_TV_{c}"]
    df = pd.read_parquet(input_file, columns=cols)
    if built:
        df = add_skip_return(df, "ticker")

    if sample == "common":
        keep = (df[f"D_{BASE_CLASS}"] != 0) & (df[f"D_{INTERACTED_CLASS}"] != 0)
        df = df[keep]

    parts = []
    for cls in (BASE_CLASS, INTERACTED_CLASS):
        sub = df if sample == "zero_included" else df[df[f"D_{cls}"] != 0]
        part = sub[["date", "ticker", return_col, *extra_cols]].copy()
        part["zMC"] = sub[f"S_MC_{cls}"].values
        part["zTV"] = sub[f"S_TV_{cls}"].values
        part["Foreign"] = float(cls == INTERACTED_CLASS)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _cell_moments(cell, cols, weight_col):
    """Return the ``(mean, sd)`` each of ``cols`` is standardised by in ``cell``.

    Without ``weight_col`` these are the plain cross-sectional moments, the
    ``pandas`` sample standard deviation included. With one they are taken
    against that column, because a z-score has to be computed on the same
    sample the fit runs on. Under the re-weighting of
    subsec:appendix_participation_matching the two classes' cells are
    re-balanced, and scaling each class's signal by the dispersion of the
    sample it no longer carries would put the two loadings back on different
    units.
    """
    if weight_col is None:
        return {c: (cell[c].mean(), cell[c].std()) for c in cols}
    w = cell[weight_col].to_numpy(dtype=float)
    moments = {}
    for c in cols:
        x = cell[c].to_numpy(dtype=float)
        mean = float(np.average(x, weights=w))
        moments[c] = (mean, float(np.sqrt(np.average((x - mean) ** 2, weights=w))))
    return moments


def standardise_within_day_class(long, cols=SIGNALS, weight_col=None, return_col=RETURN_HORIZON):
    """Day-by-class z-score ``cols``; drop degenerate (date, class) cells.

    A cell is dropped when it holds fewer than ``MIN_STOCKS_PER_DAY`` rows with
    non-missing values and return, or when any standardised column has zero or
    non-finite cross-sectional dispersion — the same pre-processing recipe
    ``shared_utils.prepare_fm_panel`` applies per day to the single-class
    specifications, here applied per (day, class) because each class is
    standardised on its own cross-section.

    ``cols`` defaults to the two signals. Each standardised column also gets
    its ``Foreign``-interacted counterpart ``F_<col>``, so each column enters
    equation eq:reversal_interaction with one slope per class rather than one
    slope pooled across the two. ``weight_col`` names the
    column the cell moments are taken against, and is what
    subsec:appendix_participation_matching passes its matching weights in as.
    ``return_col`` is the dependent variable a row needs to enter a cell.
    """
    cols = tuple(cols)
    long = long.dropna(subset=[*cols, return_col])
    kept = []
    for _, cell in long.groupby(["date", "Foreign"], sort=False):
        if len(cell) < MIN_STOCKS_PER_DAY:
            continue
        moments = _cell_moments(cell, cols, weight_col)
        if any(sd == 0 or not np.isfinite(sd) for _, sd in moments.values()):
            continue
        cell = cell.copy()
        for c in cols:
            mean, sd = moments[c]
            cell[c] = (cell[c] - mean) / sd
        kept.append(cell)
    out = pd.concat(kept, ignore_index=True)

    # Both classes must survive on a date for the interaction to be identified.
    both = out.groupby("date")["Foreign"].nunique() == 2
    out = out[out["date"].isin(both[both].index)].copy()

    for c in cols:
        out[f"F_{c}"] = out["Foreign"] * out[c]
    return out


def _contrast_packs(cap_mc, cap_tv, for_mc, for_tv):
    """FM-aggregate the two per-class advantages and the reversal.

    Arguments are the per-day loading series named in the module docstring:
    b1, b2, b1+b3, b2+b4. Both advantages are taken in the same
    ``S^MC`` minus ``S^TV`` parameterisation, so cor:regimes predicts the
    capacity one positive, the foreign one negative, and their difference
    positive. The contrast is formed per day and aggregated as its own series,
    so its standard error carries the covariance of the two advantages instead
    of ignoring it. For the joint arm that series equals ``b4 - b3`` term by
    term, the contrast of eq:reversal_contrast.
    """
    cap = np.asarray(cap_mc, dtype=float) - np.asarray(cap_tv, dtype=float)
    foreign = np.asarray(for_mc, dtype=float) - np.asarray(for_tv, dtype=float)
    return {
        "advantage_capacity": daily_stats_pack(cap),
        "advantage_foreign": daily_stats_pack(foreign),
        "reversal": daily_stats_pack(cap - foreign),
    }


def _day_fit(y, X, weights):
    """Fit one daily cross-section, weighted only when ``weights`` is given.

    The published specifications pass ``None`` and route to ``ols_fit``
    unchanged. subsec:appendix_participation_matching passes the matching
    weights, and weighted least squares against them is what estimates the
    reversal on the re-balanced cells rather than on the panel's own class
    proportions.
    """
    return ols_fit(y, X) if weights is None else wls_fit(y, X, weights)


def fama_macbeth_univariate(panel, weight_col=None, return_col=RETURN_HORIZON):
    """Per-day OLS entering one signal at a time; FM-average every coefficient.

    Each day is fitted once per signal on ``[Foreign, z(S), Foreign x z(S)]``,
    so the reported loading on each filter is its own cross-sectional slope
    rather than a slope conditional on the other filter. This is the arm
    prop:matched_filter ranks. Coefficients are stored under the joint arm's
    names, so the two arms report the same b1--b4 parameterisation and differ
    only in the conditioning set. ``weight_col`` turns each daily fit into
    weighted least squares against that column, which is how
    subsec:appendix_participation_matching re-estimates the same contrast on
    net-participation-matched cells. ``return_col`` is the dependent variable.
    """
    daily = {c: [] for c in DESIGN_COLS[2:]}
    n_rows = []
    for _, day in panel.groupby("date", sort=True):
        y = day[return_col].to_numpy(dtype=float)
        foreign = day["Foreign"].to_numpy(dtype=float)
        w = None if weight_col is None else day[weight_col].to_numpy(dtype=float)
        for sig in SIGNALS:
            z = day[sig].to_numpy(dtype=float)
            res = _day_fit(y, np.column_stack([foreign, z, foreign * z]), w)
            params = np.asarray(res.params, dtype=float)[1:]  # drop the intercept
            daily[sig].append(params[1])
            daily[f"F_{sig}"].append(params[2])
        n_rows.append(len(day))

    cap_mc, cap_tv = np.array(daily["zMC"]), np.array(daily["zTV"])
    for_mc = cap_mc + np.array(daily["F_zMC"])
    for_tv = cap_tv + np.array(daily["F_zTV"])
    out = {
        "coefficients": {name: daily_stats_pack(vals) for name, vals in daily.items()},
        **_contrast_packs(cap_mc, cap_tv, for_mc, for_tv),
        "n_days": len(n_rows),
        "avg_n_rows": float(np.mean(n_rows)),
    }
    return out


def fama_macbeth_reversal(panel, weight_col=None, return_col=RETURN_HORIZON):
    """Per-day OLS of the joint stacked cross-section; FM-average every coefficient.

    Returns ``{coefficients: {name: {mean, std, se, t}}, advantage_capacity,
    advantage_foreign, reversal, n_days, avg_n_rows}``, the contrast packs
    built by ``_contrast_packs`` from the four conditional loadings.
    ``weight_col`` weights each daily fit and
    ``return_col`` names the dependent variable, as in
    ``fama_macbeth_univariate``.
    """
    columns = DESIGN_COLS[1:]
    daily = {c: [] for c in columns}
    n_rows = []
    for _, day in panel.groupby("date", sort=True):
        X = day[list(DESIGN_COLS[1:])].to_numpy(dtype=float)
        w = None if weight_col is None else day[weight_col].to_numpy(dtype=float)
        res = _day_fit(day[return_col].to_numpy(dtype=float), X, w)
        params = np.asarray(res.params, dtype=float)[1:]  # drop the intercept
        for name, value in zip(columns, params):
            daily[name].append(value)
        n_rows.append(len(day))

    cap_mc, cap_tv = np.array(daily["zMC"]), np.array(daily["zTV"])
    return {
        "coefficients": {name: daily_stats_pack(vals) for name, vals in daily.items()},
        **_contrast_packs(cap_mc, cap_tv,
                          cap_mc + np.array(daily["F_zMC"]),
                          cap_tv + np.array(daily["F_zTV"])),
        "n_days": len(n_rows),
        "avg_n_rows": float(np.mean(n_rows)),
    }


def pooled_reversal(panel, return_col=RETURN_HORIZON):
    """Pooled OLS with cluster-by-date SEs and the contrast b4 - b3.

    Day-by-class fixed effects are absorbed by demeaning the return within
    each (date, class) cell. The regressors are already day-by-class demeaned
    (they are z-scores computed on the same cells), so the reported slopes are
    those of the explicit dummy specification. ``return_col`` names the
    dependent variable.
    """
    cols = POOLED_COLS
    y = panel[return_col] - panel.groupby(["date", "Foreign"])[return_col].transform("mean")
    X = panel[list(cols)].to_numpy(dtype=float)
    fit = ols_with_cluster_se(
        y.to_numpy(dtype=float), X, panel["date"].to_numpy(),
        contrasts={"reversal": REVERSAL_WEIGHTS},
    )
    return {
        "coefficients": {
            name: {"coef": fit["coef"][i], "se": fit["se"][i], "t_stat": fit["t"][i]}
            for i, name in enumerate(POOLED_COLS)
        },
        "reversal": fit["contrasts"]["reversal"],
        "n_obs": int(len(panel)),
        "n_clusters": int(panel["date"].nunique()),
    }


def run_sample(sample, input_file=INPUT_FILE, window="published", return_col=RETURN_HORIZON,
               joint=True):
    """Estimate both specifications on one stacking of the two class panels.

    ``input_file`` and ``window`` select the panel; the pre-period arms pass
    ``PRE_PERIOD_INPUT`` and are otherwise identical to the published ones.
    ``return_col`` selects the dependent variable; the second-day arms pass
    ``SKIP_COL``. ``joint=False`` estimates the primary univariate arm alone,
    the one arm the paper reports for the zero-included, pre-period and
    second-day samples.
    """
    print_header(f"Stacked reversal test — {sample} sample ({window}, {return_col})", width=72)
    panel = standardise_within_day_class(
        load_stacked(sample, input_file=input_file, return_col=return_col),
        return_col=return_col)

    per_class = panel.groupby("Foreign").agg(
        n_rows=("ticker", "size"), n_stocks=("ticker", "nunique"))
    print(f"Stacked rows: {len(panel):,} over {panel['date'].nunique():,} dates")
    for foreign, row in per_class.iterrows():
        label = INTERACTED_CLASS if foreign else BASE_CLASS
        print(f"  {label:<16} {row['n_rows']:>9,} rows  {row['n_stocks']:>5,} stocks")

    arms = {"fama_macbeth_univariate": fama_macbeth_univariate(panel, return_col=return_col)}
    if joint:
        arms["fama_macbeth"] = fama_macbeth_reversal(panel, return_col=return_col)
        arms["pooled_cluster_date"] = pooled_reversal(panel, return_col=return_col)

    for label, key in (("univariate", "fama_macbeth_univariate"), ("joint", "fama_macbeth")):
        if key not in arms:
            continue
        print(f"\n  Fama-MacBeth, {label} specification")
        for name, pack in (("advantage, capacity", "advantage_capacity"),
                           ("advantage, foreign", "advantage_foreign"),
                           ("REVERSAL", "reversal")):
            est = arms[key][pack]
            print(f"    {name:<20} = {est['mean']:+.6f}  (t = {est['t']:+.2f})")
    if joint:
        pooled = arms["pooled_cluster_date"]["reversal"]
        print(f"\n  Cluster-by-date REVERSAL (joint)  = {pooled['value']:+.6f}"
              f"  (t = {pooled['t_stat']:+.2f}, p = {pooled['p_value']:.2e})")

    return {
        "window": window,
        "return_col": return_col,
        # Stamped so an arm cannot be read as out of sample without the dates
        # that make it so.
        "date_first": str(pd.Timestamp(panel["date"].min()).date()),
        "date_last": str(pd.Timestamp(panel["date"].max()).date()),
        "n_rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "n_rows_by_class": {
            BASE_CLASS: int(per_class.loc[0.0, "n_rows"]),
            INTERACTED_CLASS: int(per_class.loc[1.0, "n_rows"]),
        },
        **arms,
    }


def main():
    """Entry point: run the stacked reversal test on each stacking."""
    print_header("The reversal as a single contrast", level="banner")
    results = {
        "config": {
            "return_horizon": RETURN_HORIZON,
            "second_day_return": SKIP_COL,
            "min_stocks_per_day": MIN_STOCKS_PER_DAY,
            "base_class": BASE_CLASS,
            "interacted_class": INTERACTED_CLASS,
            "primary_specification": "univariate",
            "reversal_definition": "the (class x normalisation) "
                                   "difference-in-differences, equal to b4 - b3 "
                                   "in the joint arm; positive under "
                                   "Corollary cor:regimes",
        },
        "union": run_sample("union"),
        "common": run_sample("common"),
        "zero_included": run_sample("zero_included", joint=False),
        # Out-of-sample arms: the same two stackings on the disjoint earlier
        # window, which is what subsec:contrast_halves reads the advantage for foreign flow
        # against.
        "pre_common": run_sample("common", PRE_PERIOD_INPUT, window="pre_period",
                                 joint=False),
        "pre_union": run_sample("union", PRE_PERIOD_INPUT, window="pre_period",
                                joint=False),
        # The same two stackings on the return one day later, which is how
        # long the contrast lasts (subsec:skip_day). Specified after the
        # per-class cells of tab:skip_day were read.
        "common_second_day": run_sample("common", return_col=SKIP_COL, joint=False),
        "union_second_day": run_sample("union", return_col=SKIP_COL, joint=False),
    }
    save_json(results, RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()
