#!/usr/bin/env python3
"""The three common-support perturbations re-estimated below the size boundary
(subsec:size_restricted).

Three separate robustness designs weaken the foreign half of the reversal on
common support and only on common support: the within-day rank transform
(subsec:appendix_j_rank), the flow-intensity matching
(subsec:appendix_participation_matching), and the free normalisation exponent
(subsec:free_exponent). Read as three independent failures they price the
foreign half as fragile on three axes. There is a cheaper reading available,
and the paper already carries the evidence for it.
sec:appendix_liquidity_stratified reports that the foreign advantage is absent
in the largest market-cap quintile and that common support places most of its
stock-days in that quintile against roughly a fifth for the foreign class's
own panel, and subsubsec:kospi200_falsification reports that foreign $S^{TV}$
predicts returns negatively inside the KOSPI 200. Common support is therefore
not neutral on the size axis: it concentrates foreign flow into the size range
where the foreign result is already documented to be absent. If
that concentration is what the three designs are picking up, the three
failures are one boundary counted three times.

This script separates the two readings by re-estimating all three designs
inside the size distribution, holding common support fixed throughout.

**The size arms.** ``small`` is the bottom ``N_SMALL_QUINTILES`` within-day
market-cap quintiles, ``large`` the complement, ``all`` the unrestricted
stacking. ``all`` must reproduce ``results/rank_transform_robustness.json`` and
``results/participation_matching_results.json`` cell for cell; a mismatch means the
row filters drifted between this script and those two. The complement is
estimated because a boundary is a two-sided statement: a foreign half that
survives below the cut and is absent above it locates the failure, and one
that is weak on both sides does not.

**Why a count-matched control is required.** The ``small`` arm holds under a
fifth of the common-support rows, so its daily cross-sections are roughly a
fifth the size and every Fama-MacBeth standard error widens accordingly. A
foreign half that fails to appear there would then be uninterpretable: the
design might simply be unable to resolve it. The ``resample`` block draws
``N_RESAMPLE`` random subsamples of the *unrestricted* common stacking,
matched date by date to the ``small`` arm's stock-day count, and re-estimates
the same contrast on each. It answers the one question the size arms cannot
answer about themselves — what this contrast looks like at that sample size
and at the published size composition — so a difference between ``small`` and
that distribution is a statement about size rather than about $n$.

Degrees of freedom
------------------
CHOSEN    ``N_SMALL_QUINTILES`` = 3 of 5, the split the objection names. The
          quintile grid itself is the one sec:appendix_liquidity_stratified
          already reports the size boundary on, so the cut point is inherited
          rather than searched; no other split was estimated.
CHOSEN    the quintile grid of subsec:appendix_participation_matching for the
          matching arm, and its ``matched`` cell, which is the cell that
          section reports. The decile grid is not re-run here.
CHOSEN    levels only for the exponent profile. A rank profile has no maximum
          to locate (subsec:free_exponent), so a size arm of it would report
          the width of the grid.
MEASURED  every contrast below, on the stock-days each size arm retains.
BASELINE: the unrestricted common-support stacking | obvious alternative: a
          referee says common support is adversely selected on size for the
          foreign class, so the three failures are its own size composition |
          discharge: this script, reported in subsec:size_restricted.

Provenance
----------
Post-hoc. The objection was raised against the built draft and this
measurement answers it. One cell was seen before the script was written: the
``small`` arm's ``zscore`` foreign advantage, computed in an unsaved preliminary run
to confirm the restricted cross-sections clear ``MIN_STOCKS_PER_DAY`` at all. The
rank, matched, exponent, ``large``, and ``resample`` cells were not, and the
reading rule below was fixed before any of them was estimated.

Reading rule
------------
Committed before the run. Each of the three designs restores the foreign half
below the boundary when its ``small`` cell carries the sign cor:regimes
predicts, a negative foreign advantage, and clears the paper's 5% two-sided
bar ``GATE_MIN_T``; for the exponent profile, when the foreign optimum is
interior to the grid and the class gap excludes zero. The three readings are
then

``boundary``    all three restored on ``small``, and the ``large`` arm's
                unperturbed foreign advantage absent at the same bar. The
                three failures are the documented size boundary.
``independent`` at least one not restored on ``small``, where the
                ``resample`` distribution shows the contrast is resolvable at
                that sample size. That design's failure is not size.
``bound``       at least one not restored on ``small`` and the ``resample``
                distribution does not resolve the contrast there. The
                restricted sample cannot answer, and the cell ships as a
                bound.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/size_restricted_common.json
"""

import numpy as np
import pandas as pd

from shared_utils import MC_SEED, print_header, save_json
from shared_utils.data import SIZE_QUINTILE_LABELS
from shared_utils.regression import CRIT_5PCT

from empirical.return_prediction.reversal_interaction import (
    RETURN_HORIZON,
    fama_macbeth_univariate,
    standardise_within_day_class,
)
from empirical.robustness.participation_matching import (
    GRIDS,
    WEIGHT_COL,
    balance,
    build_long,
    cem_weights,
    effective_size,
    run_arm,
)
from empirical.robustness.rank_transform_robustness import rank_within
from empirical.signal_structure.free_exponent_profile import (
    CLASSES,
    INPUT_FILE,
    daily_profiles,
    gap_summary,
    summarise,
)
from paths import RESULTS_DIR

RESULTS_PATH = str(RESULTS_DIR / "size_restricted_common.json")

# Firm-day column the size arms cut on, carried onto both class rows of the
# stacking. Assigned by the panel pipeline as a within-day quantile of M.
SIZE_COL = "size_quintile"

# Where the cut falls. sec:appendix_liquidity_stratified reports the foreign
# advantage absent in the top quintile of this same grid, so the split is
# inherited from the boundary already published rather than searched over here.
N_SMALL_QUINTILES = 3
SIZE_ARMS = {
    "all": SIZE_QUINTILE_LABELS,
    "small": SIZE_QUINTILE_LABELS[:N_SMALL_QUINTILES],
    "large": SIZE_QUINTILE_LABELS[N_SMALL_QUINTILES:],
}

# The matching cell subsec:appendix_participation_matching reports.
MATCH_GRID = "quintile"

# Count-matched draws from the unrestricted stacking. Each is a full
# re-estimation, and the block is read as a location for the ``small`` arm
# rather than as an interval on it, so the count is set for a stable mean and
# tail share rather than for a percentile of the extreme.
N_RESAMPLE = 40

# The paper's 5% two-sided bar, the one every gate in this repo is read at.
GATE_MIN_T = CRIT_5PCT

# The contrast the reading rule is stated on. cor:regimes puts the foreign
# class's advantage towards $S^{TV}$, so its $S^{MC}$-minus-$S^{TV}$ advantage
# is predicted negative.
FOREIGN_KEY = "advantage_foreign"

rng = np.random.default_rng(MC_SEED)


def size_mask(frame, arm):
    """Rows of ``frame`` inside size arm ``arm``."""
    return frame[SIZE_COL].isin(SIZE_ARMS[arm])


def restored(contrast):
    """Whether a foreign-advantage cell carries the predicted sign at the bar."""
    return bool(contrast["mean"] < 0 and contrast["t"] < -GATE_MIN_T)


# --------------------------------------------------------------------------
# Design 1 — the reversal in levels and on within-day ranks
# --------------------------------------------------------------------------

def reversal_arms(long):
    """Estimate eq:reversal_interaction on ``long`` in both transforms, under
    the univariate specification that tab:size_restricted prints.

    The size restriction is applied by the caller, so the ranks and the
    z-scores are both formed on the cross-section the regression is fitted on
    rather than on the unrestricted one it was drawn from. That ordering is
    what makes the ``rank`` cell a statement about the restricted sample and
    not a re-weighting of the full one.
    """
    long = long.dropna(subset=["zMC", "zTV", RETURN_HORIZON])
    out = {}
    for arm, frame in (("zscore", long),
                       ("rank", rank_within(long, ["zMC", "zTV"], ["date", "Foreign"]))):
        panel = standardise_within_day_class(frame)
        uni = fama_macbeth_univariate(panel)
        out[arm] = {
            "n_rows": int(len(panel)),
            "n_dates": int(panel["date"].nunique()),
            "fama_macbeth_univariate": uni,
        }
    return out


# --------------------------------------------------------------------------
# Design 2 — the reversal with the two classes' flow intensities matched
# --------------------------------------------------------------------------

def matching_arm(long):
    """Coarsened exact matching and the reversal on the re-balanced cells.

    Bin edges are cut inside the restricted sample, so a turnover or intensity
    bin is a rank among the stock-days the arm keeps. Matching on the full
    stacking and restricting afterwards would leave the cells balanced against
    a population the arm no longer holds.
    """
    matched = cem_weights(long, GRIDS[MATCH_GRID])
    panel = standardise_within_day_class(matched, weight_col=WEIGHT_COL)
    arm = run_arm(panel, weight_col=WEIGHT_COL, joint=False)
    return {
        "matched": arm,
        "balance_matched": balance(panel, weight_col=WEIGHT_COL),
        "effective_size": effective_size(panel),
    }


# --------------------------------------------------------------------------
# Design 3 — the normalisation exponent estimated freely
# --------------------------------------------------------------------------

def exponent_arm(rows):
    """Profile $\\rho(S(\\theta), R)$ per class and bootstrap the class gap.

    Both classes are profiled on the same restricted stock-days, so the gap is
    the free-exponent counterpart of the reversal on that sample. ``common``
    is passed to ``gap_summary`` because every arm here is common support, so
    a bootstrap day is one shared cross-section for the two classes.
    """
    mats = {cls: daily_profiles(rows, cls, ranks=False)["levels"] for cls in CLASSES}
    out = {cls: summarise(mats[cls], cls) for cls in CLASSES}
    out["gap_levels"] = gap_summary(mats, "common")
    return out


# --------------------------------------------------------------------------
# The count-matched control
# --------------------------------------------------------------------------

def resample_control(long, small_counts, observed):
    """Re-estimate the foreign advantage on count-matched random stackings.

    Stock-days are drawn without replacement inside each date, to the count the
    ``small`` arm holds on that date, from the unrestricted common-support
    stock-days. Sampling runs on the stock-day rather than on the stacked row
    so both classes travel together and the draw stays a common-support sample.
    The ``zscore`` univariate specification is the one estimated, because it is
    the cell all three designs perturb.

    ``observed`` carries the ``small`` arm's own estimates, which are located
    in the draw distribution rather than tested against it: the draws are
    subsamples of a population the ``small`` arm is one non-random part of, so
    the share of draws it falls outside says where the restriction moved the
    estimate, not how improbable it is.
    """
    stock_days = long[["date", "ticker"]].drop_duplicates().copy()
    quota = stock_days["date"].map(small_counts).fillna(0).to_numpy()
    draws = []
    for _ in range(N_RESAMPLE):
        # Order each date at random and keep its first ``quota`` stock-days,
        # which is a draw without replacement inside the date.
        order = np.argsort(rng.random(len(stock_days)), kind="stable")
        shuffled = stock_days.iloc[order]
        keep = shuffled.groupby("date", sort=False).cumcount() < quota[order]
        panel = standardise_within_day_class(
            long.merge(shuffled[keep.to_numpy()], on=["date", "ticker"])
                .dropna(subset=["zMC", "zTV", RETURN_HORIZON]))
        fit = fama_macbeth_univariate(panel)
        draws.append({key: {"mean": fit[key]["mean"], "t": fit[key]["t"]}
                      for key in (FOREIGN_KEY, "advantage_capacity", "reversal")})

    out = {"n_draws": N_RESAMPLE}
    for key in (FOREIGN_KEY, "advantage_capacity", "reversal"):
        means = np.array([d[key]["mean"] for d in draws], dtype=float)
        ts = np.array([d[key]["t"] for d in draws], dtype=float)
        obs = float(observed[key]["mean"])
        out[key] = {
            "mean_of_means": float(means.mean()),
            "sd_of_means": float(means.std(ddof=1)),
            "range_of_means": [float(means.min()), float(means.max())],
            "mean_t": float(ts.mean()),
            "share_significant": float((np.abs(ts) > GATE_MIN_T).mean()),
            "small_arm_mean": obs,
            "small_arm_outside_range": bool(obs < means.min() or obs > means.max()),
            "small_arm_z": float((obs - means.mean()) / means.std(ddof=1)),
        }
    return out


def main():
    """Entry point: re-estimate the three designs inside the size distribution."""
    print_header("Common-support perturbations inside the size distribution",
                 level="banner")

    long = build_long("common", extra_cols=(SIZE_COL,))
    panel = pd.read_parquet(
        INPUT_FILE,
        columns=["date", "ticker", RETURN_HORIZON, "turnover", SIZE_COL,
                 *[f"{p}_{c}" for c in CLASSES for p in ("D", "S_MC")]])
    common = panel[(panel["D_capacity_bound"] != 0) & (panel["D_foreign"] != 0)]

    results = {
        "config": {
            "return_horizon": RETURN_HORIZON,
            "size_arms": {k: list(v) for k, v in SIZE_ARMS.items()},
            "n_small_quintiles": N_SMALL_QUINTILES,
            "match_grid": MATCH_GRID,
            "n_resample": N_RESAMPLE,
            "gate_min_t": GATE_MIN_T,
            "seed": MC_SEED,
        },
        "composition": {},
        "reversal": {},
        "matching": {},
        "free_exponent": {},
    }
    for arm in SIZE_ARMS:
        print_header(f"size arm: {arm}", width=72)
        arm_long = long[size_mask(long, arm)].copy()
        arm_rows = common[size_mask(common, arm)]

        # The composition counts common support, the stock-days both classes
        # trade that carry the next-day return every fit here needs, so the
        # arms' counts add up to the count subsec:reversal_test prints.
        support = arm_rows[arm_rows[RETURN_HORIZON].notna()]
        stacked = int(arm_long[RETURN_HORIZON].notna().sum())
        results["composition"][arm] = {
            "n_stacked_rows": stacked,
            "n_stock_days": int(len(support)),
            "share_of_common": float(stacked / long[RETURN_HORIZON].notna().sum()),
            "median_stocks_per_day": float(
                support.groupby("date").size().median()),
        }
        results["reversal"][arm] = reversal_arms(arm_long)
        results["matching"][arm] = matching_arm(arm_long)
        results["free_exponent"][arm] = exponent_arm(arm_rows)

        comp = results["composition"][arm]
        print(f"  {comp['n_stock_days']:,} stock-days "
              f"({100 * comp['share_of_common']:.1f}% of common support), "
              f"median {comp['median_stocks_per_day']:.0f} per day")
        for label, cell in (
                ("levels", results["reversal"][arm]["zscore"]["fama_macbeth_univariate"]),
                ("ranks", results["reversal"][arm]["rank"]["fama_macbeth_univariate"]),
                ("matched", results["matching"][arm]["matched"]["fama_macbeth_univariate"])):
            print(f"    {label:<8} capacity {cell['advantage_capacity']['mean'] * 1e4:+7.2f} bp"
                  f" (t {cell['advantage_capacity']['t']:+6.2f})"
                  f"   foreign {cell[FOREIGN_KEY]['mean'] * 1e4:+7.2f} bp"
                  f" (t {cell[FOREIGN_KEY]['t']:+6.2f})"
                  f"   REVERSAL {cell['reversal']['mean'] * 1e4:+7.2f} bp"
                  f" (t {cell['reversal']['t']:+6.2f})")
        fex = results["free_exponent"][arm]
        for cls in CLASSES:
            s = fex[cls]
            theta = "edge" if s["argmax_at_edge"] else f"{s['theta_hat']:+.2f}"
            print(f"    theta {cls:<15} {theta:>5s} "
                  f"CI [{s['theta_ci'][0]:+.2f}, {s['theta_ci'][1]:+.2f}]"
                  f"   rho_max {s['rho_max']:+.5f}")
        g = fex["gap_levels"]
        print(f"    gap {g['gap']:+.2f} CI [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}]"
              f"  P(>0) = {g['share_positive']:.3f}")

    print_header(f"count-matched control ({N_RESAMPLE} draws)", width=72)
    small_counts = common[size_mask(common, "small")].groupby("date").size()
    results["resample"] = resample_control(
        long, small_counts,
        results["reversal"]["small"]["zscore"]["fama_macbeth_univariate"])
    ctrl = results["resample"][FOREIGN_KEY]
    print(f"  foreign advantage over draws: {ctrl['mean_of_means'] * 1e4:+.2f} bp "
          f"(sd {ctrl['sd_of_means'] * 1e4:.2f}), mean t {ctrl['mean_t']:+.2f}, "
          f"{100 * ctrl['share_significant']:.0f}% of draws clear the bar")
    print(f"  small arm {ctrl['small_arm_mean'] * 1e4:+.2f} bp, "
          f"{ctrl['small_arm_z']:+.1f} sd from the draw mean, "
          f"{'outside' if ctrl['small_arm_outside_range'] else 'inside'} their range "
          f"[{ctrl['range_of_means'][0] * 1e4:+.2f}, "
          f"{ctrl['range_of_means'][1] * 1e4:+.2f}]")

    small = results["reversal"]["small"]
    fex_small = results["free_exponent"]["small"]
    checks = {
        "rank": restored(small["rank"]["fama_macbeth_univariate"][FOREIGN_KEY]),
        "matched": restored(
            results["matching"]["small"]["matched"]["fama_macbeth_univariate"][FOREIGN_KEY]),
        "free_exponent": bool(
            fex_small["foreign"]["theta_hat"] is not None
            and fex_small["gap_levels"]["ci"][0] > 0),
    }
    large_absent = not restored(
        results["reversal"]["large"]["zscore"]["fama_macbeth_univariate"][FOREIGN_KEY])
    resolvable = bool(results["resample"][FOREIGN_KEY]["share_significant"] > 0.5)
    reading = ("boundary" if all(checks.values()) and large_absent
               else "independent" if resolvable else "bound")
    results["verdict"] = {
        "restored": checks,
        "large_arm_absent": large_absent,
        "resample_resolves": resolvable,
        "reading": reading,
    }
    print(f"\n  restored on the small arm: "
          + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in checks.items()))
    print(f"  foreign advantage absent on the large arm: "
          f"{'yes' if large_absent else 'no'}")
    print(f"  reading: {reading}")

    save_json(results, RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()
