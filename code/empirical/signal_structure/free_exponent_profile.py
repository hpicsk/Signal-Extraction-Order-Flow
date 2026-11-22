#!/usr/bin/env python3
"""The normalisation exponent estimated freely, instead of tested at two points
(subsec:free_exponent).

subsec:threshold_estimate argues the admissible normalisations are the family

    S(theta) = D / (M^(1-theta) V^theta) = S^MC tau^(-theta),

because $D$, $M$ and $V$ are all in currency and only exponents summing to one
leave the ratio invariant to firm scale. That argument fixes the *family*. It
does not license testing only its two endpoints, and the paper tests only its
two endpoints: $S^{MC}$ is $\\theta = 0$ and $S^{TV}$ is $\\theta = 1$. If each
class's return predictability peaked far from the endpoint its regime assigns,
the assignment would be a coarsening of a continuum rather than a reading of
one, and the reversal would be an artifact of where the two points were placed.

This script estimates $\\theta$ freely for each class and reports where the
endpoints sit in the profile.

**The objective is the one prop:matched_filter ranks.** That proposition orders
normalisations by $\\rho(S, R)$, the cross-sectional correlation with returns,
so the profile is the mean daily $\\rho(S(\\theta), R)$ and
$\\hat\\theta = \\arg\\max_\\theta$ of it. A correlation is bounded, which
matters here: a Fama-MacBeth slope on a day-standardised regressor is not, and
its profile runs off to whichever end of the grid degenerates the regressor
most, reporting the shape of the standardisation rather than the fit.

**What is identified and what is not.** As $|\\theta|$ grows the family tends to
a signed power of turnover, and turnover carries a return effect of its own
\\citep{DATAR1998203, Barinov2014}, so the *level* of one class's
$\\hat\\theta$ is not separated from that effect by this design. The *gap*
between the two classes is, by the argument subsec:reversal_test already
makes: turnover's own return effect is a property of the stock-day and loads
both classes' profiles alike, so it moves the two optima together and leaves
their difference where it is. The gap is therefore the estimand and the levels
are read as locations, which is also why the ``common`` arm below re-estimates
both profiles on the stock-days both classes trade.

**The rank arm, and what it can and cannot do.** The two filters are ratios
with heavy-tailed numerators and $\\theta$ changes the tail as it moves, so a
profile in levels could order the family by the shape of the regressor rather
than by its information. The rank arm replaces $S(\\theta)$ by its within-day
normal score, which fixes the marginal distribution at every $\\theta$ by
construction, exactly as subsec:appendix_j_rank does for the two endpoints.
The transform costs the profile its maximum. Ranking $S^{MC}\\tau^{-\\theta}$
keeps only the ordering, and as $|\\theta|$ grows that ordering converges to
one sorted first on the sign of the flow and then on turnover, so the rank
profile flattens towards a limit at both ends of any grid instead of turning
over. An arg-max on it is a report of where a flat region happens to peak.
The rank arm therefore checks the *endpoint comparison* with the tails held
fixed, which is a statement about $\\theta \\in \\{0, 1\\}$ and needs no
maximum, and ``theta_hat`` is recorded only where the arg-max is interior to
the grid. ``argmax_at_edge`` travels with every profile so the distinction is
in the output rather than in a reader's memory.

Degrees of freedom
------------------
CHOSEN    the family $S^{MC}\\tau^{-\\theta}$, the scale-invariant line
          subsec:threshold_estimate derives. Nothing here relaxes it; what is
          relaxed is the restriction of $\\theta$ to $\\{0, 1\\}$.
CHOSEN    the grid ``THETA_GRID``, wide enough that both classes' level
          profiles turn over inside it, and its step, which sets the
          resolution of every reported $\\hat\\theta$.
CHOSEN    mean daily $\\rho(S, R)$ as the objective, for the reason above.
CHOSEN    $R_{t1}$ as the return horizon, as elsewhere in the paper.
MEASURED  $\\hat\\theta$ per class and arm, and the gap between the two, with
          day-block bootstrap intervals.
BASELINE: the two-point comparison of tab:investor_heterogeneity | obvious
          alternative: a referee says the dichotomy cuts a continuum in two
          and asks for $\\theta$ estimated per class | discharge: this script,
          reported in subsec:free_exponent.

Provenance
----------
Post-hoc. The objection was raised against the submitted draft and this
measurement answers it; no criterion was published in advance. What the run is
asked for is the sign of the gap $\\hat\\theta_{\\text{foreign}} -
\\hat\\theta_{\\text{capacity}}$, which cor:regimes predicts positive, and the
location of each assigned endpoint in its own class's profile. Both are
reported as they fall.

Arms
----
``own``     each class on its own nonzero-flow stock-days, the panels
            tab:investor_heterogeneity reports.
``common``  both classes on the stock-days both trade, the identification
            sample of subsec:reversal_test, where the gap is the free-exponent
            counterpart of the reversal.
``own`` carries a ``levels`` and a ``ranks`` profile, and ``common`` the
``levels`` profile alone, the one subsec:free_exponent reports on common
support.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/free_exponent_profile.json
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from shared_utils import (
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    print_header,
    save_json,
)

from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR

INPUT_FILE = str(DAILY_DATA_ALL_INVESTORS)
RESULTS_PATH = str(RESULTS_DIR / "free_exponent_profile.json")

RETURN_HORIZON = "R_t1"
MIN_STOCKS_PER_DAY = MIN_STOCKS_PER_DAY_DEFAULT

CLASSES = ("capacity_bound", "foreign")

# Exponent grid. Both classes' level profiles turn over inside it, which is the
# condition an arg-max on a grid has to meet before it reports a number; the
# step is the resolution of every theta reported from it. EDGE_MARGIN is how
# close to an end an arg-max may sit and still be read as one.
THETA_STEP = 0.1
THETA_GRID = np.round(np.arange(-3.0, 4.0 + THETA_STEP / 2, THETA_STEP), 2)
EDGE_MARGIN = 3

# The two members of the family the paper tests, and the one cor:regimes
# assigns to each class.
ENDPOINT_MC, ENDPOINT_TV = 0.0, 1.0
ASSIGNED_ENDPOINT = {"capacity_bound": ENDPOINT_MC, "foreign": ENDPOINT_TV}

N_BOOTSTRAP = 2000

rng = np.random.default_rng(MC_SEED)


def _normal_scores(x):
    """Within-day van der Waerden score, the transform subsec:appendix_j_rank uses."""
    return norm.ppf(rankdata(x) / (len(x) + 1.0))


def _corr(x, y):
    """Pearson correlation, returning NaN on a degenerate cross-section."""
    xc, yc = x - x.mean(), y - y.mean()
    nx, ny = np.sqrt((xc ** 2).sum()), np.sqrt((yc ** 2).sum())
    return float(xc @ yc / (nx * ny)) if nx > 0 and ny > 0 else np.nan


def daily_profiles(panel, cls, ranks=True):
    """Per-day $\\rho(S(\\theta), R)$ over the grid, in levels and, with
    ``ranks``, in ranks.

    Returns ``{levels: (n_days, n_theta), ranks: (n_days, n_theta)}``, without
    ``ranks`` when it is not set. The
    family is built in logs, $\\mathrm{sign}(S^{MC})\\exp(\\log|S^{MC}| -
    \\theta\\log\\tau)$, so the tails of the grid neither overflow nor lose
    precision; the sign is carried separately because the exponent acts on the
    magnitude alone.
    """
    lv, rk = [], []
    for _, day in panel.groupby("date", sort=True):
        d = day[[f"S_MC_{cls}", "turnover", RETURN_HORIZON]].dropna()
        d = d[(d[f"S_MC_{cls}"] != 0) & (d["turnover"] > 0)]
        if len(d) < MIN_STOCKS_PER_DAY:
            continue
        s = d[f"S_MC_{cls}"].to_numpy(dtype=float)
        log_tau = np.log(d["turnover"].to_numpy(dtype=float))
        y = d[RETURN_HORIZON].to_numpy(dtype=float)
        sign, log_mag = np.sign(s), np.log(np.abs(s))
        row_lv, row_rk = [], []
        for theta in THETA_GRID:
            sig = sign * np.exp(log_mag - theta * log_tau)
            row_lv.append(_corr(sig, y))
            if ranks:
                row_rk.append(_corr(_normal_scores(sig), y))
        lv.append(row_lv)
        rk.append(row_rk)
    out = {"levels": np.asarray(lv, dtype=float)}
    if ranks:
        out["ranks"] = np.asarray(rk, dtype=float)
    return out


def _theta_hat(mat):
    """Grid arg-max of the mean daily correlation."""
    return float(THETA_GRID[int(np.nanargmax(np.nanmean(mat, axis=0)))])


def summarise(mat, cls):
    """Point estimate, bootstrap interval, and where the two endpoints sit.

    ``theta_hat`` is ``None`` when the arg-max sits within ``EDGE_MARGIN`` grid
    steps of either end. A profile that is still climbing at the edge has no
    maximum inside the family the grid covers, and the rank profiles do not
    have one anywhere, so the location reported would be the width of the grid
    rather than a property of the data. ``argmax_at_edge`` and the two
    end-of-grid correlations are recorded either way.
    """
    mean = np.nanmean(mat, axis=0)
    se = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0])
    t = mean / se
    i_hat = int(np.nanargmax(mean))
    at_edge = i_hat < EDGE_MARGIN or i_hat >= len(THETA_GRID) - EDGE_MARGIN
    boot = np.asarray([_theta_hat(mat[rng.integers(0, mat.shape[0], mat.shape[0])])
                       for _ in range(N_BOOTSTRAP)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    endpoints = {}
    for name, theta in (("mc", ENDPOINT_MC), ("tv", ENDPOINT_TV)):
        j = int(np.argmin(np.abs(THETA_GRID - theta)))
        endpoints[name] = {
            "theta": float(THETA_GRID[j]),
            "rho": float(mean[j]),
            "t": float(t[j]),
            "share_of_max": float(mean[j] / mean[i_hat]),
        }
    assigned = "mc" if ASSIGNED_ENDPOINT[cls] == ENDPOINT_MC else "tv"
    return {
        "theta_hat": None if at_edge else float(THETA_GRID[i_hat]),
        "theta_at_max": float(THETA_GRID[i_hat]),
        "argmax_at_edge": bool(at_edge),
        "rho_at_grid_ends": [float(mean[0]), float(mean[-1])],
        "theta_ci": [float(lo), float(hi)],
        "rho_max": float(mean[i_hat]),
        "t_at_max": float(t[i_hat]),
        "n_days": int(mat.shape[0]),
        "endpoints": endpoints,
        "assigned_endpoint": assigned,
        "assigned_share_of_max": endpoints[assigned]["share_of_max"],
        "assigned_beats_other": bool(
            endpoints[assigned]["rho"]
            > endpoints["tv" if assigned == "mc" else "mc"]["rho"]),
        "profile": {"theta": THETA_GRID.tolist(),
                    "rho": mean.tolist(), "t": t.tolist()},
    }


def gap_summary(mats, arm_name):
    """Bootstrap the gap $\\hat\\theta_{foreign} - \\hat\\theta_{capacity}$.

    Days are resampled independently for the two classes on the ``own`` arm,
    where the two panels are different row sets, and jointly on ``common``,
    where a day is one shared cross-section and the two profiles are estimated
    on the same rows.
    """
    cap, forg = mats["capacity_bound"], mats["foreign"]
    joint = arm_name == "common"
    draws = []
    for _ in range(N_BOOTSTRAP):
        ic = rng.integers(0, cap.shape[0], cap.shape[0])
        i_f = ic if joint and cap.shape[0] == forg.shape[0] else \
            rng.integers(0, forg.shape[0], forg.shape[0])
        draws.append(_theta_hat(forg[i_f]) - _theta_hat(cap[ic]))
    draws = np.asarray(draws, dtype=float)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    point = _theta_hat(forg) - _theta_hat(cap)
    return {
        "gap": point,
        "ci": [float(lo), float(hi)],
        "share_positive": float((draws > 0).mean()),
        "paired_days": bool(joint and cap.shape[0] == forg.shape[0]),
    }


def main():
    print_header("Free normalisation exponent D / (M^(1-theta) V^theta)",
                 level="banner")
    cols = ["date", "ticker", RETURN_HORIZON, "turnover",
            *[f"{p}_{c}" for c in CLASSES for p in ("D", "S_MC")]]
    panel = pd.read_parquet(INPUT_FILE, columns=cols)
    common = panel[(panel["D_capacity_bound"] != 0) & (panel["D_foreign"] != 0)]

    results = {
        "config": {
            "return_horizon": RETURN_HORIZON,
            "min_stocks_per_day": MIN_STOCKS_PER_DAY,
            "theta_grid": {"lo": float(THETA_GRID[0]), "hi": float(THETA_GRID[-1]),
                           "step": THETA_STEP, "n": int(len(THETA_GRID))},
            "endpoints": {"mc": ENDPOINT_MC, "tv": ENDPOINT_TV},
            "n_bootstrap": N_BOOTSTRAP,
            "seed": MC_SEED,
        },
        "arms": {},
    }

    for arm in ("own", "common"):
        print_header(f"arm: {arm}", level="section")
        mats = {}
        block = {}
        for cls in CLASSES:
            rows = common if arm == "common" else panel[panel[f"D_{cls}"] != 0]
            prof = daily_profiles(rows, cls, ranks=arm == "own")
            mats[cls] = prof
            block[cls] = {}
            for kind in prof:
                s = summarise(prof[kind], cls)
                block[cls][kind] = s
                ep = s["endpoints"]
                theta = ("edge" if s["argmax_at_edge"]
                         else f"{s['theta_hat']:+.2f}")
                print(f"  {cls:15s} [{kind:6s}] theta_hat {theta:>5s} "
                      f"CI [{s['theta_ci'][0]:+.2f}, {s['theta_ci'][1]:+.2f}]"
                      f"   rho_max {s['rho_max']:+.5f}"
                      f"   MC {ep['mc']['share_of_max'] * 100:5.1f}%"
                      f"   TV {ep['tv']['share_of_max'] * 100:5.1f}%")
        # The gap is an arg-max contrast, so it is formed on the levels
        # profiles alone; the rank profiles have no maximum to difference.
        g = gap_summary({c: mats[c]["levels"] for c in CLASSES}, arm)
        block["gap_levels"] = g
        print(f"  gap [levels] {g['gap']:+.2f} "
              f"CI [{g['ci'][0]:+.2f}, {g['ci'][1]:+.2f}] "
              f"P(>0) = {g['share_positive']:.3f}")
        results["arms"][arm] = block

    save_json(results, RESULTS_PATH)


if __name__ == "__main__":
    main()
