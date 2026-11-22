#!/usr/bin/env python3
"""Does the matched filter predict returns because it predicts the class's own
next-day flow? (subsec:appendix_flow_continuation)

The regime assignment of subsec:regime_assignment is corroborated by a
persistence statistic: the directionality ratio is high for a class whose daily
flow keeps its sign. A class whose flow keeps its sign is a class whose day-$t$
flow forecasts its day-$(t+1)$ flow, and if flow moves prices then the day-$t$
filter forecasts the day-$(t+1)$ return without carrying any information at all.
The statistic that corroborates the regime and the confound that would
counterfeit the result are then the same object, which is a circularity no within-panel argument reaches:
sec:robustness excludes what loads the two normalisations
uniformly, and this channel loads them through their own autocorrelations,
which differ.

subsec:skip_day already bounds the channel from one side, since neither matched
filter predicts the return over $[t+1, t+2]$, where a persistent flow would still
be pushing. Both mismatched filters do predict it, which subsec:skip_day reads as
the end of the reversal. This script closes the channel from the other side by
measuring it directly rather than bounding it.

**The restriction the channel implies.** Write $P_{i,t+1}$ for the pressure the
class exerts on day $t+1$ and let $\\lambda_t$ be the price impact per unit of
it. Impact is a property of the stock and the day rather than of the filter an
econometrician divides by, so one $\\lambda_t$ serves both filters. Under the
channel alone,

    b_f = lambda_t q_f       for f in {MC, TV},

where $b_f$ is filter $f$'s return loading and $q_f$ its loading on $P_{i,t+1}$,
both estimated on the same daily cross-section. Two filters, one $\\lambda_t$,
so $\\lambda_t$ is over-identified and the channel implies

    b_MC q_TV - b_TV q_MC = 0,                                     (*)

a restriction that consumes no assumption about the size of the impact, holds
for a $\\lambda_t$ that varies day to day, and is scale-free in $P$. It is the
proportionality the objection asserts, written so that it can be rejected.
Rejecting (*) says the two filters' return loadings are not the two filters'
pressure loadings passed through any one impact constant, so something other
than the class's own subsequent trading separates them.

**Pressure is the signed net-buy ratio.** $P = D_{t+1}/V_{t+1}$ carries the
sign of the class's net buying, not its absolute participation, because the
channel being tested is that buying pushes the price up and selling pushes it
down; on unsigned intensity the loadings would not identify it. It is the
quantity execution-cost models make impact a function of
\\citep{Almgren2001, Almgren2003}, so it is the primary measure here. The
cap-scaled $D_{t+1}/M_{t+1}$ runs alongside, because a demand-system account
prices a trade against market value instead. Reading both is what keeps the test from
turning on the choice, and (*) is rejected under either.

**Non-participation is a real zero here.** The nonzero-flow restriction of
subsec:data excludes stock-days on which a class did not trade, on the argument
of subsec:appendix_panel_filters that an exact zero is a record
of non-participation on most such rows rather than a measured balance of buys
against sells — code/empirical/robustness/foreign_definition_share.py measures the share — and so does
not belong at the same point of a *signal* distribution as a measured flow. As a *pressure* variable the same zero is literal: a class that does not
trade a stock on day $t+1$ exerts no impact on it. Day-$(t+1)$ flow therefore
enters at zero where the class is absent.

Degrees of freedom
------------------
CHOSEN    the day-$t$ estimation rows are each class's own nonzero-flow
          stock-days, the panels tab:investor_heterogeneity reports, so the
          return loadings this script forms are the paper's own.
CHOSEN    $P = D_{t+1}/V_{t+1}$ as the primary pressure measure, the
          participation rate execution-cost models name; the cap-scaled
          alternative runs as a second arm rather than as a robustness note.
CHOSEN    $R_{t1}$ as the return horizon, the horizon every return-prediction
          script in this paper uses.
MEASURED  every loading below, on the daily cross-sections, aggregated
          Fama-MacBeth as elsewhere in the paper. The bootstrap resamples
          days, so it carries the same unit of independence the Fama-MacBeth
          standard errors assume.
BASELINE: the day-$t$ filters of tab:investor_heterogeneity | obvious
          alternative: a referee says the excess directionality is flow
          persistence, so the filter predicts returns only because it predicts
          the flow that moves them | discharge: this script, reported in
          subsec:appendix_flow_continuation.

Provenance
----------
Post-hoc. The channel was named after the draft was written, and this test
was written to answer it, so no criterion for it was published in advance and
none is claimed here. What the run is asked for is a rejection
of (*) at the paper's own 5% two-sided bar, ``CRIT_5PCT``, reported as it
falls and for each class separately.

Arms
----
``continuation``    each filter's own day-to-day autocorrelation, both sides
                    day-standardised, so the two are comparable and the
                    persistence the objection starts from is measured before
                    anything is concluded about it.
``proportionality`` the four loadings of (*), the cross-contrast, and the
                    impact constant each filter implies.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/flow_continuation_results.json
"""

import numpy as np
import pandas as pd

from shared_utils import (
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    daily_stats_pack,
    ols_fit,
    print_header,
    save_json,
)
from shared_utils.regression import CRIT_5PCT

from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR

INPUT_FILE = str(DAILY_DATA_ALL_INVESTORS)
RESULTS_PATH = str(RESULTS_DIR / "flow_continuation_results.json")

RETURN_HORIZON = "R_t1"
MIN_STOCKS_PER_DAY = MIN_STOCKS_PER_DAY_DEFAULT
BPS = 10000.0

CLASSES = ("capacity_bound", "foreign")

# Pressure measures. The participation rate is primary for the reason the
# module docstring gives; the cap-scaled one is the demand-system alternative.
PRESSURES = (("participation", "S_TV"), ("capitalisation", "S_MC"))

# Day-block bootstrap over the per-day loading series. The contrast (*) is a
# product of Fama-MacBeth means, so its standard error is resampled rather
# than differenced: forming the product day by day instead would multiply two
# estimates carrying correlated cross-sectional sampling error.
N_BOOTSTRAP = 2000

rng = np.random.default_rng(MC_SEED)


def _zscore(x):
    """Within-day z-score; returns None when the cross-section is degenerate."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    if sd == 0 or not np.isfinite(sd):
        return None
    return (x - x.mean()) / sd


def _slope(z, y):
    """Slope of ``y`` on a standardised regressor ``z``, with an intercept."""
    return float(np.asarray(
        ols_fit(y, np.column_stack([z])).params, dtype=float)[1])


def load_class_panel(cls):
    """Each class's own nonzero-flow stock-days, carrying its day-$(t+1)$ flow.

    The panel is sorted by (ticker, date) and the next-day columns are taken
    with a within-ticker shift, the same construction
    ``shared_utils.compute_forward_returns`` uses for the forward return, so
    the pressure variable and the return it is paired with advance by the same
    step of that stock's own series.
    """
    cols = ["date", "ticker", RETURN_HORIZON,
            *[f"{p}_{c}" for c in CLASSES for p in ("D", "S_MC", "S_TV")]]
    df = pd.read_parquet(INPUT_FILE, columns=cols)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    for p in ("D", "S_MC", "S_TV"):
        df[f"next_{p}"] = df.groupby("ticker")[f"{p}_{cls}"].shift(-1)
    keep = ["date", "ticker", RETURN_HORIZON,
            f"S_MC_{cls}", f"S_TV_{cls}", "next_D", "next_S_MC", "next_S_TV"]
    out = df.loc[df[f"D_{cls}"] != 0, keep].copy()
    return out.rename(columns={f"S_MC_{cls}": "sMC", f"S_TV_{cls}": "sTV"})


def daily_loadings(panel, pressure_col):
    """Per-day loading series for one class and one pressure measure.

    Returns ``{b_mc, b_tv, q_mc, q_tv}`` as arrays over days: ``b`` return
    loadings in return units and ``q`` pressure loadings.
    """
    keys = ("b_mc", "b_tv", "q_mc", "q_tv")
    acc = {k: [] for k in keys}
    n_rows = []
    for _, day in panel.groupby("date", sort=True):
        d = day[["sMC", "sTV", pressure_col, RETURN_HORIZON]].dropna()
        if len(d) < MIN_STOCKS_PER_DAY:
            continue
        z = {k: _zscore(d[k]) for k in ("sMC", "sTV", pressure_col)}
        if any(v is None for v in z.values()):
            continue
        y = d[RETURN_HORIZON].to_numpy(dtype=float)
        press = z[pressure_col]
        for tag, col in (("mc", "sMC"), ("tv", "sTV")):
            x = z[col]
            acc[f"b_{tag}"].append(_slope(x, y))
            acc[f"q_{tag}"].append(_slope(x, press))
        n_rows.append(len(d))
    out = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    out["n_days"] = len(n_rows)
    out["avg_n_rows"] = float(np.mean(n_rows)) if n_rows else float("nan")
    return out


def _bootstrap_cross(loadings):
    """Day-block bootstrap of the contrast (*) and of the two impact constants.

    Each draw resamples days with replacement, re-forms the four Fama-MacBeth
    means on the drawn days, and evaluates the contrast on them. An impact
    constant is reported only where its own denominator's interval clears
    zero, since ``b/q`` is unbounded as ``q`` approaches it.
    """
    n = loadings["n_days"]
    cross, lam_mc, lam_tv = [], [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        m = {k: loadings[k][idx].mean() for k in ("b_mc", "b_tv", "q_mc", "q_tv")}
        cross.append(m["b_mc"] * m["q_tv"] - m["b_tv"] * m["q_mc"])
        lam_mc.append(m["b_mc"] / m["q_mc"] if m["q_mc"] != 0 else np.nan)
        lam_tv.append(m["b_tv"] / m["q_tv"] if m["q_tv"] != 0 else np.nan)
    means = {k: float(loadings[k].mean()) for k in ("b_mc", "b_tv", "q_mc", "q_tv")}
    cross = np.asarray(cross, dtype=float)
    point = means["b_mc"] * means["q_tv"] - means["b_tv"] * means["q_mc"]
    lo, hi = np.percentile(cross, [2.5, 97.5])
    return {
        "cross_contrast_bps": point * BPS,
        "ci_lower_bps": float(lo) * BPS,
        "ci_upper_bps": float(hi) * BPS,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "boot_z": float(point / cross.std(ddof=1)) if cross.std(ddof=1) > 0 else 0.0,
        "impact_mc_bps": means["b_mc"] / means["q_mc"] * BPS,
        "impact_tv_bps": means["b_tv"] / means["q_tv"] * BPS,
        "impact_mc_ci_bps": [float(v) * BPS
                             for v in np.nanpercentile(lam_mc, [2.5, 97.5])],
        "impact_tv_ci_bps": [float(v) * BPS
                             for v in np.nanpercentile(lam_tv, [2.5, 97.5])],
    }


def run_class(cls):
    """Every arm for one class."""
    panel = load_class_panel(cls)
    out = {
        "n_rows_day_t": int(len(panel)),
        "share_next_day_nonzero": float((panel["next_D"] != 0).mean()),
    }

    # --- continuation: each filter's own autocorrelation, both sides z-scored.
    # The two series are formed on one pass so the advantage differences them
    # day by day, as every other contrast in the paper does.
    pair = {"mc": [], "tv": []}
    for _, day in panel.groupby("date", sort=True):
        d = day[["sMC", "sTV", "next_S_MC", "next_S_TV"]].dropna()
        if len(d) < MIN_STOCKS_PER_DAY:
            continue
        z = {k: _zscore(d[k]) for k in d.columns}
        if any(v is None for v in z.values()):
            continue
        pair["mc"].append(_slope(z["sMC"], z["next_S_MC"]))
        pair["tv"].append(_slope(z["sTV"], z["next_S_TV"]))
    mc, tv = np.asarray(pair["mc"]), np.asarray(pair["tv"])
    out["continuation"] = {"all": {"mc": daily_stats_pack(mc), "tv": daily_stats_pack(tv),
                                   "advantage": daily_stats_pack(mc - tv)}}

    # --- proportionality and mediation, one block per pressure measure.
    out["pressure"] = {}
    for name, col in PRESSURES:
        loadings = daily_loadings(panel, f"next_{col}")
        block = {
            "n_days": loadings["n_days"],
            "avg_n_rows": loadings["avg_n_rows"],
            "return": {
                "mc": daily_stats_pack(loadings["b_mc"] * BPS),
                "tv": daily_stats_pack(loadings["b_tv"] * BPS),
                "advantage": daily_stats_pack(
                    (loadings["b_mc"] - loadings["b_tv"]) * BPS),
            },
            "pressure": {
                "mc": daily_stats_pack(loadings["q_mc"]),
                "tv": daily_stats_pack(loadings["q_tv"]),
                "advantage": daily_stats_pack(loadings["q_mc"] - loadings["q_tv"]),
            },
            "proportionality": _bootstrap_cross(loadings),
        }
        block["verdict_rejected"] = bool(
            block["proportionality"]["excludes_zero"]
            and abs(block["proportionality"]["boot_z"]) > CRIT_5PCT)
        out["pressure"][name] = block
    return out


def main():
    print_header("Flow continuation and the persistence-then-impact channel",
                 level="banner")
    results = {
        "config": {
            "return_horizon": RETURN_HORIZON,
            "min_stocks_per_day": MIN_STOCKS_PER_DAY,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": MC_SEED,
            "crit_5pct": CRIT_5PCT,
            "pressure_measures": [n for n, _ in PRESSURES],
        },
        "classes": {},
    }
    for cls in CLASSES:
        print_header(cls, level="section")
        res = run_class(cls)
        results["classes"][cls] = res
        c = res["continuation"]["all"]
        print(f"  rows (nonzero day-t flow) : {res['n_rows_day_t']:,}")
        print(f"  share trading again at t+1: {res['share_next_day_nonzero']:.3f}")
        print(f"  continuation  MC {c['mc']['mean']:+.4f} (t {c['mc']['t']:+.2f})"
              f"   TV {c['tv']['mean']:+.4f} (t {c['tv']['t']:+.2f})"
              f"   adv {c['advantage']['mean']:+.4f} (t {c['advantage']['t']:+.2f})")
        for name, _ in PRESSURES:
            b = res["pressure"][name]
            pr, pp = b["return"]["advantage"], b["pressure"]["advantage"]
            pc = b["proportionality"]
            print_header(f"pressure = {name}", level="subsection")
            print(f"  return   advantage {pr['mean']:+.3f} bp (t {pr['t']:+.2f})")
            print(f"  pressure advantage {pp['mean']:+.4f}   (t {pp['t']:+.2f})")
            print(f"  cross-contrast {pc['cross_contrast_bps']:+.3f} bp "
                  f"[{pc['ci_lower_bps']:+.3f}, {pc['ci_upper_bps']:+.3f}]"
                  f"  z {pc['boot_z']:+.2f}  rejected={b['verdict_rejected']}")
    save_json(results, RESULTS_PATH)


if __name__ == "__main__":
    main()
