#!/usr/bin/env python3
"""Rebuild $S^{TV}$ on a lagged ADV denominator and re-estimate the reversal.

The volume normalisation of subsec:reversal_test divides day-$t$ net flow by
day-$t$ traded value, $S^{TV}_{i,t} = D_{i,t}/V_{i,t}$. The numerator is a
component of the denominator, because $D$ is the net of buys and sells that
$V$ totals, so the two move together for reasons that have nothing to do with
the matched filter. That channel is not uniform across the two normalisations:
$D$ enters $V$ and does not enter $M$, so it loads $S^{TV}$ alone and the
within-panel argument of subsec:reversal_test does not dispose of it.

This script replaces the denominator with the stock's average daily traded
value over the ``window`` panel days ending at $t-1$,

    S^{TV,ADV}_{i,t} = D_{i,t} / ADV_{i,t-1}(window),

which contains no day-$t$ trading at all. The theoretical structure survives
the substitution, since

    S^{TV,ADV}_{i,t} = S^{MC}_{i,t} / \\bar\\tau_{i,t},
    \\bar\\tau_{i,t} = ADV_{i,t-1}(window) / M_{i,t},

so the volume normalisation is still the market-cap normalisation rotated by a
turnover, now a turnover whose traded value is predetermined at $t$.

Degrees of freedom
------------------
CHOSEN    the ADV window. The headline is \\advWindow{} panel days, the
          convention in the execution-cost literature; the same estimate is
          reported at 5 and 60 days and the sweep is the discharge.
CHOSEN    a full window is required, so the first ``max(ADV_WINDOWS)``
          observations of each stock are dropped. Every arm on the restricted
          sample is estimated on that one sample, so the sweep isolates the
          window rather than the sample.
BASELINE: same-day $V$ | obvious alternative: a referee asks what happens when
          $D$ leaves the denominator | discharge: this script, reported in
          sec:appendix_nonuniform_channels.

Arms
----
``same_day``      the subsec:reversal_test specification, re-estimated on the
                  restricted sample so the ADV arms are compared like for like.
``adv_{w}``       the lagged-ADV denominator at window ``w``.
``same_day_full`` the same specification on the unrestricted sample, which
                  must reproduce ``results/reversal_interaction_results.json`` exactly.
                  A mismatch means the stacking or standardisation drifted
                  between the two scripts.

Every arm is estimated under the univariate specification of
subsec:reversal_test, the primary estimand, since prop:matched_filter ranks
each filter's own correlation with returns. The confound this script tests is
a property of $S^{TV}$ alone, so it has to be tested on the specification the
headline reports. The joint specification and the cluster-by-date fit are
estimated where sec:appendix_nonuniform_channels prints them, at the headline
window on common support, and on ``same_day_full``, which reproduces
code/empirical/return_prediction/reversal_interaction.py under all three fits. The window sweep runs on common
support, and the union carries the headline window alone.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/lagged_adv_normalisation_results.json
"""

import numpy as np
import pandas as pd
from shared_utils import MC_SEED, print_header, save_json

from empirical.return_prediction.reversal_interaction import (
    BASE_CLASS,
    INTERACTED_CLASS,
    INPUT_FILE,
    RETURN_HORIZON,
    fama_macbeth_reversal,
    fama_macbeth_univariate,
    pooled_reversal,
    standardise_within_day_class,
)
from paths import RESULTS_DIR

RESULTS_PATH = str(RESULTS_DIR / "lagged_adv_normalisation_results.json")

# Windows in panel days. HEADLINE_WINDOW is what sec:appendix_nonuniform_channels
# reports; the other two bracket it so the choice carries a sensitivity check.
ADV_WINDOWS = (5, 20, 60)
HEADLINE_WINDOW = 20

np.random.seed(MC_SEED)


def load_panel():
    """Load the panel and attach one lagged-ADV column per window.

    The rolling mean runs over each stock's own panel rows, so a window of
    ``w`` is the last ``w`` days on which the stock survives the pipeline
    filters rather than the last ``w`` calendar trading days. Shifting by one
    row before the roll is what keeps day $t$ out of its own denominator.
    """
    cols = ["date", "ticker", RETURN_HORIZON, "V", "M"]
    for cls in (BASE_CLASS, INTERACTED_CLASS):
        cols += [f"D_{cls}", f"S_MC_{cls}"]
    df = pd.read_parquet(INPUT_FILE, columns=cols)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    lagged = df.groupby("ticker")["V"].shift(1)
    for w in ADV_WINDOWS:
        df[f"ADV_{w}"] = (
            lagged.groupby(df["ticker"])
            .rolling(w, min_periods=w)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return df


def stack(df, sample, denominator):
    """Stack the two class panels, dividing flow by ``denominator``.

    Mirrors ``reversal_interaction.load_stacked`` except that the volume
    signal is rebuilt here from ``D`` and the requested denominator column
    instead of being read off the panel's stored ``S_TV`` columns. With
    ``denominator='V'`` the two agree row for row, since ``V`` is strictly
    positive throughout the panel and the stored column's zero-volume mask
    never fires.
    """
    if sample == "common":
        keep = (df[f"D_{BASE_CLASS}"] != 0) & (df[f"D_{INTERACTED_CLASS}"] != 0)
        df = df[keep]

    parts = []
    for cls in (BASE_CLASS, INTERACTED_CLASS):
        sub = df[df[f"D_{cls}"] != 0]
        part = sub[["date", "ticker", RETURN_HORIZON]].copy()
        part["zMC"] = sub[f"S_MC_{cls}"].values
        part["zTV"] = (sub[f"D_{cls}"] / sub[denominator]).values
        part["Foreign"] = float(cls == INTERACTED_CLASS)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def run_arm(df, sample, denominator, joint):
    """Estimate equation eq:reversal_interaction for one sample and denominator.

    Returns the univariate Fama-MacBeth fit and, when ``joint`` is set, the
    joint Fama-MacBeth fit and the cluster-by-date fit. Each Fama-MacBeth arm
    carries the two class-level matched-filter advantages, $b_1 - b_2$ for the
    capacity-bound class and $(b_1 + b_3) - (b_2 + b_4)$ for the foreign class,
    whose difference is the reversal.
    """
    panel = standardise_within_day_class(stack(df, sample, denominator))
    arm = {
        "n_rows": int(len(panel)),
        "n_dates": int(panel["date"].nunique()),
        "fama_macbeth_univariate": fama_macbeth_univariate(panel),
    }
    if joint:
        arm["fama_macbeth"] = fama_macbeth_reversal(panel)
        arm["pooled_cluster_date"] = pooled_reversal(panel)
    return arm


def run_sample(df, restricted, sample):
    """Run the arms of one stacking that the module docstring lists for it."""
    print_header(f"Lagged-ADV normalisation — {sample} sample", width=72)
    arms = {"same_day_full": run_arm(df, sample, "V", joint=True)}
    if sample == "common":
        arms["same_day"] = run_arm(restricted, sample, "V", joint=False)
        for w in ADV_WINDOWS:
            arms[f"adv_{w}"] = run_arm(restricted, sample, f"ADV_{w}",
                                       joint=w == HEADLINE_WINDOW)
    else:
        arms[f"adv_{HEADLINE_WINDOW}"] = run_arm(
            restricted, sample, f"ADV_{HEADLINE_WINDOW}", joint=False)

    for name, arm in arms.items():
        uni = arm["fama_macbeth_univariate"]
        extra = ""
        if "fama_macbeth" in arm:
            joint = arm["fama_macbeth"]
            extra = (f", joint {joint['reversal']['mean']:+.6f} at t = {joint['reversal']['t']:+.2f},"
                     f" cluster t = {arm['pooled_cluster_date']['reversal']['t_stat']:+.2f}")
        print(f"  {name:<14} REVERSAL = {uni['reversal']['mean']:+.6f}"
              f"  (t = {uni['reversal']['t']:+.2f}{extra})"
              f"  edges {uni['advantage_capacity']['mean']:+.6f}"
              f" / {uni['advantage_foreign']['mean']:+.6f}"
              f"  [{arm['n_rows']:,} rows]")
    return arms


def main():
    """Entry point: sweep the ADV window on both stackings of the two panels."""
    print_header("Reversal under a lagged-ADV volume denominator", level="banner")
    df = load_panel()
    restricted = df.dropna(subset=[f"ADV_{max(ADV_WINDOWS)}"])
    print(f"Panel: {len(df):,} rows; {len(restricted):,} survive the "
          f"{max(ADV_WINDOWS)}-day window requirement "
          f"({100 * len(restricted) / len(df):.1f}%)")

    results = {
        "config": {
            "return_horizon": RETURN_HORIZON,
            "adv_windows": list(ADV_WINDOWS),
            "headline_window": HEADLINE_WINDOW,
            "base_class": BASE_CLASS,
            "interacted_class": INTERACTED_CLASS,
            "n_rows_panel": int(len(df)),
            "n_rows_restricted": int(len(restricted)),
        },
        "common": run_sample(df, restricted, "common"),
        "union": run_sample(df, restricted, "union"),
    }
    save_json(results, RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()
