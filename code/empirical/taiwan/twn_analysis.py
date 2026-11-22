#!/usr/bin/env python3
"""
Taiwan re-estimation on the TWSE/TPEx panel (sec:taiwan).

One load of the Taiwan panel, for the two classes the assignment reaches,
SITC (Investment Trust, the capacity-bound mandate) and foreign flow, gives
the panel's rows, stocks and window, which sec:taiwan prints, and the return
prediction (sec:taiwan, the subsec:return_prediction analogue): per class,
cross-sectional Fama-MacBeth regressions of next-day returns on the
day-standardised matched and mismatched signals (univariate S^MC, univariate
S^TV, horse race), and the same horse races re-run without the stock-days
whose forward window crosses a disclosed corporate action. The assignment is
the a-priori one in ``A_PRIORI_ASSIGNMENT``.
Output: ../results/twn_return_prediction_results.json

BASELINE: the disclosed Foreign_Investor line taken as one class and assigned
          to the volume regime by the Korean participation-rate reading |
          obvious alternative: a referee says the label mixes
          capacity-constrained position accumulation with participation-rate
          execution, so both filters load in the horse race because both
          components are present, with no appeal to the signal-to-noise ratio
          needed | discharge: sec:taiwan, which states that the design
          separates neither account.

Degrees of freedom
------------------
CHOSEN    the a-priori assignment: SITC to the capacity regime and foreign
          flow to the volume regime, both transported from Korea's mandate
          and execution readings.
CHOSEN    ``INVESTOR_LABELS``, the two classes the assignment reaches. TWSE
          and TPEx disclose four; the dealer line and the residual individual
          line share neither a mandate nor an execution convention, so the
          assignment makes no prediction for them and they are not estimated.
CHOSEN    ``RETURN_COL`` = R_t1 and ``MIN_STOCKS_PER_DAY``, the same horizon
          and cross-section floor the Korean tests use, so the two markets
          differ in data and not in specification.
MEASURED  every coefficient and drift below, on each class's own
          nonzero-flow stock-days.
FITTED    nothing.

"""

import pandas as pd

from shared_utils import (
    MIN_STOCKS_PER_DAY_DEFAULT,
    fama_macbeth_horse_race as _fm_horse_race,
    fama_macbeth_univariate as _fm_univariate,
    save_json,
)
from paths import RESULTS_DIR, DAILY_DATA_TAIWAN

INPUT_PATH = DAILY_DATA_TAIWAN
RETURN_RESULTS_PATH = RESULTS_DIR / "twn_return_prediction_results.json"

MIN_STOCKS_PER_DAY = MIN_STOCKS_PER_DAY_DEFAULT
RETURN_COL = "R_t1"

# The two Taiwan classes the assignment reaches, in reporting order. SITC
# (Investment Trust) is the structural capacity-bound mandate, analogous to
# Korea's pension + insurance.
INVESTOR_LABELS = {
    "institutional": "Investment Trust (SITC)",
    "foreign": "Foreign Investor",
}

# The a-priori regime assignment, carried over from Korea's mandate and
# execution readings (subsec:regime_assignment, sec:taiwan).
A_PRIORI_ASSIGNMENT = {
    "institutional": {"regime": "capacity", "matched_filter": "S^MC"},
    "foreign": {"regime": "volume", "matched_filter": "S^TV"},
}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def describe_panel(df):
    """The panel's rows, stocks and window, which sec:taiwan prints."""
    return {
        "n_rows": int(len(df)),
        "n_stocks": int(df["stock_id"].nunique()),
        "n_dates": int(df["date"].nunique()),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
    }


# ---------------------------------------------------------------------------
# Return prediction (subsec:return_prediction analogue)
# ---------------------------------------------------------------------------

def fm_univariate(df, signal_col, return_col, flow_col):
    return _fm_univariate(
        df, signal_col, return_col,
        min_stocks_per_day=MIN_STOCKS_PER_DAY,
        row_filter=lambda sub: sub[sub[flow_col] != 0],
        entity_col="stock_id",
    )


def fm_horse_race(df, mc_col, tv_col, return_col, flow_col):
    return _fm_horse_race(
        df,
        [(mc_col, "signal_MC"), (tv_col, "signal_TV")],
        return_col,
        min_stocks_per_day=MIN_STOCKS_PER_DAY,
        row_filter=lambda sub: sub[sub[flow_col] != 0],
        entity_col="stock_id",
    )


def run_for_investor(df, inv_key, label):
    flow_col = f"D_{inv_key}"
    mc_col = f"S_MC_{inv_key}"
    tv_col = f"S_TV_{inv_key}"

    sub = df[df[flow_col] != 0]
    print(f"\n[{label}]  panel = {len(sub):,} stock-days, "
          f"{sub['stock_id'].nunique():,} stocks, {sub['date'].nunique():,} dates")

    out = {
        "label": label,
        "panel_size": int(len(sub)),
        "n_unique_stocks": int(sub["stock_id"].nunique()),
        "univariate_MC": fm_univariate(sub, mc_col, RETURN_COL, flow_col),
        "univariate_TV": fm_univariate(sub, tv_col, RETURN_COL, flow_col),
        "horse_race": fm_horse_race(sub, mc_col, tv_col, RETURN_COL, flow_col),
    }
    print(f"  univariate S_MC : t = {out['univariate_MC']['t_stat']:+6.2f}")
    print(f"  univariate S_TV : t = {out['univariate_TV']['t_stat']:+6.2f}")
    print(f"  horse-race S_MC : t = {out['horse_race']['signal_MC']['t_stat']:+6.2f}")
    print(f"  horse-race S_TV : t = {out['horse_race']['signal_TV']['t_stat']:+6.2f}")
    return out


def run_return_prediction(df):
    """Per-class FM return prediction under the a-priori assignment."""
    results = {"by_investor": {inv_key: run_for_investor(df, inv_key, label)
                               for inv_key, label in INVESTOR_LABELS.items()}}
    results["a_priori_assignment"] = A_PRIORI_ASSIGNMENT
    return results


def corporate_action_sensitivity(df, results):
    """Re-run each published horse race without the stock-days whose forward
    return window crosses a disclosed corporate action.

    Taiwan returns are total-return adjusted (sec:taiwan), so both the
    structural action and the cash distribution are already removed and nothing
    survives by construction for this check to bound. What it does instead is
    test the adjustment: if the event sessions still carried a step the returns
    had not absorbed, dropping the windows that cross one would move the
    published statistics.

    The instrument is the exchange's own 除權息 / 減資 disclosure, carried by the
    panel as ``is_ex_date`` / ``is_cap_red``, rather than the size of the return.
    Return magnitude is not usable here: TWSE and TPEx cap daily moves at
    +/-10%, so a filter on |R| above that threshold drops mostly limit-hit
    sessions -- the most informative days in the sample -- while missing the
    overwhelming majority of real events, whose price step is far smaller than
    the cap. ``is_ex_date`` marks the ex-date session itself, and ``R_t1`` is a
    forward return, so the contaminated row is the one whose *next* session
    carries the step.
    """
    ordered = df.sort_values(["stock_id", "date"])
    by_stock = ordered.groupby("stock_id")
    crosses = pd.Series(False, index=df.index)
    for flag in ("is_ex_date", "is_cap_red"):
        crosses |= (by_stock[flag].shift(-1) == True).reindex(df.index)  # noqa: E712
    clean = df[~crosses]
    out = {"n_event_rows": int(crosses.sum()),
           "share_event_rows": float(crosses.mean()),
           "by_investor": {}}
    print(f"\nCorporate-action sensitivity: {int(crosses.sum()):,} stock-days "
          f"({100 * crosses.mean():.3f}%) have a disclosed 除權息 or 減資 in "
          f"their forward return window")
    drifts = {}
    for inv_key, label in INVESTOR_LABELS.items():
        flow_col = f"D_{inv_key}"
        sub = clean[clean[flow_col] != 0]
        hr = fm_horse_race(sub, f"S_MC_{inv_key}", f"S_TV_{inv_key}",
                           RETURN_COL, flow_col)
        pub = results["by_investor"][inv_key]["horse_race"]
        row = {}
        for arm in ("signal_MC", "signal_TV"):
            drift = abs(hr[arm]["t_stat"] - pub[arm]["t_stat"])
            drifts[(inv_key, arm)] = drift
            row[arm] = {"t_stat": hr[arm]["t_stat"],
                        "published_t_stat": pub[arm]["t_stat"],
                        "abs_drift": drift}
        out["by_investor"][inv_key] = row
        print(f"  {label:<24} hrMC {pub['signal_MC']['t_stat']:+6.2f} -> "
              f"{hr['signal_MC']['t_stat']:+6.2f}   hrTV "
              f"{pub['signal_TV']['t_stat']:+6.2f} -> "
              f"{hr['signal_TV']['t_stat']:+6.2f}")
    out["max_abs_drift"] = max(drifts.values())
    print(f"  max |drift| over all cells = {out['max_abs_drift']:.2f}")
    return out


def main():
    df = pd.read_parquet(INPUT_PATH)
    print(f"Panel: {len(df):,} rows, {df['stock_id'].nunique():,} stocks, "
          f"{df['date'].nunique():,} dates ({df['date'].min().date()} -> "
          f"{df['date'].max().date()})")

    results = {"panel": describe_panel(df), **run_return_prediction(df)}
    results["corporate_action_sensitivity"] = corporate_action_sensitivity(df, results)
    save_json(results, RETURN_RESULTS_PATH)


if __name__ == "__main__":
    main()
