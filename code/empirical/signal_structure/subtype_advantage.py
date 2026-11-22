#!/usr/bin/env python3
"""Matched-filter advantage, class by class, across the disclosed categories.

subsec:reversal_test identifies the reversal at one boundary,
capacity-bound institutions against foreign flow. KRX discloses ten investor
categories daily, so the advantage can be read for every category the panel
carries, not only for the two the reversal compares.

For each class we estimate the day-standardised return regressions of
subsec:return_prediction on that class's own nonzero-flow panel and take the
matched-filter advantage to be the Fama-MacBeth average of the per-day
difference b_MC - b_TV. That difference is exactly the class-level quantity
eq:reversal_contrast subtracts across two classes, and its standard error
comes from the time-series dispersion of the per-day difference rather than
from differencing two separately aggregated standard errors.

The advantage is measured under the primary specification of
subsec:reversal_test, which regresses on one signal at a time,

    R_{i,t+1} = a_t + b z(S) + e,

because prop:matched_filter ranks each filter's correlation with returns
rather than its slope conditional on the other filter.

Degrees of freedom
------------------
CHOSEN    the inclusion rule. A class enters if its return regressions are
          estimable on at least ``MIN_DAY_SHARE`` of the trading days, so a category
          that is disclosed but rarely traded does not enter as a point whose
          position is noise. Excluded classes are named in the output.
CHOSEN    the ten disclosed categories themselves, which are what KRX reports
          and not a selection made here.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/subtype_advantage_results.json

BASELINE: the per-class matched-filter advantage $b_{MC} - b_{TV}$ |
          obvious alternative: a referee says the capacity-bound advantage is
          special to the pension and insurance pair rather than shared by the
          institutional aggregate's other sub-types | discharge:
          subsec:subtype_dose_response reports the advantage for every
          estimable category.

"""

import numpy as np
import pandas as pd

from shared_utils import (
    MIN_STOCKS_PER_DAY_DEFAULT,
    daily_stats_pack,
    ols_fit,
    print_header,
    save_json,
)

from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR

INPUT_FILE = str(DAILY_DATA_ALL_INVESTORS)
RESULTS_PATH = str(RESULTS_DIR / "subtype_advantage_results.json")

RETURN_HORIZON = "R_t1"
MIN_STOCKS_PER_DAY = MIN_STOCKS_PER_DAY_DEFAULT

# A class must be estimable on at least this share of the panel's trading days.
MIN_DAY_SHARE = 0.5

# The ten categories KRX discloses daily, ordered as in
# subsec:subtype_dose_response. Labels carry the Korean disclosure name.
CLASSES = [
    ("pension", "Pension Funds (연기금등)"),
    ("insurance", "Insurance (보험)"),
    ("investment_trust", "Investment Trust (투신)"),
    ("pe_funds", "Private Equity Funds (사모펀드)"),
    ("financial_inv", "Proprietary (금융투자)"),
    ("other_financial", "Other Financial (기타금융)"),
    ("bank", "Banks (은행)"),
    ("government", "Government (국가)"),
    ("foreign", "Foreign"),
    ("individual", "Individual"),
]

def class_advantage(df, key):
    """Fama-MacBeth return regressions on one class's nonzero-flow panel.

    Returns ``None`` when the class clears fewer than ``MIN_DAY_SHARE`` of
    the panel's trading days. Signals are standardised within each day,
    matching every other Fama-MacBeth specification in the paper, and a day is
    dropped when either signal has zero or non-finite cross-sectional
    dispersion.
    """
    sub = df[df[f"D_{key}"] != 0]
    sub = sub.dropna(subset=[f"S_MC_{key}", f"S_TV_{key}", RETURN_HORIZON])

    daily = {sig: [] for sig in ("S_MC", "S_TV")}
    n_stocks = []
    for _, day in sub.groupby("date", sort=True):
        if len(day) < MIN_STOCKS_PER_DAY:
            continue
        z = {}
        for col in ("S_MC", "S_TV"):
            v = day[f"{col}_{key}"].to_numpy(dtype=float)
            sd = v.std(ddof=1)
            if sd == 0 or not np.isfinite(sd):
                break
            z[col] = (v - v.mean()) / sd
        if len(z) < 2:
            continue
        y = day[RETURN_HORIZON].to_numpy(dtype=float)
        for sig in ("S_MC", "S_TV"):
            params = np.asarray(ols_fit(y, z[sig].reshape(-1, 1)).params, dtype=float)
            daily[sig].append(params[1])
        n_stocks.append(len(day))

    n_days_panel = df["date"].nunique()
    if len(n_stocks) < MIN_DAY_SHARE * n_days_panel:
        return None
    out = {
        "n_days": len(n_stocks),
        "n_days_share": len(n_stocks) / n_days_panel,
        "avg_n_stocks": float(np.mean(n_stocks)),
        "panel_size": int(len(sub)),
    }
    mc, tv = np.array(daily["S_MC"]), np.array(daily["S_TV"])
    out["univariate"] = {
        "b_MC": daily_stats_pack(mc),
        "b_TV": daily_stats_pack(tv),
        "advantage": daily_stats_pack(mc - tv),
    }
    return out


def main():
    """Entry point: measure every disclosed class."""
    print_header("Matched-filter advantage across the disclosed classes", level="banner")
    cols = ["date", "ticker", RETURN_HORIZON]
    for key, _ in CLASSES:
        cols += [f"D_{key}", f"S_MC_{key}", f"S_TV_{key}"]
    df = pd.read_parquet(INPUT_FILE, columns=cols)

    classes, excluded = {}, {}
    for key, label in CLASSES:
        fm = class_advantage(df, key)
        if fm is None:
            excluded[key] = {
                "label": label,
                "reason": "no estimable return regression",
                "panel_size": int((df[f"D_{key}"] != 0).sum()),
            }
            print(f"  {label:<34} excluded ({excluded[key]['reason']})")
            continue
        classes[key] = {"label": label, **fm}
        adv = fm["univariate"]["advantage"]
        print(f"  {label:<34} adv = {adv['mean'] * 1e4:+7.2f} bps (t = {adv['t']:+6.2f})")

    results = {
        "config": {
            "return_horizon": RETURN_HORIZON,
            "min_stocks_per_day": MIN_STOCKS_PER_DAY,
            "min_day_share": MIN_DAY_SHARE,
            "n_days_panel": int(df["date"].nunique()),
            "primary_specification": "univariate",
        },
        "classes": classes,
        "excluded": excluded,
    }
    save_json(results, RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()
