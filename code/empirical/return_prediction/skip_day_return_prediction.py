#!/usr/bin/env python3
"""Return prediction on the first and the second day after the flow
(qf_paper.tex subsec:skip_day, qf_paper.tex tab:skip_day).

The headline next-day result regresses the signal formed on day t on the
close-to-close return over [t, t+1]. This script re-runs both normalisations
on the one-day return over [t+1, t+2], which measures how long each filter's
predictability lasts. In Korea neither matched filter predicts that return and
both mismatched filters do, capacity-bound S^TV negatively and foreign S^MC
positively, so market capitalisation leads in both classes on the second day.
subsec:skip_day reads that ordering as the end of the reversal, and
code/empirical/return_prediction/reversal_interaction.py estimates the contrast itself on the same return.
The return also contains no day-t price. The closing-print channel of
subsec:vwap_formation is closed by the VWAP formation-price test
(code/empirical/return_prediction/vwap_formation_robustness.py) rather than here.
We run it on the two assigned classes in each market, namely Capacity-Bound
Institutional flow under S^MC and foreign flow under S^TV,
against the next-day baseline. Both normalisations are stored at both horizons
in every cell, so both reach the table and the assertions.

R_skip is the per-stock daily return realised over [t+1, t+2], i.e. the
close-to-close return ``close.pct_change()`` shifted up by two trading days so
each row t carries the return that starts one full day after the signal day's
close. Every other element of the specification (day-standardisation, the
nonzero-flow filter, the >= MIN_STOCKS_PER_DAY daily-cross-section gate) is
identical to the baseline return regression.

Degrees of freedom
------------------
SKIP_DAYS       CHOSEN — 2, the smallest shift that removes close_t from the
                return window. A shift of one reproduces the baseline, whose
                [t, t+1] return carries close_t as its base price; a larger
                shift would test a different horizon rather than the same one
                displaced.
BASELINE_COL    CHOSEN — R_t1, so the two arms differ in the return window and
                in nothing else. Every remaining element of the specification
                is inherited from the baseline regression rather than set here.
MIN_STOCKS_PER_DAY  inherited from shared_utils.config, the floor every
                Fama-MacBeth specification in the paper uses.

BASELINE: the published next-day regression as the comparison arm | obvious
alternative: the cumulative return over [t, t+h], which contains the first day
and so cannot date where the ordering ends | discharge: argument.
subsec:skip_day reads the second day alone, which the cumulative return
cannot isolate.

Output: ../results/skip_day_return_prediction_results.json
"""

import numpy as np
import pandas as pd

from shared_utils import (
    MIN_STOCKS_PER_DAY_DEFAULT,
    fm_investor_signal_triple,
    save_json,
)
from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR

OUTPUT_PATH = RESULTS_DIR / "skip_day_return_prediction_results.json"

# Trading days skipped before the return window opens. SKIP_DAYS = 2 maps the
# one-day return over [t+1, t+2] onto the signal row t, removing close_t from
# the return entirely (a one-day gap after the signal day's close).
SKIP_DAYS = 2
BASELINE_COL = "R_t1"
SKIP_COL = "R_skip"


def add_skip_return(df, entity_col):
    """Add R_skip, the one-day return over [t+1, t+2], aligned to row t.

    ``close.pct_change()`` is the daily return over [t-1, t]; shifting it up by
    SKIP_DAYS within each stock pulls the [t+1, t+2] return onto row t. Assumes
    no prior sort; sorts by [entity, date] so the shift respects stock
    boundaries and calendar order.
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values([entity_col, "date"]).reset_index(drop=True)
    daily_ret = df.groupby(entity_col)["close"].pct_change()
    df[SKIP_COL] = daily_ret.groupby(df[entity_col]).shift(-SKIP_DAYS)
    return df


def _cell(df, investor_type, entity_col, label, matched_arm):
    """Baseline (t+1) vs skip-a-day matched-filter t-statistics for one cell.

    ``matched_arm`` is 'MC' or 'TV'. Returns the full FM triple for both
    horizons plus the extracted matched-filter univariate and horse-race
    t-statistics that the table and prose quote.
    """
    base = fm_investor_signal_triple(
        df, investor_type, return_col=BASELINE_COL,
        min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT, entity_col=entity_col)
    skip = fm_investor_signal_triple(
        df, investor_type, return_col=SKIP_COL,
        min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT, entity_col=entity_col)

    uni_key = f"univariate_{matched_arm}"
    hr_key = f"horse_race_{matched_arm}"
    out = {
        "label": label,
        "matched_filter": f"S^{matched_arm}",
        "matched_arm": matched_arm,
        "baseline": base,
        "skip": skip,
        "uni_t_baseline": base[uni_key]["t_stat"],
        "uni_t_skip": skip[uni_key]["t_stat"],
        "hr_t_baseline": base[hr_key]["t_stat"],
        "hr_t_skip": skip[hr_key]["t_stat"],
        "n_days_baseline": base["n_days"],
        "n_days_skip": skip["n_days"],
    }
    print(f"  {label:<32s} matched=S^{matched_arm}  "
          f"uni t: {out['uni_t_baseline']:+6.2f} (t+1) -> {out['uni_t_skip']:+6.2f} (skip)  "
          f"HR t: {out['hr_t_baseline']:+6.2f} -> {out['hr_t_skip']:+6.2f}")
    return out


def main():
    print("Korea (main window):")
    kr = pd.read_parquet(
        DAILY_DATA_ALL_INVESTORS,
        columns=["date", "ticker", "close", BASELINE_COL,
                 "D_capacity_bound", "S_MC_capacity_bound", "S_TV_capacity_bound",
                 "D_foreign", "S_MC_foreign", "S_TV_foreign"],
    )
    kr = add_skip_return(kr, "ticker")
    korea = {
        "capacity_bound": _cell(kr, "capacity_bound", "ticker",
                                "Korea Capacity-Bound Inst.", "MC"),
        "foreign": _cell(kr, "foreign", "ticker", "Korea Foreign", "TV"),
    }

    save_json(
        {"config": {"skip_days": SKIP_DAYS, "baseline_col": BASELINE_COL,
                    "min_stocks_per_day": MIN_STOCKS_PER_DAY_DEFAULT},
         "korea": korea},
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
