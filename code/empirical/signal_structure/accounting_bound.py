#!/usr/bin/env python3
"""
The accounting identity |D| <= V behind the bound |S^TV| <= 1 of
subsec:threshold_estimate, checked on the panel.

A class's net buying of a stock on a day is at most its own purchases plus
sales, which are part of the day's traded value, so |S^TV| = |D|/V <= 1 holds
for every investor class on every stock-day. subsec:threshold_estimate derives
from that bound, through S^MC = tau S^TV, that the component of S^MC which the
independence premise makes independent of turnover must shrink as turnover
falls. This script records the bound as it appears in the data, so a panel
that violates it fails the assertion suite instead of the argument.

Degrees of freedom
------------------
REGIMES         CHOSEN — each class on its own nonzero-flow stock-days, the
                panel restriction of tab:investor_heterogeneity.

BASELINE: each class's own nonzero-flow stock-days | obvious alternative: the
full panel, zero-flow stock-days included | discharge: a zero-flow stock-day has
|S^TV| = 0, so the nonzero-flow rows are the only rows on which the bound can
fail, and checking them checks the full panel.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/accounting_bound.json
"""

import numpy as np
import pandas as pd
from shared_utils import print_header, save_json

from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR

INPUT_FILE = str(DAILY_DATA_ALL_INVESTORS)
RESULTS_PATH = str(RESULTS_DIR / "accounting_bound.json")

# Each class is checked on its own nonzero-flow stock-days, matching the
# panel restriction of tab:investor_heterogeneity.
REGIMES = {
    "capacity": {"flow": "D_capacity_bound", "signal_tv": "S_TV_capacity_bound"},
    "volume": {"flow": "D_foreign", "signal_tv": "S_TV_foreign"},
}


def accounting_bound(panel, spec):
    """Realised participation rates against the |D| <= V accounting identity."""
    stv = np.abs(panel[spec["signal_tv"]].to_numpy(dtype=float))
    stv = stv[np.isfinite(stv)]
    return {
        "n_obs": int(len(stv)),
        "max_abs_stv": float(stv.max()),
        "share_abs_stv_above_one": float(np.mean(stv > 1.0)),
    }


def main():
    """Entry point: the bound on each class's own panel."""
    print_header("Accounting bound |S^TV| <= 1", level="banner")
    cols = ["date"] + [v for spec in REGIMES.values() for v in spec.values()]
    df = pd.read_parquet(INPUT_FILE, columns=cols)

    results = {}
    for regime, spec in REGIMES.items():
        panel = df[df[spec["flow"]] != 0]
        results[regime] = accounting_bound(panel, spec)
        print(f"  {regime}: n = {results[regime]['n_obs']:,}, "
              f"max |S^TV| = {results[regime]['max_abs_stv']:.4f}")

    save_json(results, RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()
