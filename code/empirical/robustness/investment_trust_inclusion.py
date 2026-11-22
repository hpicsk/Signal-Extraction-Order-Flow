#!/usr/bin/env python3
"""The stacked reversal with the investment-trust sub-type inside the capacity-bound class.

The capacity-bound class pools pension and insurance flow (subsec:data). The
reason subsec:data gives for the class, positions sized to market
capitalization, also describes investment trusts: sec:taiwan assigns Taiwan's
investment trusts (SITC) the capacity regime on that reason. The Korean
single-issuer limit on public funds, which governs investment trusts, refers
to a stock's market-cap weight, whereas the insurers' single-issuer limit is a
share of the insurer's own assets. Leaving the Korean
investment-trust sub-type out of the class is therefore a researcher choice.
This script re-estimates the headline contrast with the sub-type added, the
sensitivity check that a chosen value owes.

Timing. These criteria were committed before the run, in a commit that carries
no results of this script. Known when they were committed: the sub-type's own
advantage of S^MC over S^TV (subsec:subtype_dose_response), and a return-free
probe of its flow, in which its gross flow scales more with traded value than
with market capitalization. Not known: any return regression in which the
sub-type's flow enters the class.

The class is rebuilt with the panel pipeline's own builder,
``set_capacity_bound_canonical``, from ``CAPACITY_BOUND_COMPONENTS`` plus the
sub-type, and the contrast is estimated by code/empirical/return_prediction/reversal_interaction.py's
``run_sample`` on the ``common`` stacking, in its univariate arm, the primary
one of subsec:reversal_test, which is the arm the criteria read and the one
subsec:appendix_aggregate_alignment reports.

T1  reversal b4 - b3 positive, with t above the 1% critical value.
T2  capacity-bound half b1 - b2 positive, with t above the 1% critical value.

T1 and T2 apply the thresholds that subsec:contrast_halves applies to every
row of tab:contrast_halves. The foreign half carries no criterion: it is
reported with the two, whatever its sign.

The class does not change whichever way the verdicts fall. If both hold,
subsec:appendix_aggregate_alignment reports the contrast and both halves with
the sub-type in the class. If either fails, it reports the same numbers, and
subsec:data states that the reversal at the 1% level depends on leaving
investment trusts out of the class.

BASELINE: the capacity-bound class of pension and insurance flow | obvious
          alternative: a referee says investment trusts size positions to
          market capitalization too and belong in the class | discharge: run.

Degrees of freedom
------------------
CHOSEN    ``ADDED_COMPONENT`` = investment_trust, the one sub-type whose
          exclusion the class's stated reason does not cover.
CHOSEN    the ``common`` stacking and its univariate arm, the identification
          sample and primary specification of subsec:reversal_test.
CHOSEN    CRIT_1PCT, the two-sided 1% normal critical value of the
          manuscript's significance statements, from shared_utils.regression.
MEASURED  every loading, half, and reversal reported below.
FITTED    nothing.

Inputs
------
    results/full_daily_data_all_investors.parquet

Output
------
    results/investment_trust_inclusion.json
"""

import os
import tempfile

import numpy as np
import pandas as pd

from empirical.return_prediction import reversal_interaction as ri
from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR
from shared_utils import print_header, save_json
from shared_utils.data import CAPACITY_BOUND_COMPONENTS, set_capacity_bound_canonical
from shared_utils.regression import CRIT_1PCT

ADDED_COMPONENT = "investment_trust"
RESULTS_PATH = str(RESULTS_DIR / "investment_trust_inclusion.json")

# The columns set_capacity_bound_canonical and run_sample read.
PANEL_COLS = ["date", "ticker", "close", ri.RETURN_HORIZON, "M", "V",
              "D_institutional", "S_MC_institutional", "S_TV_institutional",
              "D_capacity_bound", "S_MC_capacity_bound", "S_TV_capacity_bound",
              *[f"D_{c}" for c in (*CAPACITY_BOUND_COMPONENTS, ADDED_COMPONENT)],
              f"D_{ri.INTERACTED_CLASS}", f"S_MC_{ri.INTERACTED_CLASS}",
              f"S_TV_{ri.INTERACTED_CLASS}"]


def panel_with_class(components):
    """The published panel with the capacity-bound columns rebuilt from ``components``.

    The builder is first run on the published components, and the rebuilt flow
    must equal the panel's own, so the rebuild is the pipeline's construction
    and not a second one.
    """
    df = pd.read_parquet(DAILY_DATA_ALL_INVESTORS, columns=PANEL_COLS)
    published = df["D_capacity_bound"].to_numpy(copy=True)
    set_capacity_bound_canonical(df, components=CAPACITY_BOUND_COMPONENTS)
    if not np.array_equal(df["D_capacity_bound"].to_numpy(), published):
        raise SystemExit("set_capacity_bound_canonical does not rebuild the panel's "
                         "D_capacity_bound from CAPACITY_BOUND_COMPONENTS")
    set_capacity_bound_canonical(df, components=components)
    return df


def verdicts(univariate):
    """T1 and T2 of the module docstring, read off the univariate arm."""
    rev, cap = univariate["reversal"], univariate["advantage_capacity"]
    return {
        "T1_reversal": bool(rev["mean"] > 0 and rev["t"] > CRIT_1PCT),
        "T2_capacity_half": bool(cap["mean"] > 0 and cap["t"] > CRIT_1PCT),
    }


def main():
    print_header("Stacked reversal with investment trusts in the capacity-bound class",
                 level="banner")
    components = (*CAPACITY_BOUND_COMPONENTS, ADDED_COMPONENT)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "panel.parquet")
        panel_with_class(components).to_parquet(path)
        common = ri.run_sample("common", input_file=path, window="published",
                               joint=False)
    results = {
        "config": {
            "components": list(components),
            "sample": "common",
            "primary_specification": "univariate",
            "crit_1pct": CRIT_1PCT,
        },
        "common": common,
        "verdicts": verdicts(common["fama_macbeth_univariate"]),
    }
    print(f"\n  Verdicts: {results['verdicts']}")
    save_json(results, RESULTS_PATH)
    return results


if __name__ == "__main__":
    main()
