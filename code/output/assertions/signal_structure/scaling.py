"""Checks on the class estimation samples and on the collinearity of the two
filters, measured by the scripts of code/empirical/signal_structure/."""

from __future__ import annotations

import json

import pandas as pd

from paths import TABLES_DIR
from shared_utils import CRIT_1PCT

from output._formatters import BPS_PER_RETURN
from output.assertions.common import (
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def _nonzero_flow_counts() -> tuple[dict[str, int], int]:
    """Each class's nonzero-flow stock-days, the estimation samples subsec:data
    counts, and the panel total they are drawn from."""
    from paths import DAILY_DATA_ALL_INVESTORS
    panel = pd.read_parquet(DAILY_DATA_ALL_INVESTORS,
                            columns=["D_capacity_bound", "D_foreign"])
    counts = {cls: int((panel[f"D_{cls}"] != 0).sum())
              for cls in ("capacity_bound", "foreign")}
    return counts, len(panel)


def check_sample_sizes(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:data (the foreign class's stock-day count)."""
    counts, _ = _nonzero_flow_counts()
    assert_close(m["sampForeignObs"], counts["foreign"], tol=0.5, label="sampForeignObs")


def check_estimation_samples(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:appendix_panel_filters.

    The Introduction used to quote the panel total beside a result estimated
    on a subset of it, so the reader could read one number as the other. It
    now quotes the common-support count the contrast is estimated on, and
    subsec:appendix_panel_filters quotes the panel total with the two shares
    of it that separate the class estimation samples. The shares carry no
    independent measurement, so what has to hold is that they reproduce the
    same panel total from two different numerators, and that this total is
    the one \\panelTotalObsApprox publishes."""
    counts, panel_rows = _nonzero_flow_counts()
    inst_obs = counts["capacity_bound"]
    foreign_obs = counts["foreign"]

    def millions(macro: str) -> float:
        return float(m[macro].split("~")[0])

    inst_total = inst_obs / (parse_num(m["sampInstPanelPct"]) / 100.0)
    foreign_total = foreign_obs / (parse_num(m["sampForeignPanelPct"]) / 100.0)
    assert_prose_close(inst_total, foreign_total, tol=0.005 * foreign_total,
                       claim="capacity-bound institutions have nonzero net buying on "
                             f"{m['sampInstPanelPct']}% of them and foreign investors on "
                             f"{m['sampForeignPanelPct']}% (of one panel, so both shares "
                             f"must imply the same total)")
    # The macro binds to the panel itself, not to the total the two rounded
    # shares reconstruct: at two decimals that reconstruction carries the
    # percentages' own rounding, so it can only bind the total to a decimal it
    # no longer prints.
    assert_prose_close(round(panel_rows / 1e6, 2), millions("panelTotalObsApprox"),
                       tol=0.0,
                       claim=f"The panel holds "
                             f"{m['panelTotalObsApprox']} stock-days")
    assert_prose_close(foreign_total / 1e6, panel_rows / 1e6, tol=0.02,
                       claim="the two class shares reconstruct the panel the macro "
                             "publishes, to the precision the shares are printed at")


def check_prose_sample_size_lower_bound(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:return_prediction.

    FM sample size shouldn't silently shrink below the manuscript's stated total.
    Also pins the per-regression day counts: the contemporaneous panel carries
    \\sampTotalDays days, while the next-day return regression drops the final
    (no-forward-return) day and averages over \\sampTotalDaysMinusOne days.

    The contemporaneous count is read from the collinearity diagnostics, which
    are computed on every daily cross-section of that panel."""
    mc = json.loads((RESULTS / "multicollinearity_test_results.json").read_text())
    expected = parse_num(m["sampTotalDays"])
    actual = mc["vif"]["n_days"]
    assert_true(actual >= 0.9 * expected,
                claim=f"~\\sampTotalDays ({m['sampTotalDays']}) trading days "
                      f"(qf_paper.tex Methodology, FM sample)",
                detail=f"actual n_days={actual}; tolerance lower bound = {0.9 * expected:.0f}")
    # Per-regression day counts. The contemporaneous panel uses the full sample;
    # the next-day return regression drops the last day (no R_{t+1} for day T).
    rp = json.loads((RESULTS / "return_prediction_results.json").read_text())
    ret_days = rp["R_t1"]["horse_race"]["n_dates"]
    assert_true(parse_num(m["sampTotalDaysMinusOne"]) == ret_days,
                claim=f"next-day return regression averages over "
                      f"\\sampTotalDaysMinusOne ({m['sampTotalDaysMinusOne']}) "
                      f"trading days (qf_paper.tex subsec:return_prediction)",
                detail=f"return_prediction R_t1 n_dates={ret_days}")
    assert_true(ret_days == actual - 1,
                claim="next-day return regression drops exactly the final day "
                      "of the contemporaneous sample (no forward return)",
                detail=f"return n_dates={ret_days}, contemporaneous n_days={actual}")


def check_prose_multicollinearity(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:return_prediction.

    results/multicollinearity_test_results.json. The paragraph on capacity-bound
    institutions prints the mean daily correlation and the mean variance
    inflation factor of the two filters through the mcoll* macros."""
    mc = json.loads((RESULTS / "multicollinearity_test_results.json").read_text())
    assert_prose_close(mc["correlation"]["mean"], parse_num(m["mcollCorr"]), tol=5e-4,
                       claim="the two filters' 'correlation averages \\mcollCorr' "
                             "(subsec:return_prediction)")
    assert_prose_close(mc["vif"]["mean"], parse_num(m["mcollVif"]), tol=5e-4,
                       claim="'their variance inflation factor \\mcollVif' "
                             "(subsec:return_prediction)")