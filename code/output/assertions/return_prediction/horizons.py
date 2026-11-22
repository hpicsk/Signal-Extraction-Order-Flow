"""Checks on the return prediction over the second day after the flow and at
a VWAP formation price, measured by the scripts of
code/empirical/return_prediction/."""

from __future__ import annotations

import json

from output._formatters import BPS_PER_RETURN
from output.assertions.common import (
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def check_prose_skip_day(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:skip_day, the second day after the flow in
    How Far the Boundary Extends; qf_paper.tex subsec:appendix_skip_day, which
    prints the stacked second-day estimates; qf_paper.tex
    subsec:vwap_formation, the positive continuation it reads off the first.

    On the return over [t+1, t+2] the stacked reversal is absent on both
    stackings. The advantage for capacity-bound flow keeps its sign and the advantage for foreign flow turns
    positive on common support, while market capitalisation leads both classes'
    horse races in the per-class cells of tab:skip_day, which
    subsec:appendix_skip_day reports. The baseline t-stats
    must also reproduce the canonical
    return-prediction values to rule out a second, drifting computation of the
    same quantity.
    """
    sk = json.loads((RESULTS / "skip_day_return_prediction_results.json").read_text())
    cells = {
        "Korea cap": sk["korea"]["capacity_bound"],
        "Korea foreign": sk["korea"]["foreign"],
    }
    for name, c in cells.items():
        # Positive continuation at baseline (rules out bounce reversal).
        assert_true(c["uni_t_baseline"] > 2.576 and c["hr_t_baseline"] > 0,
                    claim=f"{name}: matched filter predicts positive continuation at [t,t+1]",
                    detail=f"uni_t={c['uni_t_baseline']:.2f}, hr_t={c['hr_t_baseline']:.2f}")
    # "In the per-class regressions of Table skip_day, S^MC has the larger
    # horse-race t-statistic in both classes on the second day."
    for name in ("Korea cap", "Korea foreign"):
        sk2 = cells[name]["skip"]
        assert_true(sk2["horse_race_MC"]["t_stat"] > sk2["horse_race_TV"]["t_stat"],
                    claim=f"{name}: S^MC has the larger horse-race t-statistic on the "
                          f"day-two return (subsec:appendix_skip_day)",
                    detail=f"MC={sk2['horse_race_MC']['t_stat']:.2f}, "
                           f"TV={sk2['horse_race_TV']['t_stat']:.2f}")

    # The stacked contrast on the same return, univariate as the paper's primary
    # specification. The reversal's absence is a negative claim, so it is held
    # on both stackings; the advantage for foreign flow is positive on both.
    ri = json.loads((RESULTS / "reversal_interaction_results.json").read_text())
    for stacking, key in (("SkipCommon", "common_second_day"), ("SkipUnion", "union_second_day")):
        fm = ri[key]["fama_macbeth_univariate"]
        rev, fore = fm["reversal"], fm["advantage_foreign"]
        assert_true(abs(rev["t"]) < 1.96,
                    claim="On the return over the second day after the flow the reversal "
                          "runs ... on common support and ... on the union stacking, and "
                          "the topic sentence reads that as its absence (subsec:skip_day)",
                    detail=f"{key} contrast t = {rev['t']:.2f}")
        # "The advantage for foreign flow changes sign on common support ... On the union
        # stacking, the advantage for foreign flow is positive and indistinguishable from zero."
        resolved = fore["t"] > 1.96
        assert_true(fore["mean"] > 0 and resolved == (stacking == "SkipCommon"),
                    claim="The advantage for foreign flow changes sign on common support, and on the "
                          "union stacking it is positive and cannot be separated from "
                          "zero (subsec:skip_day)",
                    detail=f"{key} advantage for foreign flow {fore['mean'] * BPS_PER_RETURN:+.2f} bp, "
                           f"t = {fore['t']:.2f}")
        assert_close(m[f"rev{stacking}Rev"], rev["mean"] * BPS_PER_RETURN,
                     tol=0.005, label=f"rev{stacking}Rev")
        assert_close(m[f"rev{stacking}T"], rev["t"], tol=0.005, label=f"rev{stacking}T")
        assert_close(m[f"rev{stacking}ForEdgeT"], fore["t"], tol=0.005,
                     label=f"rev{stacking}ForEdgeT")
    second = ri["common_second_day"]["fama_macbeth_univariate"]
    cap = second["advantage_capacity"]
    assert_true(cap["mean"] > 0 and cap["t"] > 1.96,
                claim="The advantage for capacity-bound flow keeps its sign (subsec:skip_day)",
                detail=f"{cap['mean'] * BPS_PER_RETURN:+.2f} bp, t = {cap['t']:.2f}")
    first_for = ri["common"]["fama_macbeth_univariate"]["advantage_foreign"]["mean"]
    assert_true(first_for < 0,
                claim="The advantage for foreign flow changes sign, from its first-day value",
                detail=f"first-day advantage for foreign flow {first_for * BPS_PER_RETURN:+.2f} bp")
    for macro, value in (("revSkipCommonCapEdge", cap["mean"] * BPS_PER_RETURN),
                         ("revSkipCommonCapEdgeT", cap["t"]),
                         ("revSkipCommonForEdge",
                          second["advantage_foreign"]["mean"] * BPS_PER_RETURN)):
        assert_close(m[macro], value, tol=0.005, label=macro)


def check_prose_vwap_formation(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:robustness, tab:robustness_summary; qf_paper.tex subsec:vwap_formation.

    Re-forming the market cap on VWAP instead of the closing price leaves the
    reversal unchanged in both Korean cells. The close arm must also
    reproduce the canonical horse-race macros, ruling out a second, drifting
    computation of the same quantity, and the emitted VWAP macros must equal the
    JSON so prose, macro, and data move together.
    """
    vf = json.loads((RESULTS / "vwap_formation_robustness_results.json").read_text())
    cap = vf["korea"]["capacity_bound"]
    foreign = vf["korea"]["foreign"]
    for arm in ("close", "vwap"):
        assert_true(cap[arm]["hr_matched_t"] > 2.576,
                    claim=f"Capacity-bound matched S^MC survives on {arm} formation price",
                    detail=f"hr_matched_t={cap[arm]['hr_matched_t']:.2f}")
        assert_true(cap[arm]["hr_mismatched_t"] < 0,
                    claim=f"Capacity-bound mismatched S^TV stays negative on {arm} formation price",
                    detail=f"hr_mismatched_t={cap[arm]['hr_mismatched_t']:.2f}")
        assert_true(foreign[arm]["hr_matched_t"] > 2.576,
                    claim=f"Foreign matched S^TV survives on {arm} formation price",
                    detail=f"hr_matched_t={foreign[arm]['hr_matched_t']:.2f}")

    # The VWAP-formation matched t is essentially unchanged from the close baseline.
    assert_true(abs(cap["vwap"]["hr_matched_t"] - cap["close"]["hr_matched_t"]) < 2.0,
                claim="Capacity-bound matched t is invariant to close vs VWAP formation price",
                detail=f"close={cap['close']['hr_matched_t']:.2f}, vwap={cap['vwap']['hr_matched_t']:.2f}")

    # Close arm reproduces the canonical horse-race macros (no drifting recomputation).
    assert_prose_close(cap["close"]["hr_matched_t"], parse_num(m["rpHRMCtone"]), tol=0.05,
                       claim="VWAP-test close arm reproduces capacity-bound horse-race MC t (\\rpHRMCtone)")
    assert_prose_close(cap["close"]["hr_mismatched_t"], parse_num(m["rpHRTVtone"]), tol=0.05,
                       claim="VWAP-test close arm reproduces capacity-bound horse-race TV t (\\rpHRTVtone)")
    # The foreign horse-race MC t carries no macro; the close arm is held to the
    # panel estimate tab:investor_heterogeneity prints.
    mi = json.loads((RESULTS / "multi_investor_robustness_results.json").read_text())
    assert_prose_close(foreign["close"]["hr_mismatched_t"],
                       mi["return_predictions"]["foreign"]["horse_race_MC"]["t_stat"], tol=0.05,
                       claim="VWAP-test close arm reproduces the foreign horse-race MC t "
                             "of tab:investor_heterogeneity")

    # Prose/macro/data move together: emitted VWAP macros equal the JSON values.
    assert_prose_close(cap["vwap"]["hr_matched_t"], parse_num(m["vwapCapMatchedVwap"]), tol=0.005,
                       claim="\\vwapCapMatchedVwap matches the data")
    assert_prose_close(foreign["vwap"]["hr_matched_t"], parse_num(m["vwapForMatchedVwap"]), tol=0.005,
                       claim="\\vwapForMatchedVwap matches the data")
    # subsec:vwap_formation prints the foreign mismatched pair in prose since the
    # table was folded into it, so both cells need a binding of their own.
    assert_prose_close(foreign["close"]["hr_mismatched_t"], parse_num(m["vwapForMismatchClose"]),
                       tol=0.005, claim="\\vwapForMismatchClose matches the data")
    assert_prose_close(foreign["vwap"]["hr_mismatched_t"], parse_num(m["vwapForMismatchVwap"]),
                       tol=0.005, claim="\\vwapForMismatchVwap matches the data")
