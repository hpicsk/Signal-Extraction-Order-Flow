"""Checks on the reversal re-estimated on common support with the two classes'
net participation rates matched and inside the size distribution, measured by the
scripts of code/empirical/robustness/."""

from __future__ import annotations

import json
import math
import re

from shared_utils import CRIT_1PCT, CRIT_5PCT

from output._formatters import BPS_PER_RETURN
from output.assertions.common import (
    PAPER_TEX,
    RESULTS,
    ROOT,
    assert_prose_close,
    assert_true,
    parse_num,
)


def restored_levels(sz: dict) -> bool:
    """Whether the small arm's levels cell resolves the advantage for foreign flow with the
    predicted sign at the paper's 5% two-sided bar."""
    cell = sz["reversal"]["small"]["zscore"]["fama_macbeth_univariate"]["advantage_foreign"]
    return cell["mean"] < 0 and cell["t"] < -CRIT_5PCT


def check_prose_size_restricted(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:size_restricted, subsec:reversal_test,
    subsec:limitations; qf_paper.tex subsec:appendix_size_restricted.

    The section argues that the three perturbations weakening the advantage for foreign flow
    on common support are reading that sample's size composition. Five things
    carry it. (i) The unrestricted arm reproduces the published cells, without
    which none of the restricted ones is comparable to them. (ii) The advantage
    for foreign flow is stronger below the size boundary than on the whole of common
    support and absent above it. (iii) The count-matched draws put the
    restricted estimate several draw standard deviations below their mean, which
    is what makes the movement size rather than sample size. (v) The
    within-arm criterion, committed before the run, returns its verdict as it
    fell. (vi) The reversal clears the bar inside the large
    size subsample in all three rows, with only the advantage for capacity-bound flow significant,
    which is what licenses subsec:size_restricted, subsec:limitations, and the
    abstract in denying that the contrast is a small-stock phenomenon.

    The composition the appendix prints is also read against the population it
    names. "The bottom subsample, the bottom \\sizSmallQuintiles quintiles, holds
    \\sizSmallStockDays stock-days (... of
    common support" makes the two arms a partition of common support, so their
    counts add up to \\revCommonStockDays, and each arm stacks two class rows
    per stock-day. The arms once counted the overlap before the return filter,
    837 stock-days more than common support, and the sum disagreed with the
    count the reversal section prints.
    """
    sz = json.loads((RESULTS / "size_restricted_common.json").read_text())
    rk = json.loads((RESULTS / "rank_transform_robustness.json").read_text())
    cem = json.loads((RESULTS / "participation_matching_results.json").read_text())

    comp = sz["composition"]
    support = parse_num(m["revCommonStockDays"])
    for arm in ("all", "small", "large"):
        assert_true(
            comp[arm]["n_stacked_rows"] == 2 * comp[arm]["n_stock_days"],
            claim=f"the {arm} size arm stacks both classes on each of its "
                  f"stock-days (subsec:appendix_size_restricted)",
            detail=f"{comp[arm]['n_stacked_rows']:,} stacked rows against "
                   f"{comp[arm]['n_stock_days']:,} stock-days")
    assert_true(
        comp["all"]["n_stock_days"] == support
        and comp["small"]["n_stock_days"] + comp["large"]["n_stock_days"] == support,
        claim="the small and large subsamples of subsec:appendix_size_restricted "
              "partition common support, the \\revCommonStockDays stock-days that "
              "subsec:reversal_test prints",
        detail=f"small {comp['small']['n_stock_days']:,} + large "
               f"{comp['large']['n_stock_days']:,} against all "
               f"{comp['all']['n_stock_days']:,} and common support {support:,.0f}")

    # (i) The unrestricted arm is the published one.
    for label, published in (
            ("zscore", rk["stacked"]["common"]["zscore"]["fama_macbeth_univariate"]),
            ("rank", rk["stacked"]["common"]["rank"]["fama_macbeth_univariate"])):
        here = sz["reversal"]["all"][label]["fama_macbeth_univariate"]
        for key in ("advantage_capacity", "advantage_foreign", "reversal"):
            assert_prose_close(
                here[key]["mean"] * 1e4, published[key]["mean"] * 1e4, tol=0.01,
                claim=f"subsec:size_restricted holds common support fixed, so its "
                      f"unrestricted {label} arm must reproduce the {key} of "
                      f"subsec:appendix_j_rank")
    matched_here = sz["matching"]["all"]["matched"]["fama_macbeth_univariate"]
    matched_pub = cem["common"]["quintile"]["matched"]["fama_macbeth_univariate"]
    assert_prose_close(
        matched_here["advantage_foreign"]["mean"] * 1e4,
        matched_pub["advantage_foreign"]["mean"] * 1e4, tol=0.01,
        claim="subsec:size_restricted's unrestricted matched arm must reproduce the "
              "advantage for foreign flow of subsec:appendix_participation_matching")

    # (ii) The finding: stronger below the boundary, absent above it.
    small = sz["reversal"]["small"]["zscore"]["fama_macbeth_univariate"]
    large = sz["reversal"]["large"]["zscore"]["fama_macbeth_univariate"]
    full = sz["reversal"]["all"]["zscore"]["fama_macbeth_univariate"]
    assert_true(
        small["advantage_foreign"]["mean"] < full["advantage_foreign"]["mean"]
        and small["advantage_foreign"]["t"] < -CRIT_1PCT,
        claim="subsec:size_restricted says the advantage for foreign flow in levels below the size "
              "boundary is significant at the 1% level and larger in magnitude than on "
              "the whole of common support",
        detail=f"small {small['advantage_foreign']['mean'] * 1e4:+.2f} bp "
               f"(t = {small['advantage_foreign']['t']:+.2f}) against "
               f"{full['advantage_foreign']['mean'] * 1e4:+.2f} bp full")
    assert_true(
        abs(large["advantage_foreign"]["t"]) < CRIT_5PCT,
        claim="subsec:size_restricted and subsec:limitations both say the advantage "
              "for foreign flow is absent in the large size subsample",
        detail=f"large {large['advantage_foreign']['mean'] * 1e4:+.2f} bp "
               f"(t = {large['advantage_foreign']['t']:+.2f})")

    # (iii) The count-matched control, which is what licenses reading (ii) as size.
    res = sz["resample"]["advantage_foreign"]
    assert_true(
        res["small_arm_z"] < -CRIT_1PCT,
        claim="subsec:appendix_size_restricted says the restricted estimate lies "
              "\\sizResampleZ standard deviations below the mean of the count-matched "
              "draws, which it reads as the movement not being the loss of cross-section",
        detail=f"small arm {res['small_arm_mean'] * 1e4:+.2f} bp, "
               f"z = {res['small_arm_z']:+.2f} over {sz['resample']['n_draws']} draws")
    for value, macro in ((res["mean_of_means"] * 1e4, "sizResampleMean"),
                         (res["sd_of_means"] * 1e4, "sizResampleSd"),
                         (abs(res["small_arm_z"]), "sizResampleZ")):
        assert_prose_close(value, parse_num(m[macro]), tol=0.01,
                           claim=f"\\{macro} ({m[macro]}) is a count-matched draw "
                                 f"statistic of subsec:appendix_size_restricted")

    # (v) The criterion committed before the run, reported as it fell.
    assert_true(
        sz["verdict"]["reading"] == "bound"
        and not any(sz["verdict"]["restored"].values()),
        claim="subsec:size_restricted says the advantage for foreign flow is not significant "
              "under the rank transform or the matching inside the restricted arm and "
              "the exponent profile locates no gap, the outcome of the criterion "
              "committed before the run",
        detail=f"restored {sz['verdict']['restored']}, reading "
               f"{sz['verdict']['reading']!r}")
    # The committed rule grades the rank transform, the matching, and the
    # exponent profile, not the primary specification, which the restricted arm
    # does resolve. The sentence reporting the outcome has to name the
    # estimates the verdict was read on, or the outcome reads as a claim about
    # levels that the data contradict.
    assert_true(
        set(sz["verdict"]["restored"]) == {"rank", "matched", "free_exponent"}
        and restored_levels(sz),
        claim="the criterion graded the rank transform, the matching and the free "
              "exponent, and the primary specification resolves the advantage for foreign flow "
              "inside the bottom subsample",
        detail=f"graded {sorted(sz['verdict']['restored'])}")
    text = PAPER_TEX.read_text()
    sentences = re.split(r"(?<=[.])\s+", text)
    crit = [s for s in sentences
            if "bottom" in s
            and re.search(r"advantage for foreign flow\b.*\bnot significant", s)]
    expo = [s for s in sentences
            if "exponent profile" in s and "does not locate" in s]
    assert_true(
        crit and expo and all("rank transform" in s and "matching" in s
                              for s in crit),
        claim="the sentences reporting the size criterion's outcome name the rank "
              "transform, the matching and the exponent profile, the designs the "
              "committed rule grades",
        detail=f"{len(crit)} significance sentence(s), {len(expo)} exponent "
               f"sentence(s) found")
    # (vi) The large arm, which is what answers the small-stock reading of the
    # whole contrast in subsec:size_restricted and in the abstract.
    for label, cell in (("levels", sz["reversal"]["large"]["zscore"]),
                        ("rank", sz["reversal"]["large"]["rank"]),
                        ("matching", sz["matching"]["large"]["matched"])):
        fit = cell["fama_macbeth_univariate"]
        assert_true(
            fit["reversal"]["t"] > CRIT_1PCT,
            claim=f"subsec:size_restricted says the contrast is significant at the 1% "
                  f"level inside the large size subsample under the {label} "
                  f"specification, so the contrast is not confined to small stocks",
            detail=f"{fit['reversal']['mean'] * 1e4:+.2f} bp "
                   f"(t = {fit['reversal']['t']:+.2f})")
    for label, cell in (("levels", sz["reversal"]["large"]["zscore"]),
                        ("rank", sz["reversal"]["large"]["rank"]),
                        ("matching", sz["matching"]["large"]["matched"])):
        fit = cell["fama_macbeth_univariate"]
        assert_true(
            fit["advantage_capacity"]["t"] > CRIT_5PCT
            and abs(fit["advantage_foreign"]["t"]) < CRIT_5PCT,
            claim=f"subsec:size_restricted says only the advantage for capacity-bound flow is "
                  f"significant in the top size quintiles under the {label} "
                  f"specification, and the abstract says only the institutional "
                  f"advantage is significant alone there",
            detail=f"capacity {fit['advantage_capacity']['mean'] * 1e4:+.2f} bp "
                   f"(t = {fit['advantage_capacity']['t']:+.2f}), foreign "
                   f"{fit['advantage_foreign']['mean'] * 1e4:+.2f} bp "
                   f"(t = {fit['advantage_foreign']['t']:+.2f})")

    # "Inside the subsamples, the exponent profile ..., estimated in levels, does
    # not locate the gap between the two classes' best exponents, because the
    # gap's interval extends below zero in both, down to $\sizSmallGapLo$ in the
    # bottom subsample and $\sizLargeGapLo$ in the top"
    # (subsec:appendix_size_restricted).
    for arm, macro in (("small", "sizSmallGapLo"), ("large", "sizLargeGapLo")):
        ci = sz["free_exponent"][arm]["gap_levels"]["ci"]
        assert_true(
            ci[0] < 0.0,
            claim=f"subsec:appendix_size_restricted says the exponent profile does not "
                  f"locate the gap between the two classes' best exponents in the {arm} "
                  f"subsample, whose gap interval extends below zero (\\{macro})",
            detail=f"gap interval [{ci[0]:+.2f}, {ci[1]:+.2f}]")
        assert_prose_close(ci[0], parse_num(m[macro]), tol=0.005,
                           claim=f"the {arm} subsample's gap interval reaches down to "
                                 f"\\{macro} ({m[macro]})")


def check_prose_participation_matching(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:reversal_test; qf_paper.tex
    subsec:appendix_participation_matching, tab:participation_matching.

    The two classes place their flow at different points of the participation-
    rate distribution, so the section's claim is that the reversal is the
    carrier each filter scales with rather than the shape of either class's
    flow. Six things are pinned. (i) The full arm reproduces
    subsec:reversal_test exactly, which is what licenses reading the matched
    arm as a perturbation of it. (ii) The imbalance runs the direction the
    prose states, with the capacity class taking the smaller share of the
    day's trading, and turnover already balanced on common support because the
    stacking shares the firm-day. (iii) The matching closes it, and no weight
    exceeds one, so the retained population is the overlap rather than an
    extrapolation. (iv) The contrast keeps its sign and clears the bar on both
    stackings and under both grids. (v) The advantage for capacity-bound flow survives and the
    advantage for foreign flow does not, on common support under either grid, which is the
    cost the section discloses. (vi) The macros read the univariate arm, the
    specification the headline reversal is estimated under
    (subsec:return_prediction).

    The criterion lives in the module that runs the estimate
    (code/empirical/robustness/participation_matching.py) and is committed before the run; asserted
    here so a regenerated panel re-reports the verdict rather than leaving it
    stale."""
    pm = json.loads((RESULTS / "participation_matching_results.json").read_text())
    ri = json.loads((RESULTS / "reversal_interaction_results.json").read_text())
    rk = json.loads((RESULTS / "rank_transform_robustness.json").read_text())
    common, union = pm["common"], pm["union"]
    grid, fine = common["quintile"], common["decile"]
    bar = pm["config"]["balance_bar"]

    # (i) Same specification, same rows, so the two scripts must agree.
    for sample in ("common", "union"):
        full, ref = pm[sample]["full"], ri[sample]
        assert_true(full["n_rows"] == ref["n_rows"],
                    claim=f"the matching script's unmatched arm is the "
                          f"subsec:reversal_test specification ({sample} sample)",
                    detail=f"rows {full['n_rows']} vs {ref['n_rows']}")
        for arm in ("fama_macbeth_univariate", "fama_macbeth"):
            assert_true(full[arm]["reversal"]["mean"] == ref[arm]["reversal"]["mean"],
                        claim=f"the matching script reproduces the published reversal "
                              f"before matching ({arm}, {sample} sample)",
                        detail=f"{full[arm]['reversal']['mean']!r} vs "
                               f"{ref[arm]['reversal']['mean']!r}")

    # (ii) "Capacity-bound institutions take the smaller share of the day's
    # trading when they trade, not the larger one." The direction is the half
    # of the objection's premise the panel contradicts, so it is asserted and
    # not only quoted.
    gap = common["balance_full"]["logIntensity"]
    assert_true(gap["capacity_mean"] < gap["foreign_mean"],
                claim="capacity-bound institutions take the smaller share of the "
                      "day's trading when they trade, not the larger one",
                detail=f"log |D|/V: capacity {gap['capacity_mean']:.4f}, "
                       f"foreign {gap['foreign_mean']:.4f}")
    for macro, value in (("cemIntensityCap", gap["capacity_mean"]),
                         ("cemIntensityFor", gap["foreign_mean"])):
        assert_prose_close(100 * math.exp(value), parse_num(m[macro]), tol=0.05,
                           claim=f"the geometric mean participation rate is "
                                 f"\\{macro}\\% ({m[macro]}\\%)")
    assert_prose_close(gap["std_diff"], parse_num(m["cemGapFull"]), tol=0.01,
                       claim=f"a standardised class difference of \\cemGapFull "
                             f"({m['cemGapFull']}) standard deviations")

    # "Turnover is shared row by row on this stacking", which is what makes the
    # participation rate the one axis the two classes separate on. Common
    # support puts both classes on the same firm-days, so the difference is
    # zero by construction and a nonzero value means the stacking drifted.
    tau = common["balance_full"]["logTurnover"]
    assert_true(tau["std_diff"] == 0.0,
                claim="turnover is shared row by row on this stacking",
                detail=f"standardised class difference {tau['std_diff']!r}")

    # (iii) "The standardised class difference on the participation rate falls
    # from \cemGapFull ... to \cemGapMatched ... and \cemGapDecile", and the
    # capacity filter's magnitude "follows it".
    for col, full_macro, q_macro, d_macro in (
            ("logIntensity", "cemGapFull", "cemGapMatched", "cemGapDecile"),
            ("logCapMagnitude", "cemMagGapFull", "cemMagGapMatched", "cemMagGapDecile")):
        full_gap = common["balance_full"][col]["std_diff"]
        q_gap = grid["balance_matched"][col]["std_diff"]
        d_gap = fine["balance_matched"][col]["std_diff"]
        for value, macro in ((full_gap, full_macro), (q_gap, q_macro), (d_gap, d_macro)):
            assert_prose_close(value, parse_num(m[macro]), tol=0.01,
                               claim=f"\\{macro} ({m[macro]}) is the standardised "
                                     f"class difference on {col}")
        assert_true(abs(d_gap) < abs(q_gap) < abs(full_gap),
                    claim=f"the matching closes the class difference on {col}, and "
                          f"the finer grid closes more of it",
                    detail=f"full {full_gap:.3f}, quintile {q_gap:.3f}, "
                           f"decile {d_gap:.3f}")
    assert_prose_close(bar, parse_num(m["cemBalanceBar"]), tol=1e-9,
                       claim=f"the \\cemBalanceBar ({m['cemBalanceBar']}) a matched "
                             f"design is conventionally asked to clear")
    # "The quintile grid therefore sits at the bar on the rate and outside it on
    # the magnitude, and the decile grid is inside it on both."
    assert_true(abs(fine["balance_matched"]["logIntensity"]["std_diff"]) < bar
                and abs(fine["balance_matched"]["logCapMagnitude"]["std_diff"]) < bar,
                claim="the decile grid is inside the bar on both",
                detail=f"rate {fine['balance_matched']['logIntensity']['std_diff']:.3f}, "
                       f"magnitude "
                       f"{fine['balance_matched']['logCapMagnitude']['std_diff']:.3f}")
    assert_true(abs(grid["balance_matched"]["logCapMagnitude"]["std_diff"]) > bar,
                claim="the quintile grid sits outside the bar on the magnitude",
                detail=f"{grid['balance_matched']['logCapMagnitude']['std_diff']:.3f} "
                       f"against {bar}")

    # "No weight exceeds one, so the retained population is the overlap of the
    # two flow distributions rather than an extrapolation of either." A weight
    # above one would inflate a cell, which the min() in cem_weights forbids;
    # the summed weights are what would show it.
    for sample, arms in (("common", common), ("union", union)):
        for name in pm["config"]["grids"]:
            for cls, ess in arms[name]["effective_size"].items():
                assert_true(ess["sum_weights"] <= ess["n_rows"],
                            claim=f"no weight exceeds one ({sample} sample, {name} "
                                  f"grid, {cls})",
                            detail=f"summed {ess['sum_weights']:.1f} over "
                                   f"{ess['n_rows']} rows")

    def rev(arm):
        return arm["fama_macbeth_univariate"]["reversal"]

    # (iv) "Under both grids and on both stackings, the contrast remains
    # significant at the 1% level in the univariate specification and in the
    # horse race" (subsec:appendix_participation_matching).
    for sample, arms in (("common", common), ("union", union)):
        for name in pm["config"]["grids"]:
            for spec in ("fama_macbeth_univariate", "fama_macbeth"):
                fm = arms[name]["matched"][spec]["reversal"]
                assert_true(fm["mean"] > 0 and fm["t"] > CRIT_1PCT,
                            claim=f"the contrast remains significant at the 1% level "
                                  f"({sample} sample, {name} grid, {spec})",
                            detail=f"b4-b3={fm['mean']:.6f}, t={fm['t']:.2f}")

    for label, arm, rev_key, t_key in (
            ("the quintile grid", grid["matched"], "cemRev", "cemT"),
            ("unit weights on the retained rows", grid["overlap"],
             "cemOverlapRev", "cemOverlapT"),
            ):
        fm = rev(arm)
        assert_prose_close(fm["mean"] * BPS_PER_RETURN, parse_num(m[rev_key]), tol=0.01,
                           claim=f"the contrast is \\{rev_key} ({m[rev_key]}) basis "
                                 f"points under {label}")
        assert_prose_close(fm["t"], parse_num(m[t_key]), tol=0.01,
                           claim=f"$t = \\{t_key}$ ({m[t_key]}) under {label}")

    # "... so the re-weighting rather than the dropping of single-class cells
    # moves the estimate."
    # The overlap arm is the restriction without the re-balancing, so it has to
    # sit far closer to the unmatched estimate than the matched arm does.
    base = rev(common["full"])["mean"]
    drop = abs(rev(grid["overlap"])["mean"] - base)
    whole = abs(rev(grid["matched"])["mean"] - base)
    assert_true(drop < 0.1 * whole,
                claim="the re-weighting rather than the dropping of single-class "
                      "cells moves the estimate",
                detail=f"restriction moves {drop * BPS_PER_RETURN:.3f} bps of the "
                       f"{whole * BPS_PER_RETURN:.3f} bps the design moves")

    # (v) The two advantages. The capacity one survives the matching; the foreign
    # one does not on common support, which is what the section discloses.
    for cls, key, edge_macro, t_macro in (
            ("Cap", "advantage_capacity", "cemCapEdge", "cemCapEdgeT"),
            ("For", "advantage_foreign", "cemForEdge", "cemForEdgeT")):
        edge = grid["matched"]["fama_macbeth_univariate"][key]
        assert_prose_close(edge["mean"] * BPS_PER_RETURN, parse_num(m[edge_macro]),
                           tol=0.01,
                           claim=f"\\{edge_macro} ({m[edge_macro]}) basis points on "
                                 f"the matched cells")
        assert_prose_close(edge["t"], parse_num(m[t_macro]), tol=0.01,
                           claim=f"$t = \\{t_macro}$ ({m[t_macro]})")
    cap = grid["matched"]["fama_macbeth_univariate"]["advantage_capacity"]
    assert_true(cap["mean"] > 0 and cap["t"] > CRIT_1PCT,
                claim="the advantage for capacity-bound flow does not move with it",
                detail=f"{cap['mean'] * BPS_PER_RETURN:+.2f} bps, t={cap['t']:.2f}")

    for name, arms in (("quintile", grid), ("decile", fine)):
        edge = arms["matched"]["fama_macbeth_univariate"]["advantage_foreign"]
        assert_true(abs(edge["t"]) < CRIT_5PCT,
                    claim=f"the advantage for foreign flow is no longer distinguishable "
                          f"from zero on common support ({name} grid)",
                    detail=f"{edge['mean'] * BPS_PER_RETURN:+.2f} bps, "
                           f"t={edge['t']:.2f}")
    for name in pm["config"]["grids"]:
        union_for = union[name]["matched"]["fama_macbeth_univariate"]["advantage_foreign"]
        assert_true(union_for["mean"] < 0 and union_for["t"] < -CRIT_5PCT,
                    claim=f"it remains distinguishable from zero on the union ({name} grid)",
                    detail=f"{union_for['mean'] * BPS_PER_RETURN:+.2f} bps, "
                           f"t={union_for['t']:.2f}")

    # "On common support the loss rises with the balance the grid achieves. On
    # the union the finer grid does not cost the advantage for foreign flow more than the
    # quintile grid does."
    q_t = {s: abs(a["quintile"]["matched"]["fama_macbeth_univariate"]["advantage_foreign"]["t"])
           for s, a in (("common", common), ("union", union))}
    d_t = {s: abs(a["decile"]["matched"]["fama_macbeth_univariate"]["advantage_foreign"]["t"])
           for s, a in (("common", common), ("union", union))}
    assert_true(d_t["common"] < q_t["common"],
                claim="on common support the loss rises with the balance the grid achieves",
                detail=f"quintile |t|={q_t['common']:.2f}, decile |t|={d_t['common']:.2f}")
    assert_true(d_t["union"] >= q_t["union"],
                claim="on the union the finer grid does not cost the advantage for foreign flow more "
                      "than the quintile grid does",
                detail=f"quintile |t|={q_t['union']:.2f}, decile |t|={d_t['union']:.2f}")

    # "So part of that loss is precision." The Kish count is what says an unmatched
    # standard error is not the one the matched estimate carries.
    for cls, tag in (("capacity_bound", "Cap"), ("foreign", "For")):
        ess = grid["effective_size"][cls]
        assert_true(ess["effective_n"] < ess["n_rows"],
                    claim=f"a matched estimate carries a wider standard error than "
                          f"an unmatched estimate on the same rows ({cls})",
                    detail=f"effective {ess['effective_n']:,.0f} of "
                           f"{ess['n_rows']:,} rows")
        assert_prose_close(100 * ess["effective_n"] / ess["n_rows"],
                           parse_num(m[f"cemEss{tag}Pct"]), tol=0.5,
                           claim=f"\\cemEss{tag}Pct\\% ({m[f'cemEss{tag}Pct']}\\%) of "
                                 f"the raw {cls} count")

    # qf_paper.tex subsec:contrast_halves: "Neither perturbation that removes
    # the advantage for foreign flow removes it from the union stacking", and "The two
    # perturbations of the signal distribution leave it where it is" of the
    # advantage for capacity-bound flow. The rank arm is the other perturbation, so both files are
    # read.
    for label, perturbed in (
            ("matching", {s: pm[s]["quintile"]["matched"]["fama_macbeth_univariate"]
                          for s in ("common", "union")}),
            ("ranks", {s: rk["stacked"][s]["rank"]["fama_macbeth_univariate"]
                       for s in ("common", "union")})):
        assert_true(abs(perturbed["common"]["advantage_foreign"]["t"]) < CRIT_5PCT
                    and abs(perturbed["union"]["advantage_foreign"]["t"]) > CRIT_5PCT,
                    claim=f"the advantage for foreign flow under {label} is removed on common "
                          f"support and \"does not [leave] the union stacking\"",
                    detail=f"common t="
                           f"{perturbed['common']['advantage_foreign']['t']:+.2f}, "
                           f"union t="
                           f"{perturbed['union']['advantage_foreign']['t']:+.2f}")
        assert_true(perturbed["common"]["advantage_capacity"]["t"] > CRIT_5PCT
                    and perturbed["union"]["advantage_capacity"]["t"] > CRIT_5PCT,
                    claim=f"the {label} perturbation \"leaves [the advantage for "
                          f"capacity-bound flow] where it is\", on both stackings",
                    detail=f"common t="
                           f"{perturbed['common']['advantage_capacity']['t']:+.2f}, "
                           f"union t="
                           f"{perturbed['union']['advantage_capacity']['t']:+.2f}")

    # (vi) The criterion, re-evaluated against the data it was
    # committed for.
    verdict = pm["verdict"]
    gate = rev(grid["matched"])
    assert_true(verdict["passed"] and gate["mean"] > 0
                and gate["t"] > pm["config"]["gate_min_t"],
                claim=f"the criterion of participation_matching.py, committed before the run, "
                      f"({verdict['criterion']}) is met",
                detail=f"REVERSAL={verdict['reversal']:.6f}, t={verdict['t']:.2f}")
