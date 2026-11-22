"""Checks on the extended robustness battery, measured by the scripts of
code/empirical/robustness/: out-of-sample and yearly stability, long-short
returns, factor neutrality, outliers and alternative standard errors, the
within-day rank transform and the market-cap threshold."""

from __future__ import annotations

import json
import re

import pandas as pd

from paths import TABLES_DIR
from shared_utils import CRIT_1PCT, CRIT_5PCT, CRIT_10PCT, END_DATE_MAIN, START_DATE_MAIN

from output._formatters import BPS_PER_RETURN
from output.assertions.common import (
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def check_extended_robustness_yearly(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex tab:robustness_summary; qf_paper.tex subsec:appendix_j_oos."""
    ext = json.loads((RESULTS / "extended_robustness_results.json").read_text())
    yrs = ext["yearly_subperiods"]
    hr_mc = [(int(y), d["horse_race"]["MC"]["t_stat"]) for y, d in sorted(yrs.items())]
    hr_tv = [(int(y), d["horse_race"]["TV"]["t_stat"]) for y, d in sorted(yrs.items())]
    mc_min, mc_max = min(hr_mc, key=lambda x: x[1]), max(hr_mc, key=lambda x: x[1])
    assert_close(m["ryMinHRMCt"], mc_min[1], tol=0.02, label="ryMinHRMCt")
    assert_close(m["ryMaxHRMCt"], mc_max[1], tol=0.02, label="ryMaxHRMCt")
    assert m["ryMinHRMCYear"] == str(mc_min[0]), f"ryMinHRMCYear={m['ryMinHRMCYear']} vs data={mc_min[0]}"
    assert m["ryMaxHRMCYear"] == str(mc_max[0]), f"ryMaxHRMCYear={m['ryMaxHRMCYear']} vs data={mc_max[0]}"
    assert int(m["ryNYears"]) == len(hr_mc), f"ryNYears={m['ryNYears']} vs len(yrs)={len(hr_mc)}"
    # Yearly stability of t_MC^HR > 0 — the data-consistency check that
    # would have caught Triad 1 if it existed when the test suite was written.
    for y, t in hr_mc:
        assert t > 0, f"horse-race t_MC went non-positive in {y} (t={t}); manuscript claims yearly positivity"
    # subsec:appendix_j_oos: "The mismatched coefficient is negative at the 5\%
    # level in \rySignifNegYears and not distinguishable from zero in
    # \ryInsignifYears". The macro year-lists are auto-generated; this
    # assertion forces the per-year data to agree with the partition rather
    # than trusting the generator alone, and a year significantly positive,
    # which the sentence rules out, falls in neither list.
    sig_years = {int(s) for s in re.findall(r"\d{4}", m["rySignifNegYears"])}
    insignif_years = {int(s) for s in re.findall(r"\d{4}", m["ryInsignifYears"])}
    for y, t in hr_tv:
        if y in sig_years:
            assert t <= -CRIT_5PCT, (
                f"\\rySignifNegYears claims {y} is 5%-significant negative, "
                f"but data t_TV^HR={t:.3f}")
        elif y in insignif_years:
            assert abs(t) < CRIT_5PCT, (
                f"\\ryInsignifYears claims {y} is not distinguishable from zero, "
                f"but data t_TV^HR={t:.3f}")
        else:
            raise AssertionError(
                f"year {y} (t_TV^HR={t:.3f}) is in neither \\rySignifNegYears nor "
                f"\\ryInsignifYears; the sentence in subsec:appendix_j_oos rules "
                f"out a significantly positive year")


def check_extended_robustness_foreign(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex tab:robustness_summary (foreign column);
    qf_paper.tex subsec:appendix_foreign_robustness."""
    ext = json.loads((RESULTS / "extended_robustness_foreign_results.json").read_text())
    # subsec:appendix_foreign_robustness: "On the pre-period, both foreign
    # normalized flows have positive coefficients in the univariate and the
    # horse-race specifications."
    pre = ext["pre_period_oos"]
    signs = {f"{spec} {sig}": pre[spec][sig]["coef"]
             for spec in ("univariate", "horse_race") for sig in ("MC", "TV")}
    assert_true(all(c > 0 for c in signs.values()),
                claim="both foreign normalized flows have positive coefficients on the "
                      "pre-period in the univariate and the horse-race specifications",
                detail=", ".join(f"{k} {v:+.6f}" for k, v in signs.items()))
    yrs = ext["yearly_subperiods"]
    hr_tv = [(int(y), d["horse_race"]["TV"]["t_stat"]) for y, d in sorted(yrs.items())]
    tv_min, tv_max = min(hr_tv, key=lambda x: x[1]), max(hr_tv, key=lambda x: x[1])
    assert_close(m["fryMinHRTVt"], tv_min[1], tol=0.02, label="fryMinHRTVt")
    assert_close(m["fryMaxHRTVt"], tv_max[1], tol=0.02, label="fryMaxHRTVt")
    assert m["fryMinHRTVYear"] == str(tv_min[0])
    assert m["fryMaxHRTVYear"] == str(tv_max[0])


def check_long_short(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex tab:robustness_summary; qf_paper.tex subsec:appendix_j_longshort."""
    ext = json.loads((RESULTS / "extended_robustness_results.json").read_text())
    ls = ext["long_short"]
    assert_close(m["lsMCbps"],    ls["S_MC"]["mean_bps"], tol=0.02, label="lsMCbps")
    assert_close(m["lsMCt"],      ls["S_MC"]["t_stat"],   tol=0.02, label="lsMCt")
    assert_close(m["lsTVbps"],    ls["S_TV"]["mean_bps"], tol=0.02, label="lsTVbps")
    fn = ext["factor_neutrality"]
    assert_close(m["fnAlpha"],  fn["alpha"]["coef"] * 10000, tol=0.02, label="fnAlpha")
    assert_close(m["fnAlphaT"], fn["alpha"]["t_stat"],       tol=0.02, label="fnAlphaT")
    assert_close(m["fnSMB"],    fn["SMB"]["coef"],           tol=0.005, label="fnSMB")
    assert_close(m["fnSMBt"],   fn["SMB"]["t_stat"],         tol=0.02, label="fnSMBt")
    # subsec:appendix_j_factor rests the size-premium rebuttal on the sign of
    # this loading, so the sign needs a significance bound rather than an eye.
    assert_true(fn["SMB"]["t_stat"] < -1.96,
                claim="The negative SMB loading indicates that the portfolio tilts toward "
                      "large stocks, opposite to a size premium explanation",
                detail=f"SMB t={fn['SMB']['t_stat']:.2f}")


def check_prose_rank_transform(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:appendix_j_rank; qf_paper.tex
    tab:robustness_summary (rank row), subsec:return_prediction (mismatched
    sign), subsec:reversal_test.

    The appendix asks whether the ordering claims are carried by a few extreme
    stock-days. Five things are pinned. (i) The level arm reproduces both
    published files exactly, which is what licenses reading the rank arm as a
    perturbation of them rather than as a different sample. (ii) Every
    univariate t-statistic rises in magnitude and each class keeps its matched
    filter ahead. (iii) The reversal survives on both stackings while its
    advantage for foreign flow survives on the union alone, which is the appendix's one
    honest exception. (iv) The two filters' within-day correlation rises under
    the transform, which is what makes the rank arm a floor on the advantage
    rather than a competing estimate of it. (v) The summary table sets each
    class's rank arm against a level base that another script prints, so the
    rank script's level arm must print as that base."""
    rk = json.loads((RESULTS / "rank_transform_robustness.json").read_text())
    mi = json.loads((RESULTS / "multi_investor_robustness_results.json").read_text())
    ri = json.loads((RESULTS / "reversal_interaction_results.json").read_text())
    per_class, stacked = rk["per_class"], rk["stacked"]

    # (i) Same specification, same rows, so the two scripts must agree exactly.
    for cls in rk["config"]["classes"]:
        for spec in ("univariate_MC", "univariate_TV", "horse_race_MC", "horse_race_TV"):
            mine = per_class[cls]["zscore"][spec]["t_stat"]
            ref = mi["return_predictions"][cls][spec]["t_stat"]
            assert_true(mine == ref,
                        claim=f"the rank script's level arm is the "
                              f"tab:investor_heterogeneity specification "
                              f"({cls}, {spec})",
                        detail=f"t {mine!r} vs {ref!r}")
    for sample in ("common", "union"):
        for arm in ("fama_macbeth_univariate", "fama_macbeth"):
            mine = stacked[sample]["zscore"][arm]["reversal"]["mean"]
            ref = ri[sample][arm]["reversal"]["mean"]
            assert_true(mine == ref,
                        claim=f"the rank script's level arm is the "
                              f"subsec:reversal_test specification "
                              f"({sample} sample, {arm})",
                        detail=f"{mine!r} vs {ref!r}")

    # (ii) "Every univariate t-statistic rises in magnitude, so the published
    # estimates are the conservative ones on significance, and each class keeps
    # its matched filter ahead of its mismatched one."
    for cls in rk["config"]["classes"]:
        for spec in ("univariate_MC", "univariate_TV"):
            level = abs(per_class[cls]["zscore"][spec]["t_stat"])
            rank = abs(per_class[cls]["rank"][spec]["t_stat"])
            assert_true(rank > level,
                        claim=f"every univariate t-statistic rises in magnitude "
                              f"({cls}, {spec})",
                        detail=f"level |t|={level:.2f}, rank |t|={rank:.2f}")
    for cls, matched, mismatched in (("capacity_bound", "univariate_MC", "univariate_TV"),
                                     ("foreign", "univariate_TV", "univariate_MC")):
        won = per_class[cls]["rank"][matched]["t_stat"]
        lost = per_class[cls]["rank"][mismatched]["t_stat"]
        assert_true(won > lost,
                    claim=f"each class keeps its matched filter ahead of its "
                          f"mismatched one ({cls})",
                    detail=f"matched t={won:.2f}, mismatched t={lost:.2f}")

    # "Each advantage on the union is the difference between the class's two
    # univariate slopes on its own nonzero-flow stock-days, so each assigned
    # class also keeps the larger slope on its matched filter."
    union_rank = stacked["union"]["rank"]["fama_macbeth_univariate"]
    for cls, key, matched, mismatched in (
            ("capacity_bound", "advantage_capacity", "univariate_MC", "univariate_TV"),
            ("foreign", "advantage_foreign", "univariate_TV", "univariate_MC")):
        slopes = per_class[cls]["rank"]
        diff = slopes["univariate_MC"]["mean"] - slopes["univariate_TV"]["mean"]
        assert_prose_close(union_rank[key]["mean"] * 1e4, diff * 1e4, tol=0.01,
                           claim=f"subsec:appendix_j_rank: the union advantage of {cls} "
                                 "is the difference between its two univariate slopes")
        assert_true(slopes[matched]["mean"] > slopes[mismatched]["mean"],
                    claim=f"subsec:appendix_j_rank: {cls} keeps the larger univariate "
                          "slope on its matched filter",
                    detail=f"matched {slopes[matched]['mean']:+.6f}, "
                           f"mismatched {slopes[mismatched]['mean']:+.6f}")

    for macro, cls, spec in (("rankCapMCt", "capacity_bound", "univariate_MC"),
                             ("rankCapTVt", "capacity_bound", "univariate_TV"),
                             ("rankCapHRMCt", "capacity_bound", "horse_race_MC"),
                             ("rankCapHRTVt", "capacity_bound", "horse_race_TV"),
                             ("rankForMCt", "foreign", "univariate_MC"),
                             ("rankForTVt", "foreign", "univariate_TV"),
                             ("rankForHRMCt", "foreign", "horse_race_MC"),
                             ("rankForHRTVt", "foreign", "horse_race_TV")):
        assert_prose_close(per_class[cls]["rank"][spec]["t_stat"], parse_num(m[macro]),
                           tol=0.01,
                           claim=f"\\{macro} ({m[macro]}) is the rank-arm t-statistic "
                                 f"for {cls}, {spec}")

    # (v) tab:robustness_summary prints each class's rank row against a level
    # base that another script's macro carries, so the pair shares one sample
    # only if the rank script's level arm prints as that macro.
    for macro, cls, spec in (("rpHRMCtone", "capacity_bound", "horse_race_MC"),
                             ("ffBaseTVhrt", "foreign", "horse_race_TV"),
                             ("ffBaseMChrt", "foreign", "horse_race_MC")):
        assert_prose_close(per_class[cls]["zscore"][spec]["t_stat"], parse_num(m[macro]),
                           tol=0.005,
                           claim=f"tab:robustness_summary's rank row reads \\{macro} "
                                 f"({m[macro]}) as the level arm for {cls}, {spec}")

    # "The capacity-bound mismatched filter holds its negative sign and
    # strengthens ..., while the foreign mismatched filter loses its negative
    # sign", the same pair qf_paper.tex subsec:return_prediction now reads as a
    # reason to leave the residual uninterpreted.
    cap_hr_tv = per_class["capacity_bound"]["rank"]["horse_race_TV"]["t_stat"]
    assert_true(cap_hr_tv < -CRIT_1PCT
                and cap_hr_tv < per_class["capacity_bound"]["zscore"]["horse_race_TV"]["t_stat"],
                claim="the capacity-bound mismatched filter holds its negative sign "
                      "and strengthens",
                detail=f"level t="
                       f"{per_class['capacity_bound']['zscore']['horse_race_TV']['t_stat']:.2f}, "
                       f"rank t={cap_hr_tv:.2f}")
    for_hr_mc = per_class["foreign"]["rank"]["horse_race_MC"]["t_stat"]
    assert_true(for_hr_mc > -CRIT_5PCT,
                claim="the foreign mismatched filter loses its negative sign",
                detail=f"level t="
                       f"{per_class['foreign']['zscore']['horse_race_MC']['t_stat']:.2f}, "
                       f"rank t={for_hr_mc:.2f}")

    # (iii) "The reversal survives on both stackings, at \rankCommonRev basis
    # points ... and \rankUnionRev ..., roughly two-fifths of its magnitude in
    # levels at an unchanged order of significance." 
    # (subsec:appendix_j_rank; qf_paper.tex subsec:reversal_test states the same
    # fraction, and neither print is a macro, so this band is what pins both.)
    for sample, rev_key, t_key in (("common", "rankCommonRev", "rankCommonT"),
                                   ("union", "rankUnionRev", "rankUnionT")):
        uni = stacked[sample]["rank"]["fama_macbeth_univariate"]
        rev = uni["reversal"]
        assert_prose_close(rev["mean"] * BPS_PER_RETURN, parse_num(m[rev_key]), tol=0.01,
                           claim=f"the contrast is \\{rev_key} ({m[rev_key]}) basis "
                                 f"points under the rank transform ({sample} sample)")
        assert_prose_close(rev["t"], parse_num(m[t_key]), tol=0.01,
                           claim=f"at t = \\{t_key} ({m[t_key]}) ({sample} sample)")
        assert_true(rev["mean"] > 0 and rev["t"] > CRIT_1PCT,
                    claim=f"the reversal survives on both stackings ({sample} sample)",
                    detail=f"b4-b3={rev['mean']:.6f}, t={rev['t']:.2f}")
        ratio = rev["mean"] / stacked[sample]["zscore"]["fama_macbeth_univariate"]["reversal"]["mean"]
        assert_true(0 < ratio < 0.50,
                    claim=f"'less than half of its magnitude in levels' "
                          f"({sample} sample)",
                    detail=f"rank / level = {ratio:.3f}")

    # "Its advantage for capacity-bound flow survives with it ... Its advantage for foreign flow does not
    # survive on common support ... and does survive on the union."
    for sample, edge_key, t_key in (("common", "rankCommonCapEdge", "rankCommonCapEdgeT"),
                                    ("union", "rankUnionCapEdge", "rankUnionCapEdgeT")):
        cap = stacked[sample]["rank"]["fama_macbeth_univariate"]["advantage_capacity"]
        assert_prose_close(cap["mean"] * BPS_PER_RETURN, parse_num(m[edge_key]), tol=0.01,
                           claim=f"\\{edge_key} ({m[edge_key]}) basis points for "
                                 f"capacity-bound institutions ({sample} sample)")
        assert_prose_close(cap["t"], parse_num(m[t_key]), tol=0.01,
                           claim=f"at t = \\{t_key} ({m[t_key]}) ({sample} sample)")
        assert_true(cap["mean"] > 0 and cap["t"] > CRIT_1PCT,
                    claim=f"the reversal's advantage for capacity-bound flow survives with it "
                          f"({sample} sample)",
                    detail=f"{cap['mean'] * BPS_PER_RETURN:+.2f} bps, t={cap['t']:.2f}")

    common_for = stacked["common"]["rank"]["fama_macbeth_univariate"]["advantage_foreign"]
    union_for = stacked["union"]["rank"]["fama_macbeth_univariate"]["advantage_foreign"]
    assert_prose_close(common_for["mean"] * BPS_PER_RETURN,
                       parse_num(m["rankCommonForEdge"]), tol=0.01,
                       claim=f"\\rankCommonForEdge ({m['rankCommonForEdge']}) basis points "
                             f"for the advantage for foreign flow on common support")
    assert_prose_close(common_for["t"], parse_num(m["rankCommonForEdgeT"]), tol=0.01,
                       claim=f"at t = \\rankCommonForEdgeT ({m['rankCommonForEdgeT']})")
    assert_true(abs(common_for["t"]) < CRIT_5PCT,
                claim="its advantage for foreign flow does not survive on common support",
                detail=f"{common_for['mean'] * BPS_PER_RETURN:+.2f} bps, "
                       f"t={common_for['t']:.2f}")
    assert_prose_close(union_for["mean"] * BPS_PER_RETURN,
                       parse_num(m["rankUnionForEdge"]), tol=0.01,
                       claim=f"\\rankUnionForEdge ({m['rankUnionForEdge']}) on the union")
    assert_prose_close(union_for["t"], parse_num(m["rankUnionForEdgeT"]), tol=0.01,
                       claim=f"at t = \\rankUnionForEdgeT ({m['rankUnionForEdgeT']})")
    assert_true(union_for["mean"] < 0 and union_for["t"] < -CRIT_1PCT,
                claim="and does survive on the union",
                detail=f"{union_for['mean'] * BPS_PER_RETURN:+.2f} bps, "
                       f"t={union_for['t']:.2f}")

    # (iv) "It rises from \rankCorrCapZ to \rankCorrCapRank for capacity-bound
    # institutions and from \rankCorrForZ to \rankCorrForRank for foreign
    # investors." The rise is what makes the rank arm a floor, so a fall would
    # invalidate the paragraph's reading rather than only its numbers.
    corr = stacked["common"]
    for cls, tag in (("capacity_bound", "Cap"), ("foreign", "For")):
        level = corr["zscore"]["filter_correlation"][cls]
        rank = corr["rank"]["filter_correlation"][cls]
        for value, macro in ((level, f"rankCorr{tag}Z"), (rank, f"rankCorr{tag}Rank")):
            assert_prose_close(value, parse_num(m[macro]), tol=0.005,
                               claim=f"\\{macro} ({m[macro]}) is the mean within-day "
                                     f"correlation of the two filters ({cls})")
        assert_true(rank > level,
                    claim=f"the within-day correlation rises under the transform, so "
                          f"the rank arm is a floor on the ordering ({cls})",
                    detail=f"level={level:.4f}, rank={rank:.4f}")


def check_prose_extended_robustness_inst(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex tab:robustness_summary; qf_paper.tex subsec:appendix_j_oos, subsec:appendix_j_outlier.

    Pre-period hold-out sign reversal + winsorised t-stats + Newey-West (5 and 10 lags).

    Each expected value is read from the corresponding macro in tables/numbers.tex
    rather than hardcoded as a literal: the prose, the macro, and this test
    move together when results regenerate. The directional/significance
    assertions remain literal because they encode the qualitative claim
    rather than a specific number.
    """
    er = json.loads((RESULTS / "extended_robustness_results.json").read_text())
    pp = er["pre_period_oos"]["horse_race"]
    assert_prose_close(pp["MC"]["t_stat"], parse_num(m["oosPPMChr"]), tol=0.05,
                       claim=f"Pre-period hold-out (2015-2019): t_MC^HR = \\oosPPMChr ({m['oosPPMChr']})")
    assert_prose_close(pp["TV"]["t_stat"], parse_num(m["oosPPTVhr"]), tol=0.05,
                       claim=f"Pre-period hold-out: t_TV^HR flips to \\oosPPTVhr ({m['oosPPTVhr']})")
    assert_true(pp["MC"]["t_stat"] > 0 > pp["TV"]["t_stat"],
                claim="Pre-period hold-out confirms sign reversal: MC>0, TV<0",
                detail=f"MC t={pp['MC']['t_stat']:.2f}, TV t={pp['TV']['t_stat']:.2f}")
    # subsec:appendix_j_oos: "The pattern matches the primary sample
    # (\\sampleYears) ... in sign, and both statistics clear the 1% level on each
    # window." The sign is asserted above; the level is asserted on both windows.
    headline_mc = parse_num(m["rpHRMCtone"])
    headline_tv = parse_num(m["rpHRTVtone"])
    for window, mc, tv in (("pre-period", pp["MC"]["t_stat"], pp["TV"]["t_stat"]),
                           ("primary", headline_mc, headline_tv)):
        assert_true(mc > CRIT_1PCT and tv < -CRIT_1PCT,
                    claim=f"both horse-race statistics clear the 1% level on the "
                          f"{window} window, with the signs of the other",
                    detail=f"MC t={mc:.2f}, TV t={tv:.2f}")
    win = er["outlier_robustness"]["winsorized"]["horse_race"]
    base = er["outlier_robustness"]["baseline"]["horse_race"]
    assert_prose_close(win["MC"]["t_stat"], parse_num(m["winMCt"]), tol=0.05,
                       claim=f"winsorised horse-race MC t = \\winMCt ({m['winMCt']})")
    assert_prose_close(win["TV"]["t_stat"], parse_num(m["winTVt"]), tol=0.05,
                       claim=f"winsorised horse-race TV t = \\winTVt ({m['winTVt']})")
    # subsec:appendix_j_outlier claims both horse-race statistics increase, so
    # both are pinned. Winsorization is applied within each day, matching the
    # day-standardised regression it perturbs.
    for signal in ("MC", "TV"):
        assert_true(abs(win[signal]["t_stat"]) > abs(base[signal]["t_stat"]),
                    claim=f"Winsorizing strengthens rather than weakens results "
                          f"(horse-race {signal})",
                    detail=f"baseline {signal} t={base[signal]['t_stat']:.2f}, "
                           f"winsorised {signal} t={win[signal]['t_stat']:.2f}")
    nw = er["alternative_se"]
    assert_prose_close(nw["MC"]["nw10_t"], parse_num(m["nwTenMCt"]), tol=0.05,
                       claim=f"Newey-West (10 lags) t_MC^HR = \\nwTenMCt ({m['nwTenMCt']})")
    assert_prose_close(nw["TV"]["nw10_t"], parse_num(m["nwTenTVt"]), tol=0.05,
                       claim=f"Newey-West (10 lags) t_TV^HR = \\nwTenTVt ({m['nwTenTVt']})")
    # subsec:appendix_j_outlier: "Newey-West standard errors with 5 and 10 lags yield virtually
    # unchanged horse race t-statistics" — pin both NW5 values and the
    # "virtually unchanged" qualifier (|NW5 - NW10| < 0.5).
    assert_prose_close(nw["MC"]["nw5_t"], parse_num(m["nwFiveMCt"]), tol=0.05,
                       claim=f"Newey-West (5 lags) t_MC^HR = \\nwFiveMCt ({m['nwFiveMCt']})")
    assert_prose_close(nw["TV"]["nw5_t"], parse_num(m["nwFiveTVt"]), tol=0.05,
                       claim=f"Newey-West (5 lags) t_TV^HR = \\nwFiveTVt ({m['nwFiveTVt']})")
    assert_true(abs(nw["MC"]["nw5_t"] - nw["MC"]["nw10_t"]) < 0.5
                and abs(nw["TV"]["nw5_t"] - nw["TV"]["nw10_t"]) < 0.5,
                claim="Newey-West 5- and 10-lag t-statistics are 'virtually unchanged'",
                detail=f"MC: nw5={nw['MC']['nw5_t']:.3f}, nw10={nw['MC']['nw10_t']:.3f}; "
                       f"TV: nw5={nw['TV']['nw5_t']:.3f}, nw10={nw['TV']['nw10_t']:.3f}")


def check_prose_extended_robustness_foreign(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex tab:robustness_summary (foreign column); qf_paper.tex subsec:appendix_foreign_robustness.

    Foreign long-short + factor-neutral alpha.

    Expected values read from \\fLs*/\\fFn* macros so the test moves in lockstep
    with the prose.
    """
    er = json.loads((RESULTS / "extended_robustness_foreign_results.json").read_text())
    ls = er["long_short"]["S_TV"]
    assert_prose_close(ls["mean_bps"], parse_num(m["fLsTVbps"]), tol=0.05,
                       claim=f"Foreign S^TV long-short \\fLsTVbps ({m['fLsTVbps']}) bps daily")
    assert_prose_close(ls["t_stat"], parse_num(m["fLsTVt"]), tol=0.05,
                       claim=f"Foreign S^TV long-short t = \\fLsTVt ({m['fLsTVt']})")
    fn = er["factor_neutrality"]
    alpha_bps = fn["alpha"]["coef"] * 10000
    assert_prose_close(alpha_bps, parse_num(m["fFnAlpha"]), tol=0.1,
                       claim=f"Foreign factor-neutral alpha = \\fFnAlpha ({m['fFnAlpha']}) bps")
    assert_prose_close(fn["alpha"]["t_stat"], parse_num(m["fFnAlphaT"]), tol=0.05,
                       claim=f"Foreign factor-neutral alpha t = \\fFnAlphaT ({m['fFnAlphaT']})")
    # Significance bound for the foreign factor-neutral alpha prose
    # ("positive and highly significant"). 2.576 = two-sided 1% critical value.
    assert_true(abs(fn["alpha"]["t_stat"]) > 2.576,
                claim="Foreign factor-neutral alpha is highly significant",
                detail=f"|t_alpha|={abs(fn['alpha']['t_stat']):.3f} (threshold 2.576)")


def check_prose_mismatched_sign_tails(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex tab:investor_heterogeneity; qf_paper.tex
    tab:robustness_summary (the foreign cells of the winsorization and
    rank-transform rows); qf_paper.tex subsec:appendix_j_outlier and
    subsec:appendix_j_rank.

    The mismatched filter's horse-race coefficient is negative in both classes,
    S^TV for capacity-bound flow at the 1% level and S^MC for foreign flow at
    the 10% level (tab:investor_heterogeneity). subsec:appendix_j_outlier prints
    the capacity-bound S^TV coefficient under winsorization, and
    subsec:appendix_j_rank says that under the rank transform "the
    capacity-bound mismatched filter holds its negative sign and strengthens",
    whereas "the foreign mismatched filter loses its negative sign".
    tab:robustness_summary prints the foreign S^MC coefficient under both
    treatments.

    The macro *values* move with the generator, so what has to be asserted is
    the direction those places print: the foreign mismatched coefficient is
    negative at the 10% level before either tail treatment and at that level
    under neither, and the capacity-bound one clears the 1% level with its sign
    intact under both.
    """
    fr = json.loads((RESULTS / "extended_robustness_foreign_results.json").read_text())
    rk = json.loads((RESULTS / "rank_transform_robustness.json").read_text())["per_class"]

    for_base = fr["outlier_robustness"]["baseline"]["horse_race"]["MC"]["t_stat"]
    for_win = fr["outlier_robustness"]["winsorized"]["horse_race"]["MC"]["t_stat"]
    for_rank = rk["foreign"]["rank"]["horse_race_MC"]["t_stat"]
    cap_base = rk["capacity_bound"]["zscore"]["horse_race_TV"]["t_stat"]
    cap_rank = rk["capacity_bound"]["rank"]["horse_race_TV"]["t_stat"]
    cap_win = parse_num(m["winTVt"])

    assert_true(for_base < -CRIT_10PCT,
                claim="the foreign mismatched filter turns negative at the 10% level "
                      "before either tail treatment",
                detail=f"foreign horse-race MC t={for_base:.2f}")
    for label, t_stat in (("winsorisation", for_win), ("the rank transform", for_rank)):
        assert_true(t_stat > -CRIT_10PCT,
                    claim=f"the foreign mismatched coefficient is not negative at the "
                          f"10% level under {label} (tab:robustness_summary)",
                    detail=f"foreign horse-race MC t={t_stat:.2f} (10% bound "
                           f"{CRIT_10PCT:.3f})")
    assert_true(for_rank > CRIT_5PCT,
                claim="under the rank transform 'the foreign mismatched filter loses its "
                      "negative sign' (subsec:appendix_j_rank) and turns positive at the "
                      "5% level (tab:robustness_summary)",
                detail=f"foreign rank horse-race MC t={for_rank:.2f} (5% bound "
                       f"{CRIT_5PCT:.3f})")
    for label, t_stat in (("the baseline", cap_base), ("winsorisation", cap_win),
                          ("the rank transform", cap_rank)):
        assert_true(t_stat < 0 and abs(t_stat) > 2.576,
                    claim=f"the capacity-bound mismatched coefficient keeps its negative "
                          f"sign at the 1% level under {label} "
                          f"(subsec:appendix_j_outlier, subsec:appendix_j_rank)",
                    detail=f"capacity-bound horse-race TV t={t_stat:.2f} "
                           f"(1% bound 2.576)")
    # The ledger row adds "while the matched arm holds at \ffWinTVhrt", so the
    # surviving side is pinned too rather than left to the mismatched pair.
    assert_prose_close(fr["outlier_robustness"]["winsorized"]["horse_race"]["TV"]["t_stat"],
                       parse_num(m["ffWinTVhrt"]), tol=0.05,
                       claim=f"the foreign matched arm holds at \\ffWinTVhrt "
                             f"({m['ffWinTVhrt']}) under winsorisation")


def check_prose_threshold_sensitivity(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:appendix_j_marketcap.

    The appendix makes a two-sided claim. "Both horse-race statistics clear
    the 1\\% level, the mismatched filter's with its negative sign, up to the
    floor that discards \\threshHeldRemovedPct\\% of stock-days", and "at the
    highest floor, which leaves \\threshTopTickers{} stocks, neither statistic
    reaches the 5\\% level". Both sides are asserted here, along with the
    grid's coverage, because a grid that removed almost nothing would satisfy
    the first claim vacuously.

    The subsection prints no table, so every value the sweep produced reaches
    the reader through a range or an every-threshold quantifier. Each of those
    ranges over the grid derived from the results file, never over the points
    the sentence names."""
    df = pd.read_csv(RESULTS / "threshold_sensitivity_results.csv"
                     ).sort_values("threshold").reset_index(drop=True)
    removed = df["pct_stock_days_removed"].astype(float)

    baseline = df[df["threshold"] == 0].iloc[0]
    assert_prose_close(float(baseline["fm_hr_tstat_S_MC_t+1"]),
                       parse_num(m["rpHRMCtone"]), tol=0.5,
                       claim="threshold=0 t(S^MC) reproduces headline \\rpHRMCtone")
    assert_prose_close(float(baseline["fm_hr_tstat_S_TV_t+1"]),
                       parse_num(m["rpHRTVtone"]), tol=0.5,
                       claim="threshold=0 t(S^TV) reproduces headline \\rpHRTVtone")

    # The grid must actually vary the universe, or the survival claim is vacuous.
    assert_true(float(removed.max()) > 50.0,
                claim="the floors 'discard between nothing and \\threshTopRemovedPct\\% "
                      "of stock-days', past the panel's median market cap, so the "
                      "survival of both horse-race statistics is not vacuous",
                detail=f"top column removes {float(removed.max()):.1f}% of stock-days")

    # Side one. Both arms survive discarding the small tail, at the column the
    # macros quote (the largest such column, by the emitter's construction).
    held_pct = parse_num(m["threshHeldRemovedPct"])
    held = df.iloc[int((removed - held_pct).abs().idxmin())]
    assert_true(float(held["fm_hr_tstat_S_MC_t+1"]) > 2.576
                and float(held["fm_hr_tstat_S_TV_t+1"]) < -2.576,
                claim=f"both horse-race arms stay 1%-significant after discarding the "
                      f"smallest {held_pct:.0f}% of stock-days",
                detail=f"t(S^MC)={float(held['fm_hr_tstat_S_MC_t+1']):.2f}, "
                       f"t(S^TV)={float(held['fm_hr_tstat_S_TV_t+1']):.2f}")
    assert_close(m["threshHeldMCt"], float(held["fm_hr_tstat_S_MC_t+1"]),
                 tol=0.005, label="threshHeldMCt")

    # Side two, the boundary. In the top column neither arm is significant at the
    # 5% level, and the mismatched coefficient has crossed zero rather than
    # merely widened, which is the distinction the prose draws.
    top = df.iloc[-1]
    assert_true(abs(float(top["fm_hr_tstat_S_MC_t+1"])) < 1.96
                and abs(float(top["fm_hr_tstat_S_TV_t+1"])) < 1.96,
                claim="at the highest floor neither horse-race arm is "
                      "significant at the 5% level",
                detail=f"t(S^MC)={float(top['fm_hr_tstat_S_MC_t+1']):.2f}, "
                       f"t(S^TV)={float(top['fm_hr_tstat_S_TV_t+1']):.2f}")
    assert_true(float(top["fm_hr_coef_S_TV_t+1"]) >= 0.0,
                claim="the appendix says that at the highest floor 'the mismatched "
                      "coefficient crosses zero'",
                detail=f"top-column coef(S^TV)={float(top['fm_hr_coef_S_TV_t+1']):.3e}")
    coef_ratio = float(top["fm_hr_coef_S_MC_t+1"]) / float(baseline["fm_hr_coef_S_MC_t+1"])
    se_ratio = float(top["fm_hr_se_S_MC_t+1"]) / float(baseline["fm_hr_se_S_MC_t+1"])
    assert_true(coef_ratio < 1.0 and se_ratio > 1.0,
                claim="the appendix attributes the attenuation to the coefficient falling "
                      "as well as the standard error growing, not to precision alone",
                detail=f"coef ratio={coef_ratio:.2f}, se ratio={se_ratio:.2f}")
    assert_close(m["threshTopCoefRatio"], coef_ratio, tol=0.005, label="threshTopCoefRatio")
    assert_close(m["threshTopSeRatio"], se_ratio, tol=0.005, label="threshTopSeRatio")

    # Grid coverage, which the prose states in place of a table row.
    assert_close(m["threshTopTickers"], float(top["n_tickers"]),
                 tol=0.5, label="threshTopTickers")

    # The two arms leave the 1% level at different floors, which the prose
    # states in place of a table panel. Both quantifiers below range over the grid
    # derived from the file, never over the two points the prose names.
    t_mc = df["fm_hr_tstat_S_MC_t+1"].astype(float)
    t_tv = df["fm_hr_tstat_S_TV_t+1"].astype(float)
    assert_true(bool((t_mc.diff().dropna() < 0).all())
                and bool((t_tv.abs().iloc[1:].diff().dropna() < 0).all()),
                claim="'The matched horse-race statistic attenuates monotonically "
                      "across the grid', and the mismatched one does from the second "
                      "floor on",
                detail=f"t(S^MC)={[round(v, 2) for v in t_mc]}, "
                       f"|t(S^TV)|={[round(abs(v), 2) for v in t_tv]}")
    mc_last = df[t_mc > CRIT_1PCT].iloc[-1]
    assert_close(m["threshMCLastPct"], float(mc_last["pct_stock_days_removed"]),
                 tol=0.5, label="threshMCLastPct")
    assert_close(m["threshMCLastT"], float(mc_last["fm_hr_tstat_S_MC_t+1"]),
                 tol=0.005, label="threshMCLastT")

    # Panel B, as the two ranges and the two every-threshold quantifiers that
    # replaced it. The spread is the counterweight the prose spends: stable in
    # bps across the whole grid.
    bps_mc = df["ls_mean_bps_S_MC"].astype(float)
    bps_tv = df["ls_mean_bps_S_TV"].astype(float)
    for tag, series in (("MC", bps_mc), ("TV", bps_tv)):
        assert_close(m[f"threshLsBps{tag}Min"], float(series.min()), tol=0.05,
                     label=f"threshLsBps{tag}Min")
        assert_close(m[f"threshLsBps{tag}Max"], float(series.max()), tol=0.05,
                     label=f"threshLsBps{tag}Max")
    assert_true(float(bps_mc.max() - bps_mc.min()) < 0.25 * float(bps_mc.iloc[0]),
                claim="'the long-short spread does not fall as the floor rises', in basis "
                      "points across the grid",
                detail=f"bps range = [{float(bps_mc.min()):.1f}, {float(bps_mc.max()):.1f}]")
    assert_true(bool((df["ls_tstat_S_MC"].abs() > CRIT_1PCT).all())
                and bool((df["ls_tstat_S_TV"].abs() > CRIT_1PCT).all()),
                claim="'Both spreads are significant at the 1\\% level at every floor'",
                detail=f"smallest |t| = {min(df['ls_tstat_S_MC'].abs().min(), df['ls_tstat_S_TV'].abs().min()):.2f}")
    assert_true(bool((bps_mc > bps_tv).all()),
                claim="the matched filter's spread exceeds the mismatched one's at every "
                      "floor",
                detail=f"smallest lead = {float((bps_mc - bps_tv).min()):.2f} bps")

    # The matched filter's rank IC is a null on every grid point, which the
    # prose states in place of a table row. The ordering
    # between the filters is what survives the statistic, so it is asserted over
    # the grid rather than at the full-panel point the prose quotes.
    ic_t_mc = df["ic_tstat_S_MC"].astype(float).abs()
    assert_true(bool((ic_t_mc < CRIT_5PCT).all()),
                claim="the matched filter's rank information coefficient 'is "
                      "indistinguishable from zero at every floor'",
                detail=f"largest |t| = {float(ic_t_mc.max()):.2f}")
    assert_close(m["threshIcMCmaxT"], float(ic_t_mc.max()), tol=0.005,
                 label="threshIcMCmaxT")
    assert_true(bool((df["ic_mean_S_MC"] > df["ic_mean_S_TV"]).all()),
                claim="the matched filter's mean rank information coefficient exceeds "
                      "the mismatched one's at every floor",
                detail=f"smallest margin = "
                       f"{float((df['ic_mean_S_MC'] - df['ic_mean_S_TV']).min()):.5f}")

    # The baseline row, whose verdict subsec:appendix_j_marketcap states: the
    # same rows carry a rank statistic that does not clear the 5% bar and two
    # statistics that clear the 1% one, which is what "by the size of next-day
    # returns rather than by their within-day ordering" asserts. Bound to the
    # row rather than to the grid, because the sentence is about the full panel
    # and the grid maximum coinciding with it today is not the claim.
    row = df.sort_values("threshold").iloc[0]
    assert_true(abs(float(row["ic_tstat_S_MC"])) < CRIT_5PCT
                and float(row["fm_hr_tstat_S_MC_t+1"]) > CRIT_1PCT
                and float(row["ls_tstat_S_MC"]) > CRIT_1PCT,
                claim="subsec:appendix_j_marketcap says the matched filter's 'predictive "
                      "power therefore comes from the size of next-day returns rather "
                      "than from their within-day ordering': on the panel row the rank statistic "
                      "misses the 5% level while the Fama-MacBeth coefficient and the "
                      "long-short spread clear the 1% level",
                detail=f"panel row: IC t = {float(row['ic_tstat_S_MC']):+.2f}, "
                       f"FM t = {float(row['fm_hr_tstat_S_MC_t+1']):+.2f}, "
                       f"LS t = {float(row['ls_tstat_S_MC']):+.2f}")
    assert_true(abs(float(baseline["fm_hr_tstat_S_MC_t+1"])) > CRIT_1PCT
                and abs(float(baseline["ls_tstat_S_MC"])) > CRIT_1PCT,
                claim="the matched filter's full-panel Fama-MacBeth coefficient and "
                      "long-short spread are both significant at the 1% level, which "
                      "is what makes the null rank IC a statement about return magnitude",
                detail=f"t(FM)={float(baseline['fm_hr_tstat_S_MC_t+1']):.2f}, "
                       f"t(L/S)={float(baseline['ls_tstat_S_MC']):.2f}")


# The claim each criterion of code/empirical/robustness/window_2025_criteria.py restates, so a verdict
# that flips names the sentence to rewrite.
WINDOW_CRITERION_CLAIMS = {
    "K1": "subsec:return_prediction: in the horse race MC strengthens and TV "
          "turns negative for capacity-bound flow",
    "K2": "subsec:return_prediction: the foreign matched filter S^TV dominates "
          "the horse race and the mismatched S^MC turns negative",
    "K3": "subsec:reversal_test and the abstract: the stacked reversal is "
          "positive at the 1% level",
    "K4": "subsec:regime_assignment: capacity-bound and foreign flow fall on "
          "opposite sides of their own sign-flip nulls",
    "K5": "subsec:reversal_test and subsec:contrast_halves: the advantage for foreign flow "
          "is statistically significant on its own on common support",
    "K6": "subsec:skip_day: the reversal is absent on the second day after the "
          "flow and the advantage for foreign flow is positive there",
    "K7": "subsec:appendix_j_oos: the capacity-bound horse-race MC coefficient "
          "is significantly positive in every year",
}


def check_window_2025_verdicts(m: dict[str, str]) -> None:
    """Covers: the criteria of code/empirical/robustness/window_2025_criteria.py and
    the qf_paper.tex claims each one restates (WINDOW_CRITERION_CLAIMS).

    The criteria were committed before the panel was extended to 2020--2025,
    so their verdicts are the record of what the extension could have
    overturned. Two things are checked. (i) The verdicts grade the window the
    manuscript reports, on the values the manuscript prints. (ii) Every
    criterion holds, which is what the sentences each one restates assert on
    the longer window. A regenerated panel that flips one fails here and names
    the sentence to rewrite, since a criterion whose verdict no longer matches
    the prose is a gate that stopped binding.
    """
    v = json.loads((RESULTS / "window_2025_verdicts.json").read_text())
    window = f"{START_DATE_MAIN[:4]}-{END_DATE_MAIN[:4]}"
    assert_true(v["window"] == window,
                claim="the committed criteria grade the window the manuscript reports",
                detail=f"verdicts cover {v['window']}, the manuscript {window}")
    by = {x["id"]: x for x in v["verdicts"]}
    assert_true(sorted(by) == sorted(WINDOW_CRITERION_CLAIMS),
                claim="every committed criterion has a verdict and a sentence it restates",
                detail=f"verdicts {sorted(by)}")
    # (i) The graded values are the printed ones.
    for key, field, macro in (("K1", "t_MC_HR", "rpHRMCtone"),
                              ("K3", "t", "revCommonT"),
                              ("K4", "capacity_excess", "dirExcessCap"),
                              ("K4", "foreign_excess", "dirExcessFor"),
                              ("K5", "t", "revCommonForEdgeT"),
                              ("K6", "reversal_t", "revSkipCommonT"),
                              ("K6", "foreign_half_t", "revSkipCommonForEdgeT")):
        assert_prose_close(by[key]["values"][field], parse_num(m[macro]), tol=0.01,
                           claim=f"criterion {key} grades \\{macro} ({m[macro]}), "
                                 f"the value the manuscript prints")
    assert_prose_close(min(by["K7"]["values"].values()), parse_num(m["ryMinHRMCt"]),
                       tol=0.01, claim="criterion K7 grades the year \\ryMinHRMCt reports")
    # (ii) Every verdict matches the sentence that restates it.
    for key, claim in WINDOW_CRITERION_CLAIMS.items():
        assert_true(by[key]["passed"],
                    claim=f"{claim} (criterion {key}, window_2025_criteria.py)",
                    detail=f"{by[key]['claim']}: {by[key]['values']}")
