"""Checks on the return prediction by investor class and on the reversal
between the two classes, measured by the scripts of
code/empirical/return_prediction/."""

from __future__ import annotations

import json
import re

import pandas as pd

from shared_utils import CRIT_1PCT, CRIT_5PCT, CRIT_10PCT

from output._formatters import BPS_PER_RETURN
from output.assertions.common import (
    PAPER_TEX,
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def check_return_prediction(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:return_prediction."""
    rp = pd.read_csv(RESULTS / "return_prediction_summary.csv")
    hr1 = rp[(rp.Horizon == "R_{t+1}") & (rp.Specification == "Horse Race")].iloc[0]
    assert_close(m["rpHRMCtone"], hr1.t_MC, tol=0.02, label="rpHRMCtone")
    assert_close(m["rpHRTVtone"], hr1.t_TV, tol=0.02, label="rpHRTVtone")


def check_prose_capacity_return_prediction(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:return_prediction (next-day directionals)."""
    rp = json.loads((RESULTS / "return_prediction_results.json").read_text())
    h1 = rp["R_t1"]
    assert_true(h1["mc_only"]["p_value_S_MC"] < 0.001,
                claim="MC normalisation ... p < 0.001",
                detail=f"p_value_S_MC={h1['mc_only']['p_value_S_MC']:.2e}")
    assert_true(h1["horse_race"]["t_stat_S_MC"] > h1["mc_only"]["t_stat_S_MC"],
                claim="In the horse race, MC strengthens",
                detail=f"t_MC_HR={h1['horse_race']['t_stat_S_MC']:.2f}, "
                       f"t_MC_uni={h1['mc_only']['t_stat_S_MC']:.2f}")
    assert_true(h1["horse_race"]["t_stat_S_TV"] < 0 < h1["tv_only"]["t_stat_S_TV"],
                claim="TV reverses sign in the horse race",
                detail=f"t_TV_uni={h1['tv_only']['t_stat_S_TV']:.2f}, "
                       f"t_TV_HR={h1['horse_race']['t_stat_S_TV']:.2f}")


def check_prose_foreign_return_prediction(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:return_prediction (foreign).

    Foreign horse-race S^MC sign flip — not behind a macro."""
    mi = json.loads((RESULTS / "multi_investor_robustness_results.json").read_text())
    f = mi["return_predictions"]["foreign"]
    # "In the horse race the mismatched filter turns negative, S^TV for
    # capacity-bound flows at the 1% level and S^MC for foreign flows at the 10%
    # level" (subsec:return_prediction). The foreign level is pinned from both
    # sides, because the sentence names the level it clears and, by naming it,
    # the one it does not.
    cap = mi["return_predictions"]["capacity_bound"]
    assert_true(cap["horse_race_TV"]["t_stat"] < -CRIT_1PCT,
                claim="the capacity-bound mismatched filter turns negative at the 1% "
                      "level in the horse race (subsec:return_prediction)",
                detail=f"horse_race_TV_t={cap['horse_race_TV']['t_stat']:.2f}")
    assert_true(-CRIT_5PCT < f["horse_race_MC"]["t_stat"] < -CRIT_10PCT,
                claim="the foreign mismatched filter turns negative at the 10% level "
                      "and not the 5% one in the horse race (subsec:return_prediction)",
                detail=f"horse_race_MC_t={f['horse_race_MC']['t_stat']:.2f}")
    # "The mismatched filter S^MC also predicts them, as the capacity-bound S^TV
    # does, so in both classes the matched-filter principle ranks two informative
    # filters" (subsec:return_prediction).
    assert_true(f["univariate_MC"]["t_stat"] > CRIT_5PCT
                and cap["univariate_TV"]["t_stat"] > CRIT_5PCT,
                claim="each class's mismatched filter predicts next-day returns on its "
                      "own at the 5% level (subsec:return_prediction)",
                detail=f"foreign MC t={f['univariate_MC']['t_stat']:.2f}, "
                       f"capacity-bound TV t={cap['univariate_TV']['t_stat']:.2f}")


def check_prose_aggregate_alignment(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:appendix_aggregate_alignment; qf_paper.tex subsec:boundary_extent.

    KRX aggregate is S^MC-dominant at h=1 — expected values read from
    \\ihAggregate* macros, with directional corroboration of the matched filter."""
    mi = json.loads((RESULTS / "multi_investor_robustness_results.json").read_text())
    agg = mi["return_predictions"]["institutional_aggregate"]
    assert_prose_close(agg["univariate_MC"]["t_stat"], parse_num(m["ihAggregateMCt"]), tol=0.05,
                       claim=f"KRX aggregate univariate S^MC t = \\ihAggregateMCt ({m['ihAggregateMCt']})")
    assert_prose_close(agg["horse_race_MC"]["t_stat"], parse_num(m["ihAggregateHRMCt"]), tol=0.05,
                       claim=f"KRX aggregate t_MC^HR = \\ihAggregateHRMCt ({m['ihAggregateHRMCt']})")
    assert_prose_close(agg["horse_race_TV"]["t_stat"], parse_num(m["ihAggregateHRTVt"]), tol=0.05,
                       claim=f"KRX aggregate t_TV^HR = \\ihAggregateHRTVt ({m['ihAggregateHRTVt']}) in the horse race")
    # subsec:boundary_extent and subsec:appendix_aggregate_alignment: "S^MC
    # predicts better than S^TV" for the aggregate, in the primary univariate
    # specification as well as in the horse race.
    assert_true(agg["univariate_MC"]["mean"] > agg["univariate_TV"]["mean"],
                claim="for the KRX domestic institutional aggregate the univariate "
                      "slope on S^MC exceeds the one on S^TV",
                detail=f"univ MC {agg['univariate_MC']['mean']:+.6f}, "
                       f"univ TV {agg['univariate_TV']['mean']:+.6f}")
    assert_true(agg["univariate_MC"]["t_stat"] > 2.576
                and agg["horse_race_MC"]["t_stat"] > 2.576
                and agg["horse_race_TV"]["t_stat"] < 0,
                claim="KRX domestic institutional aggregate is itself S^MC-dominant at the "
                      "next-day horizon (corroborates the matched filter, not a null)",
                detail=f"univ MC t={agg['univariate_MC']['t_stat']:.2f}, "
                       f"HR MC t={agg['horse_race_MC']['t_stat']:.2f}, "
                       f"HR TV t={agg['horse_race_TV']['t_stat']:.2f}")


def check_prose_reversal_interaction(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:reversal_test, tab:reversal_interaction, abstract.

    The reversal is a class-by-normalisation interaction that the contrast,
    the difference-in-differences b4 - b3, measures, so the first claim is that
    the contrast is positive and clears the noise. Checked on both stackings and under both
    specifications — the univariate one prop:matched_filter ranks and the joint
    one that conditions each filter on the other — and cross-checked against the
    separately estimated Panel A regressions, which the stacked specification
    nests exactly."""
    ri = json.loads((RESULTS / "reversal_interaction_results.json").read_text())
    arms = (("univariate", "fama_macbeth_univariate"), ("joint", "fama_macbeth"))
    for sample in ("union", "common"):
        for arm, key in arms:
            fm = ri[sample][key]
            assert_true(fm["reversal"]["mean"] > 0 and fm["reversal"]["t"] > CRIT_1PCT,
                        claim=f"the contrast is a positive, significant "
                              f"class-by-normalisation interaction ({sample} sample, "
                              f"{arm} specification)",
                        detail=f"b4-b3={fm['reversal']['mean']:.6f}, t={fm['reversal']['t']:.2f}")
            cap, foreign = fm["advantage_capacity"], fm["advantage_foreign"]
            assert_true(cap["t"] > CRIT_5PCT and foreign["t"] < -CRIT_5PCT,
                        claim=f"the matched-filter advantage reverses sign across the two "
                              f"classes rather than merely weakening, each side significant "
                              f"on its own ({sample} sample, {arm} specification)",
                        detail=f"capacity MC-TV edge={cap['mean']:+.6f} (t={cap['t']:+.2f}), "
                               f"foreign={foreign['mean']:+.6f} (t={foreign['t']:+.2f})")
        pooled = ri[sample]["pooled_cluster_date"]["reversal"]
        assert_true(pooled["value"] > 0 and pooled["t_stat"] > CRIT_1PCT,
                    claim=f"the reversal survives cluster-by-date inference ({sample} sample)",
                    detail=f"b4-b3={pooled['value']:.6f}, t={pooled['t_stat']:.2f}, "
                           f"p={pooled['p_value']:.3g}")

    # The base class of the stacked specification is estimated on the same rows
    # as tab:investor_heterogeneity Panel A, so its loadings must reproduce the
    # panel's under both specifications ("That union stacking nests the two
    # panels ... exactly").
    panel_a = json.loads((RESULTS / "return_prediction_results.json").read_text())["R_t1"]
    for panel_t, key, coef in ((panel_a["mc_only"]["t_stat_S_MC"], "fama_macbeth_univariate", "zMC"),
                               (panel_a["tv_only"]["t_stat_S_TV"], "fama_macbeth_univariate", "zTV"),
                               (panel_a["horse_race"]["t_stat_S_MC"], "fama_macbeth", "zMC"),
                               (panel_a["horse_race"]["t_stat_S_TV"], "fama_macbeth", "zTV")):
        assert_prose_close(ri["union"][key]["coefficients"][coef]["t"], panel_t, tol=0.05,
                           claim=f"the union stacking nests the Panel A regressions "
                                 f"exactly ({key}, {coef}: Panel A t = {panel_t:.2f})")

    # Common support is the primary identification sample of subsec:reversal_test,
    # so its stacking must be balanced for "one set of firms, days, prices, and
    # realised returns" to hold literally.
    per_class = ri["common"]["n_rows_by_class"]
    assert_true(len(set(per_class.values())) == 1,
                claim="the two classes are compared on one set of firms, days, prices, "
                      "and realised returns rather than on two overlapping sets",
                detail=f"rows by class: {per_class}")
    assert_prose_close(float(next(iter(per_class.values()))),
                       parse_num(m["revCommonStockDays"]), tol=0.5,
                       claim=f"the \\revCommonStockDays ({m['revCommonStockDays']}) "
                             f"stock-days on which both classes trade and the stock has a "
                             f"next-day return")
    assert_prose_close(ri["common"]["n_dates"], parse_num(m["sampTotalDaysMinusOne"]), tol=0.5,
                       claim=f"the stacked test spans every forward-return day of the "
                             f"sample, \\sampTotalDaysMinusOne ({m['sampTotalDaysMinusOne']})")

    for stacking in ("Common", "Union"):
        res = ri[stacking.lower()]
        uni = res["fama_macbeth_univariate"]
        pinned = [
            (f"rev{stacking}Rev", uni["reversal"]["mean"] * BPS_PER_RETURN),
            (f"rev{stacking}CapEdge",
             uni["advantage_capacity"]["mean"] * BPS_PER_RETURN),
            (f"rev{stacking}ForEdge",
             uni["advantage_foreign"]["mean"] * BPS_PER_RETURN),
        ]
        # The union contrast's and advantages' t are read off
        # tab:reversal_interaction, so only the common-support ones carry a macro.
        if stacking == "Common":
            pinned += [(f"rev{stacking}T", uni["reversal"]["t"]),
                       (f"rev{stacking}CapEdgeT", uni["advantage_capacity"]["t"]),
                       (f"rev{stacking}ForEdgeT", uni["advantage_foreign"]["t"])]
        for macro, value in pinned:
            assert_prose_close(value, parse_num(m[macro]), tol=0.01,
                               claim=f"\\{macro} ({m[macro]})")

    # Relative tolerance: an absolute one is vacuous against a p-value of 1e-14.
    p_macro = parse_num(m["revCommonClusterP"])
    assert_prose_close(ri["common"]["pooled_cluster_date"]["reversal"]["p_value"],
                       p_macro, tol=0.05 * p_macro,
                       claim=f"p = \\revCommonClusterP ({m['revCommonClusterP']}) in the "
                             f"pooled fit with standard errors clustered by date")

    assert_prose_close(ri["union"]["n_rows"], parse_num(m["revUnionRows"]), tol=0.5,
                       claim=f"raises the sample to \\revUnionRows ({m['revUnionRows']}) rows")
    assert_prose_close(ri["union"]["pooled_cluster_date"]["reversal"]["t_stat"],
                       parse_num(m["revUnionClusterT"]), tol=0.01,
                       claim=f"t = ${m['revUnionClusterT']}$ for the union horse-race "
                             f"contrast in the pooled fit")
    union_t = ri["union"]["fama_macbeth_univariate"]["reversal"]["t"]
    assert_true(parse_num(m["revCommonT"]) < union_t,
                claim="the common-support estimate is the smaller of the two",
                detail=f"common t={m['revCommonT']}, union t={union_t:.2f}")

    # "In the horse race, the contrast roughly doubles without a matching rise
    # in its $t$-statistic ..., so the horse race gives no stronger evidence for
    # the reversal than the univariate specification" — the sentence
    # subsec:reversal_test uses to demote the horse race. Both arms are read off the results the table prints.
    for stacking in ("Common", "Union"):
        res = ri[stacking.lower()]
        uni = res["fama_macbeth_univariate"]["reversal"]["mean"] * BPS_PER_RETURN
        joint = res["fama_macbeth"]["reversal"]["mean"] * BPS_PER_RETURN
        assert_true(joint > 1.8 * uni,
                    claim=f"the joint specification roughly doubles the contrast "
                          f"({stacking.lower()} sample)",
                    detail=f"univariate={uni:.2f} bp, joint={joint:.2f} bp, "
                           f"ratio={joint / uni:.2f}")
        uni_t = res["fama_macbeth_univariate"]["reversal"]["t"]
        joint_t = res["fama_macbeth"]["reversal"]["t"]
        assert_true(abs(joint_t - uni_t) < 0.1 * uni_t,
                    claim=f"the joint specification does not sharpen the evidence "
                          f"({stacking.lower()} sample)",
                    detail=f"univariate t={uni_t:.2f}, joint t={joint_t:.2f}")


def check_prose_pre_period_reversal(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:contrast_halves, subsec:limitations, tab:contrast_halves, abstract;
    qf_paper.tex subsec:appendix_j_oos.

    The contrast is re-estimated on the window preceding the published one, and
    the two classes' advantages come back differently. Four things are checked. (i) The
    pre-period arms really do run on a window disjoint from and earlier than the
    published one, which is the whole of what makes them out of sample. (ii) The
    contrast is positive and clears the conventional level on both stackings,
    the claim the abstract's "remain significant on a \\preYears{} pre-period"
    rests on.
    (iii) The advantage for capacity-bound flow is significant there. (iv) The advantage for foreign flow keeps the
    sign cor:regimes predicts and reaches the conventional level on neither
    stacking, which is asserted in both directions because the prose says both.
    A future panel that made the advantage for foreign flow significant there would leave
    subsec:contrast_halves reporting a null the data no longer show."""
    ri = json.loads((RESULTS / "reversal_interaction_results.json").read_text())
    published, pre = ri["common"], {"common": ri["pre_common"], "union": ri["pre_union"]}

    # (i) Out of sample means disjoint and earlier, so both are checked here
    # rather than inferred from the panel the arm was read out of.
    for stacking, res in pre.items():
        assert_true(res["window"] == "pre_period" and res["date_last"] < published["date_first"],
                    claim=f"the pre-period contrast runs on a window \"disjoint from "
                          f"it and preceding it\" ({stacking} stacking)",
                    detail=f"pre-period {res['date_first']}..{res['date_last']}, "
                           f"published from {published['date_first']}")

    for stacking, res in pre.items():
        uni = res["fama_macbeth_univariate"]
        rev, cap, foreign = (uni["reversal"], uni["advantage_capacity"],
                             uni["advantage_foreign"])

        # (ii) and (iii): the abstract's "The contrast and the advantage for
        # pension and insurance flow ... remain significant on a \preYears{}
        # pre-period", and subsec:contrast_halves' "The contrast and the
        # advantage for capacity-bound flow are significant".
        assert_true(rev["mean"] > 0 and rev["t"] > CRIT_1PCT,
                    claim=f"the contrast and the advantage for pension and insurance flow "
                          f"\"remain significant on a \\preYears{{}} pre-period\" "
                          f"({stacking} stacking)",
                    detail=f"b4-b3={rev['mean']:.6f}, t={rev['t']:.2f}")
        assert_true(cap["t"] > CRIT_1PCT and cap["mean"] > abs(foreign["mean"]),
                    claim=f"\"the contrast and the advantage for capacity-bound flow are significant\" on the pre-period "
                          f"({stacking} stacking)",
                    detail=f"capacity={cap['mean']:+.6f} (t={cap['t']:+.2f}), "
                           f"foreign={foreign['mean']:+.6f} (t={foreign['t']:+.2f})")

        # (iv) Both directions, because the prose claims both.
        assert_true(foreign["mean"] < 0,
                    claim=f"the advantage for foreign flow \"keeps the sign Corollary cor:regimes "
                          f"predicts\" on the pre-period ({stacking} stacking)",
                    detail=f"foreign advantage={foreign['mean']:+.6f}")
        assert_true(abs(foreign["t"]) < CRIT_5PCT,
                    claim=f"the advantage for foreign flow \"reaches the conventional level on neither "
                          f"stacking\" on the pre-period ({stacking} stacking)",
                    detail=f"foreign t={foreign['t']:+.2f}, conventional level "
                           f"{CRIT_5PCT:.2f}")

    assert_prose_close(float(set(ri["pre_common"]["n_rows_by_class"].values()).pop()),
                       parse_num(m["revPreCommonStockDays"]), tol=0.5,
                       claim=f"\\revPreCommonStockDays ({m['revPreCommonStockDays']}) "
                             f"stock-days of common support")
    for stacking, key in (("PreCommon", "pre_common"), ("PreUnion", "pre_union")):
        uni = ri[key]["fama_macbeth_univariate"]
        pinned = [
            (f"rev{stacking}Rev", uni["reversal"]["mean"] * BPS_PER_RETURN),
            (f"rev{stacking}T", uni["reversal"]["t"]),
            (f"rev{stacking}ForEdge", uni["advantage_foreign"]["mean"] * BPS_PER_RETURN),
            (f"rev{stacking}ForEdgeT", uni["advantage_foreign"]["t"]),
        ]
        # The claim for foreign flow is that its advantage reaches the level on neither
        # stacking, so both are quoted. The advantage for capacity-bound flow is established on
        # common support, which is the stacking the section reports it on.
        if stacking == "PreCommon":
            pinned += [
                (f"rev{stacking}CapEdge",
                 uni["advantage_capacity"]["mean"] * BPS_PER_RETURN),
                (f"rev{stacking}CapEdgeT", uni["advantage_capacity"]["t"]),
            ]
        for macro, value in pinned:
            assert_prose_close(value, parse_num(m[macro]), tol=0.01,
                               claim=f"\\{macro} ({m[macro]})")


def check_prose_zero_flow(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:appendix_zero_flow; qf_paper.tex
    subsec:contrast_halves, tab:contrast_halves, subsec:data.

    Every specification in the paper drops the stock-days on which the tested
    class did not trade, and that filter removes most of one class's rows and
    almost none of the other's. The section's claim is that the reversal is not
    an artefact of a restriction shaped like that, so what has to be pinned is
    the survival of both signs once the restriction is lifted, not the size of
    the zero-inclusive estimate. Four things are checked. (i) The filter is
    one-sided, which is what gives the objection its force and what a future
    panel could quietly remove. (ii) The zero-included stacking is balanced
    across the two classes, which is why the section can say the union and
    common stackings coincide on it. (iii) The reversal and both per-class
    advantages keep their signs and significance without the filter. (iv) The
    advantage that moves is the one belonging to the class the filter removes
    most of, the sentence the section reads the attenuation through."""
    ri = json.loads((RESULTS / "reversal_interaction_results.json").read_text())
    union, zero = ri["union"], ri["zero_included"]
    uni_u, uni_z = union["fama_macbeth_univariate"], zero["fama_macbeth_univariate"]

    # (i) The asymmetry the objection rests on ("of which the restriction
    # removes far more for the capacity-bound class than for the foreign
    # class", subsec:contrast_halves).
    excluded = {}
    for cls in ("capacity_bound", "foreign"):
        excluded[cls] = 100.0 * (1.0 - union["n_rows_by_class"][cls]
                                 / zero["n_rows_by_class"][cls])
    assert_true(excluded["capacity_bound"] > 10 * excluded["foreign"],
                claim="the nonzero-flow restriction falls unequally on the two classes",
                detail=f"capacity {excluded['capacity_bound']:.1f}% vs "
                       f"foreign {excluded['foreign']:.1f}%")

    # (ii) Every row carries both classes once the filter is lifted, which is
    # what makes the union and common stackings the same sample here.
    assert_true(len(set(zero["n_rows_by_class"].values())) == 1,
                claim="both stackings of subsec:reversal_test coincide on the "
                      "zero-included sample, because every row carries both classes",
                detail=f"{zero['n_rows_by_class']}")
    assert_prose_close(zero["n_rows"], parse_num(m["revZeroRows"]), tol=0.5,
                       claim=f"the reversal runs on \\revZeroRows "
                             f"({m['revZeroRows']}) stacked rows")

    # (iii) The claim of the section. Both classes' advantages keep their sign without the
    # filter, so the restriction is not what produces the reversal.
    assert_true(uni_z["reversal"]["t"] > CRIT_1PCT,
                claim="the nonzero-flow restriction is not what produces the reversal",
                detail=f"reversal={uni_z['reversal']['mean']:.6f}, "
                       f"t={uni_z['reversal']['t']:.2f}")
    assert_true(uni_z["advantage_capacity"]["t"] > CRIT_5PCT
                > -CRIT_5PCT > uni_z["advantage_foreign"]["t"],
                claim="each class's advantage keeps its sign with no participation filter",
                detail=f"capacity t={uni_z['advantage_capacity']['t']:+.2f}, "
                       f"foreign t={uni_z['advantage_foreign']['t']:+.2f}")

    for macro, t_macro, pack in (("revZeroRev", "revZeroT", uni_z["reversal"]),
                                 ("revZeroCapEdge", "revZeroCapEdgeT",
                                  uni_z["advantage_capacity"]),
                                 ("revZeroForEdge", "revZeroForEdgeT",
                                  uni_z["advantage_foreign"])):
        assert_prose_close(pack["mean"] * BPS_PER_RETURN, parse_num(m[macro]), tol=0.02,
                           claim=f"\\{macro} ({m[macro]}) basis points")
        assert_prose_close(pack["t"], parse_num(m[t_macro]), tol=0.02,
                           claim=f"$t = \\{t_macro}$ ({m[t_macro]}) beside \\{macro}")

    # (iv) "The class the filter removes most of is the one whose advantage
    # moves." Asserted as a comparison of the two shifts, because a future
    # panel could leave both macros correct and this sentence false.
    shift = {c: abs(uni_z[k]["mean"] - uni_u[k]["mean"])
             for c, k in (("capacity_bound", "advantage_capacity"),
                          ("foreign", "advantage_foreign"))}
    assert_true(shift["capacity_bound"] > shift["foreign"],
                claim="the class the nonzero-flow filter removes most of is the one "
                      "whose matched-filter advantage moves when it is lifted",
                detail=f"capacity shift {shift['capacity_bound'] * BPS_PER_RETURN:.2f} bps, "
                       f"foreign shift {shift['foreign'] * BPS_PER_RETURN:.2f} bps")


def check_prose_contrast_halves_rows(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:contrast_halves, subsec:size_restricted and
    tab:contrast_halves.

    "The contrast and the advantage for capacity-bound flow are significant at
    the 1% level under each of five alternative estimations" (subsec:contrast_halves)
    and "The advantage for foreign flow is not significant at the 5% level under
    three of the five alternative estimations" (subsec:size_restricted). The two
    sentences range over the alternative rows of tab:contrast_halves, every row
    after the primary one, a set the paper enumerates itself, so the set is read
    from the tabular: each row's three cells name a coefficient macro and its
    t-statistic macro, and a row added to the table enters the check with no
    edit here. The one row the check leaves out is `Second-day return', which
    re-estimates the primary specification on another return and is no
    alternative estimation (subsec:skip_day).
    """
    tex = PAPER_TEX.read_text()
    end = tex.index(r"\label{tab:contrast_halves}")
    start = tex.rindex(r"\begin{tabular*}", 0, end)
    body = tex[start:tex.index(r"\end{tabular*}", start)]
    # A data row prints a t-statistic macro in parentheses; the header rows,
    # \cmidrule(lr) included, print none.
    rows = [ln for ln in body.split(r"\\")
            if ln.count("&") == 3 and re.search(r"\(\$?\\\w+", ln)
            and "Second-day return" not in ln.split("&")[0]]
    assert_true(len(rows) >= 2,
                claim="tab:contrast_halves carries the rows the sentence ranges over",
                detail=f"{len(rows)} row(s) parsed")
    foreign_t = []
    for row in rows:
        label, cap, foreign, rev = (c.strip() for c in row.split("&"))
        t = [parse_num(m[name]) for name in
             (re.findall(r"\\(\w+)\$?\)", c)[-1] for c in (cap, foreign, rev))]
        cap_t, for_t, rev_t = t
        foreign_t.append(for_t)
        assert_true(cap_t > CRIT_1PCT and rev_t > CRIT_1PCT,
                    claim="subsec:contrast_halves says the contrast and the "
                          "advantage for capacity-bound flow are significant at the 1% level in "
                          "every row of tab:contrast_halves",
                    detail=f"'{label[:40]}': capacity t = {cap_t}, contrast t = {rev_t}")
    words = {"three": 3, "four": 4, "five": 5}
    each = re.search(r"under each of (\w+) alternative estimations", tex)
    said = re.search(r"advantage for foreign flow is not significant at the 5\\% level "
                     r"under (\w+) of the (\w+) alternative estimations", tex)
    assert each and said, ("subsec:contrast_halves or subsec:size_restricted no longer "
                           "counts the alternative estimations or the ones that leave the "
                           "advantage for foreign flow insignificant; re-anchor this check "
                           "or delete it with the sentence")
    alternatives = foreign_t[1:]
    weak = sum(abs(x) < CRIT_5PCT for x in alternatives)
    assert_true(words.get(each.group(1)) == len(alternatives)
                and words.get(said.group(2)) == len(alternatives)
                and words.get(said.group(1)) == weak,
                claim="subsec:size_restricted says the advantage for foreign flow is not "
                      f"significant at the 5% level under {said.group(1)} of the "
                      f"{said.group(2)} alternative estimations of tab:contrast_halves",
                detail=f"foreign t = {foreign_t}")


def check_reversal_word_needs_both_halves(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex eq:reversal_contrast, which names b4 - b3 the
    contrast, and every sentence that prints an estimate of it.

    A significant difference between two effects establishes neither effect's
    sign (claims-discipline), so a sentence may call an estimate the reversal
    only where its advantage for capacity-bound flow is significantly positive and its
    advantage for foreign flow significantly negative. The estimates whose advantages the paper
    prints are derived twice: from every tabular whose header names both
    advantages, and from the parameter file's advantage macros (<P>CapEdgeT and
    <P>ForEdgeT beside <P>Rev and <P>T). A sentence that prints an estimate
    failing that test must not name the reversal, and no sentence makes the
    reversal the subject of "significant", because significance belongs to the
    contrast. A sentence that asserts the reversal for a sample it names in
    words, with no macro, is not read here.
    """
    text = re.sub(r"(?<!\\)%.*", "", PAPER_TEX.read_text())
    halves = []
    for p in sorted(k[:-len("ForEdgeT")] for k in m if k.endswith("ForEdgeT")):
        if {f"{p}CapEdgeT", f"{p}Rev", f"{p}T"} <= m.keys():
            halves.append((f"{p}CapEdgeT", f"{p}ForEdgeT", (f"{p}Rev", f"{p}T")))
    for head in re.finditer(r"Capacity-bound & Foreign & Contrast \\\\", text):
        body = text[head.end():text.index(r"\end{tabular*}", head.end())]
        for row in body.split(r"\\"):
            cells = row.split("&")
            if len(cells) != 4 or "(" not in row:
                continue
            cap, foreign, contrast = (re.findall(r"\\(\w+)", c) for c in cells[1:])
            halves.append((cap[-1], foreign[-1], tuple(contrast)))
    assert_true(len(halves) >= 2,
                claim="the paper prints the advantages behind the contrasts it reports",
                detail=f"{len(halves)} estimate(s) with both advantages found")
    weak = {name for cap_t, for_t, names in halves
            if not (parse_num(m[cap_t]) > CRIT_5PCT and parse_num(m[for_t]) < -CRIT_5PCT)
            for name in names}
    plain = re.sub(r"\\(?:label|ref|eqref|input)\{[^}]*\}", "", text)
    wrong = []
    for unit in re.split(r"(?<=[.?!])\s+(?=[A-Z\\$])|\n\s*\n|\\\\", plain):
        if not re.search(r"\breversal\b", unit, re.I):
            continue
        prints_weak = any(re.search(rf"\\{n}(?![A-Za-z])", unit) for n in weak)
        called_significant = re.search(
            r"\breversal\b[^.;]{0,80}?\b(?:is|are|remains|stays)\s+(?:also\s+)?"
            r"(?:statistically\s+)?significant", unit, re.I)
        if prints_weak or called_significant:
            wrong.append(" ".join(unit.split())[:120])
    assert_true(not wrong,
                claim="eq:reversal_contrast names b4 - b3 the contrast, and a sentence "
                      "calls an estimate the reversal only where both classes' advantages are "
                      "significant with the predicted signs",
                detail=" | ".join(wrong))
