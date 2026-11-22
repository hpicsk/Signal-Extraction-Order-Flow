"""Checks on the return prediction inside subsets of the panel, by index
membership, liquidity and short-sale regime, measured by the scripts of
code/empirical/return_prediction/."""

from __future__ import annotations

import json

import pandas as pd

from shared_utils import CRIT_5PCT

from output.assertions.common import (
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def check_kospi200_falsification(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsubsec:kospi200_falsification; qf_paper.tex
    subsec:appendix_kospi200, both the return columns of
    tab:foreign_kospi200_split and the size control's regressions."""
    k200 = json.loads((RESULTS / "foreign_kospi200_split_results.json").read_text())
    for key, prefix in [("full", "All"), ("kospi200", "In"), ("non_kospi200", "Out")]:
        d = k200[key]
        # The three primary subsets are quoted by sample size alone, their
        # t-statistics, and the size control's, being read off
        # tab:foreign_kospi200_split.
        assert parse_num(m[f"retForeignKospi{prefix}Obs"]) == float(d["panel_size"]), (
            f"retForeignKospi{prefix}Obs={m[f'retForeignKospi{prefix}Obs']} vs data={d['panel_size']}")

    # "Membership also tracks firm size, but the out-of-index stocks of the
    # largest size quintile do not reproduce the negative coefficient inside
    # the index" (subsubsec:kospi200_falsification). Foreign S^TV is negative
    # inside the index in both the univariate regression and the horse race,
    # so both are held positive on the size control, with S^TV still ahead of
    # S^MC in the horse race.
    largest = k200["non_kospi200_largest"]
    largest_hr = largest["horse_race"]
    assert_true(largest["univariate_TV"]["t_stat"] > 0,
                claim=("the out-of-index stocks of the largest size quintile do not "
                       "reproduce the negative coefficient inside the index"),
                detail=f"univariate TV t={largest['univariate_TV']['t_stat']:.2f}")
    assert_true(largest_hr["signal_TV"]["t_stat"] > largest_hr["signal_MC"]["t_stat"] > 0,
                claim=("on the largest out-of-index names both filters load positive "
                       "with S^TV still ahead"),
                detail=(f"TV={largest_hr['signal_TV']['t_stat']:.2f}, "
                        f"MC={largest_hr['signal_MC']['t_stat']:.2f}"))


def check_prose_kospi200_horse_race(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsubsec:kospi200_falsification, and the size-boundary
    sentence in subsec:limitations that cites the same split as its second axis.

    The KOSPI 200 cells are read off tab:foreign_kospi200_split, so what is
    pinned here is each direction the paragraph states."""
    ks = json.loads((RESULTS / "foreign_kospi200_split_results.json").read_text())
    full = ks["full"]
    k_in = ks["kospi200"]
    k_out = ks["non_kospi200"]
    out_hr = k_out["horse_race"]["signal_TV"]["t_stat"]
    in_hr = k_in["horse_race"]["signal_TV"]["t_stat"]
    assert_true(out_hr > 0 > in_hr,
                claim="horse-race preserves contrast: positive outside, negative inside KOSPI 200",
                detail=f"out_hr={out_hr:.2f}, in_hr={in_hr:.2f}")
    assert_true(k_out["univariate_TV"]["t_stat"] > full["univariate_TV"]["t_stat"],
                claim="subsubsec:kospi200_falsification: foreign S^TV has a larger univariate "
                      "t-statistic outside the index than in the full foreign-flow panel",
                detail=f"out={k_out['univariate_TV']['t_stat']:.2f}, "
                       f"full={full['univariate_TV']['t_stat']:.2f}")
    # subsec:limitations reads this split as the second axis of the foreign size
    # boundary, so the sign inside the index carries the claim there and not only
    # the magnitude tab:foreign_kospi200_split publishes.
    assert_true(k_in["univariate_TV"]["t_stat"] < 0,
                claim="subsubsec:kospi200_falsification: foreign S^TV has a negative "
                      "univariate t-statistic inside the index",
                detail=f"in={k_in['univariate_TV']['t_stat']:.2f}")
    assert_true(k_in["univariate_TV"]["t_stat"] < 0,
                claim="Foreign S^TV turns negative inside the KOSPI 200",
                detail=f"inside_t={k_in['univariate_TV']['t_stat']:.2f}")


def check_prose_liquidity_stratified(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:appendix_liquidity_stratified, plus the
    sentences that cite it, in qf_paper.tex subsec:size_restricted,
    subsec:limitations and subsec:appendix_size_restricted.

    Size axis, in subsec:appendix_size_stratified. "Capacity-bound S^MC has a
    larger t-statistic than S^TV in every quintile. In the two smallest it is
    indistinguishable from zero". "The t-statistic of foreign S^TV declines
    monotonically from Q1, the smallest stocks, to Q5, the largest. In the
    largest quintile it no longer exceeds that of S^MC. Neither is significant
    at the 5% level there".

    Where the headline sample sits on that axis. In
    subsec:appendix_size_restricted, "Common support holds
    \\liqCommonSizeQFivePct\\% of its stock-days in the largest
    market-capitalization quintile, against \\liqForSizeQFivePct\\% of the
    foreign class's own nonzero-flow days". The sentence of subsec:contrast_halves saying the
    rank transform and the net-participation matching each cost the advantage for foreign flow
    its sign on common support and leave it standing on the union stacking is
    pinned in check_prose_participation_matching against both results files."""
    d = json.loads((RESULTS / "liquidity_stratified_results.json").read_text())
    qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]

    def t(axis, investor, q, arm):
        cell = d[f"by_{axis}"][investor][q][f"univariate_{arm}"]
        return cell[f"S_{arm}_{investor}"]["t_stat"]

    # --- Size axis -------------------------------------------------------
    cap_matched = [t("size", "capacity_bound", q, "MC") for q in qs]
    cap_mis = [t("size", "capacity_bound", q, "TV") for q in qs]
    assert_true(all(mt > xt for mt, xt in zip(cap_matched, cap_mis)),
                claim="capacity-bound matched filter leads its mismatched alternative "
                      "in every market-capitalisation quintile",
                detail=f"matched={[f'{x:+.2f}' for x in cap_matched]}, "
                       f"mismatched={[f'{x:+.2f}' for x in cap_mis]}")
    assert_true(abs(cap_matched[0]) < 1.96 and abs(cap_matched[1]) < 1.96,
                claim="the two smallest quintiles carry too little capacity-bound flow "
                      "to separate either filter from zero",
                detail=f"Q1 t={cap_matched[0]:+.2f}, Q2 t={cap_matched[1]:+.2f}")
    assert_prose_close(cap_matched[0], parse_num(m["liqCapSizeQOneT"]), tol=0.02,
                       claim=f"capacity-bound size Q1 t = \\liqCapSizeQOneT ({m['liqCapSizeQOneT']})")
    assert_prose_close(cap_matched[1], parse_num(m["liqCapSizeQTwoT"]), tol=0.02,
                       claim=f"capacity-bound size Q2 t = \\liqCapSizeQTwoT ({m['liqCapSizeQTwoT']})")

    for_matched = [t("size", "foreign", q, "TV") for q in qs]
    for_mis_q5 = t("size", "foreign", "Q5", "MC")
    assert_true(all(for_matched[i] > for_matched[i + 1] for i in range(4)),
                claim="foreign S^TV declines monotonically from Q1 to Q5",
                detail=f"t={[f'{x:+.2f}' for x in for_matched]}")
    assert_true(for_matched[4] < for_mis_q5,
                claim="in the largest quintile the foreign matched filter no longer "
                      "leads the mismatched one",
                detail=f"matched t={for_matched[4]:+.2f}, mismatched t={for_mis_q5:+.2f}")
    assert_true(abs(for_matched[4]) < 1.96 and abs(for_mis_q5) < 1.96,
                claim="the advantage is absent from the largest quintile as a whole, not reversed "
                      "(neither filter clears the 5% level there)",
                detail=f"matched t={for_matched[4]:+.2f}, mismatched t={for_mis_q5:+.2f}")
    assert_prose_close(for_matched[4], parse_num(m["liqForSizeQFiveT"]), tol=0.02,
                       claim=f"foreign size Q5 t = \\liqForSizeQFiveT ({m['liqForSizeQFiveT']})")
    assert_prose_close(for_mis_q5, parse_num(m["liqForSizeQFiveMisT"]), tol=0.02,
                       claim=f"foreign size Q5 mismatched t = \\liqForSizeQFiveMisT "
                             f"({m['liqForSizeQFiveMisT']})")

    # Day counts: the prose reads the capacity-bound null in the smallest
    # quintile as a power boundary, so the count that makes that reading true is
    # gated with the t-statistics, against the foreign count for the same bin.
    cap_q1_days = d["by_size"]["capacity_bound"]["Q1"]["univariate_MC"]["n_days"]
    for_q1_days = d["by_size"]["foreign"]["Q1"]["univariate_TV"]["n_days"]
    assert_true(cap_q1_days < 0.2 * for_q1_days,
                claim="capacity-bound flow is scarce among small stocks (the smallest "
                      "quintile is estimable on a small fraction of the sample)",
                detail=f"size Q1 days={cap_q1_days}, foreign size Q1 days={for_q1_days}")
    assert_close(m["liqCapSizeQOneDays"], cap_q1_days, tol=0, label="\\liqCapSizeQOneDays")

    # --- Where the headline sample sits on the size axis ------------------
    comp = d["size_composition"]
    common_q5 = comp["common_support"]["share"]["Q5"]
    foreign_q5 = comp["foreign_nonzero"]["share"]["Q5"]
    assert_true(common_q5 > 2 * foreign_q5,
                claim="common support is a size restriction as well as a "
                      "participation one (it holds a far larger share of its "
                      "stock-days in the top quintile than the foreign class's "
                      "own nonzero-flow days do)",
                detail=f"common support Q5 share={common_q5:.4f}, "
                       f"foreign nonzero-flow Q5 share={foreign_q5:.4f}")
    assert_true(common_q5 > max(comp["common_support"]["share"][q] for q in qs[:4]),
                claim="the top quintile is the modal cell of the common-support "
                      "sample, so the sample concentrates in the range the "
                      "foreign advantage is absent from",
                detail=", ".join(f"{q}={comp['common_support']['share'][q]:.4f}"
                                 for q in qs))
    assert_true(abs(foreign_q5 - 0.2) < 0.01,
                claim="the foreign class's own nonzero-flow days are the panel "
                      "itself, so they carry no size restriction to compare "
                      "against",
                detail=f"foreign nonzero-flow Q5 share={foreign_q5:.4f} "
                       f"of {comp['foreign_nonzero']['n_stock_days']:,} stock-days")
    for macro, val in (("liqCommonSizeQFivePct", common_q5),
                       ("liqForSizeQFivePct", foreign_q5)):
        assert_prose_close(100 * val, parse_num(m[macro]), tol=0.05,
                           claim=f"top-quintile share = \\{macro} ({m[macro]}\\%)")
