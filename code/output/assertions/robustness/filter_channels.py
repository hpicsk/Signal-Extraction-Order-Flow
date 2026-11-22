"""Checks on the channels that enter one normalisation and not the other,
measured by the scripts of code/empirical/robustness/: a lagged traded-value
denominator and the pressure persistent flow would exert."""

from __future__ import annotations

import json

from shared_utils import CRIT_1PCT, CRIT_5PCT, CRIT_10PCT

from output._formatters import BPS_PER_RETURN
from output.assertions.common import (
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def check_prose_lagged_adv_normalisation(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:contrast_halves, tab:contrast_halves;
    qf_paper.tex sec:appendix_nonuniform_channels, subsec:appendix_adv_normalisation.

    The volume denominator contains the numerator, so the section's claim is
    that the reversal survives a denominator built from trading that predates
    day t. Four things are pinned. (i) The same-day arm reproduces
    subsec:reversal_test exactly on the unrestricted sample, under both
    specifications, which is what licenses reading the restricted same-day arm
    as the baseline. (ii) The lagged arm keeps the contrast positive,
    significant, and sign-reversing across the two classes. (iii) The window
    sweep brackets the headline, so the chosen window is not doing the work.
    (iv) The macros read the univariate arm, which is the specification the
    headline reversal is estimated under (subsec:return_prediction); the
    confound is a property of S^TV, so testing it on the conditioning check
    alone would leave the published estimand untested."""
    adv = json.loads((RESULTS / "lagged_adv_normalisation_results.json").read_text())
    ri = json.loads((RESULTS / "reversal_interaction_results.json").read_text())
    cfg = adv["config"]
    headline = cfg["headline_window"]

    # (i) Same specification, same rows, so the two scripts must agree. The
    # cluster t is compared in relative terms because statsmodels' clustered
    # covariance is not bit-reproducible across differently-built frames.
    for sample in ("common", "union"):
        full = adv[sample]["same_day_full"]
        ref = ri[sample]
        assert_true(full["n_rows"] == ref["n_rows"],
                    claim=f"the lagged-ADV script's same-day arm is the "
                          f"subsec:reversal_test specification ({sample} sample)",
                    detail=f"rows {full['n_rows']} vs {ref['n_rows']}")
        for arm in ("fama_macbeth_univariate", "fama_macbeth"):
            assert_true(full[arm]["reversal"]["mean"] == ref[arm]["reversal"]["mean"],
                        claim=f"the lagged-ADV script reproduces the published reversal "
                              f"when the denominator is left at day-t traded value "
                              f"({arm}, {sample} sample)",
                        detail=f"{full[arm]['reversal']['mean']!r} vs "
                               f"{ref[arm]['reversal']['mean']!r}")
        t_new = full["pooled_cluster_date"]["reversal"]["t_stat"]
        t_ref = ref["pooled_cluster_date"]["reversal"]["t_stat"]
        assert_true(abs(t_new / t_ref - 1.0) < 1e-9,
                    claim=f"the same-day arm reproduces the published cluster-by-date "
                          f"inference ({sample} sample)",
                    detail=f"t {t_new!r} vs {t_ref!r}")

    common = adv["common"]
    base, head = common["same_day"], common[f"adv_{headline}"]

    assert_close(m["advWindow"], headline, tol=0.5, label="advWindow")
    assert_close(m["advWindowMin"], min(cfg["adv_windows"]), tol=0.5, label="advWindowMin")
    assert_close(m["advWindowMax"], max(cfg["adv_windows"]), tol=0.5, label="advWindowMax")
    assert_prose_close(100 * cfg["n_rows_restricted"] / cfg["n_rows_panel"],
                       parse_num(m["advSampleShare"]), tol=0.05,
                       claim=f"retains \\advSampleShare\\% ({m['advSampleShare']}\\%) "
                             f"of the panel")

    # "We re-estimate ... on that sample under both denominators, so the two
    # estimates differ in the denominator alone."
    assert_true(base["n_rows"] == head["n_rows"] and base["n_dates"] == head["n_dates"],
                claim="the two estimates differ in the denominator alone",
                detail=f"same-day {base['n_rows']} rows / {base['n_dates']} days, "
                       f"lagged {head['n_rows']} / {head['n_dates']}")

    for label, arm, rev_key, t_key in (
            ("same-day", base, "advBaseRev", "advBaseT"),
            ("lagged", head, "advRev", "advT")):
        fm = arm["fama_macbeth_univariate"]["reversal"]
        assert_prose_close(fm["mean"] * BPS_PER_RETURN, parse_num(m[rev_key]), tol=0.01,
                           claim=f"the contrast is \\{rev_key} ({m[rev_key]}) basis points "
                                 f"under the {label} denominator")
        assert_prose_close(fm["t"], parse_num(m[t_key]), tol=0.01,
                           claim=f"t = \\{t_key} ({m[t_key]}) under the {label} denominator")
    assert_prose_close(head["pooled_cluster_date"]["reversal"]["t_stat"],
                       parse_num(m["advClusterT"]), tol=0.01,
                       claim=f"and \\advClusterT ({m['advClusterT']}) under cluster-by-date "
                             f"standard errors")

    # "The reversal survives the substitution."
    fm_head = head["fama_macbeth_univariate"]["reversal"]
    assert_true(fm_head["mean"] > 0 and fm_head["t"] > CRIT_1PCT,
                claim="the reversal survives the substitution",
                detail=f"b4-b3={fm_head['mean']:.6f}, t={fm_head['t']:.2f}")
    assert_true(head["pooled_cluster_date"]["reversal"]["t_stat"] > CRIT_1PCT,
                claim="the reversal survives the substitution under cluster-by-date "
                      "standard errors",
                detail=f"t={head['pooled_cluster_date']['reversal']['t_stat']:.2f}")

    # "Both classes' advantages keep the predicted signs" under the lagged denominator.
    # The prose reports a t on each advantage, so each must clear the noise on
    # its own and not only their difference.
    edges = {cls: head["fama_macbeth_univariate"][key]
             for cls, key in (("Cap", "advantage_capacity"),
                              ("For", "advantage_foreign"))}
    assert_true(edges["Cap"]["mean"] > 0 > edges["For"]["mean"],
                claim="both classes' advantages keep the predicted signs under the lagged "
                      "denominator",
                detail=f"capacity={edges['Cap']['mean']:+.6f}, "
                       f"foreign={edges['For']['mean']:+.6f}")
    for cls, label in (("Cap", "capacity-bound institutions"),
                       ("For", "foreign investors")):
        assert_true(abs(edges[cls]["t"]) > CRIT_5PCT,
                    claim=f"the sign change is carried by both classes' advantages in the "
                          f"contrast ({label})",
                    detail=f"t={edges[cls]['t']:+.2f} against {CRIT_5PCT:.2f}")
        assert_prose_close(edges[cls]["mean"] * BPS_PER_RETURN,
                           parse_num(m[f"adv{cls}Edge"]), tol=0.01,
                           claim=f"\\adv{cls}Edge ({m[f'adv{cls}Edge']}) basis points "
                                 f"for {label}")
        assert_prose_close(edges[cls]["t"], parse_num(m[f"adv{cls}EdgeT"]), tol=0.01,
                           claim=f"$t = \\adv{cls}EdgeT$ ({m[f'adv{cls}EdgeT']}) "
                                 f"for {label}")

    # "the estimate does not turn on the length of the window"
    sweep = [common[f"adv_{w}"]["fama_macbeth_univariate"]["reversal"]
             for w in cfg["adv_windows"]]
    assert_prose_close(min(r["mean"] for r in sweep) * BPS_PER_RETURN,
                       parse_num(m["advSweepRevLo"]), tol=0.01,
                       claim=f"between \\advSweepRevLo ({m['advSweepRevLo']}) and "
                             f"\\advSweepRevHi ({m['advSweepRevHi']}) basis points")
    assert_prose_close(max(r["mean"] for r in sweep) * BPS_PER_RETURN,
                       parse_num(m["advSweepRevHi"]), tol=0.01,
                       claim=f"between \\advSweepRevLo ({m['advSweepRevLo']}) and "
                             f"\\advSweepRevHi ({m['advSweepRevHi']}) basis points")
    assert_prose_close(min(r["t"] for r in sweep), parse_num(m["advSweepTLo"]), tol=0.01,
                       claim=f"at t no lower than \\advSweepTLo ({m['advSweepTLo']})")
    assert_true(min(r["t"] for r in sweep) > CRIT_1PCT,
                claim="the estimate does not turn on the length of the window",
                detail=f"window t-statistics: "
                       f"{[round(r['t'], 2) for r in sweep]}")

    union_head = adv["union"][f"adv_{headline}"]["fama_macbeth_univariate"]["reversal"]
    assert_prose_close(union_head["mean"] * BPS_PER_RETURN, parse_num(m["advUnionRev"]),
                       tol=0.01,
                       claim=f"the union stacking gives \\advUnionRev "
                             f"({m['advUnionRev']}) basis points")
    assert_prose_close(union_head["t"], parse_num(m["advUnionT"]), tol=0.01,
                       claim=f"at t = \\advUnionT ({m['advUnionT']})")

    # "Entering the two filters jointly gives \advJointRev basis points."
    joint_head = head["fama_macbeth"]["reversal"]
    assert_prose_close(joint_head["mean"] * BPS_PER_RETURN, parse_num(m["advJointRev"]),
                       tol=0.01,
                       claim=f"entering the two filters jointly gives \\advJointRev "
                             f"({m['advJointRev']}) basis points")
    assert_prose_close(joint_head["t"], parse_num(m["advJointT"]), tol=0.01,
                       claim=f"at $t = \\advJointT$ ({m['advJointT']})")

    # "the bound is \advGap basis points" — a bound, so the prose only holds
    # while the lagged estimate is the smaller of the two.
    gap = (base["fama_macbeth_univariate"]["reversal"]["mean"]
           - fm_head["mean"]) * BPS_PER_RETURN
    assert_true(gap > 0,
                claim="the gap between the contrasts under the two denominators "
                      "is positive, as \\advGap prints it",
                detail=f"same-day minus lagged = {gap:+.4f} bps")
    assert_prose_close(gap, parse_num(m["advGap"]), tol=0.01,
                       claim=f"the bound is \\advGap ({m['advGap']}) basis points")


def check_prose_flow_continuation(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:robustness; qf_paper.tex
    subsec:appendix_flow_continuation, eq:proportionality.

    The section answers the objection that the regime assignment is a
    persistence statistic and persistent flow moving prices would manufacture
    the return result. Two things carry it. (i) The premise holds, so the
    section may not open by denying it: both classes' flow is autocorrelated,
    and each continuation coefficient the prose quotes is pinned. (ii) The
    proportionality restriction eq:proportionality is rejected for both
    classes and under both pressure measures, which is the section's claim and
    the one a regenerated panel could overturn.
    """
    fc = json.loads((RESULTS / "flow_continuation_results.json").read_text())

    for cls, tag in (("capacity_bound", "Cap"), ("foreign", "For")):
        block = fc["classes"][cls]
        cont = block["continuation"]["all"]

        # (i) The premise. Persistence is real, so the answer has to be the
        # joint restriction and not a denial that flow repeats.
        for key, macro in (("mc", f"cont{tag}MC"), ("tv", f"cont{tag}TV")):
            assert_prose_close(cont[key]["mean"], parse_num(m[macro]), tol=0.001,
                               claim=f"\\{macro} ({m[macro]}) is the {cls} class's "
                                     f"day-to-day continuation of its {key.upper()} filter")
        assert_true(
            cont["tv"]["t"] > CRIT_1PCT,
            claim=f"the {cls} class's traded-value filter is autocorrelated, the premise "
                  f"subsec:appendix_flow_continuation concedes before answering it",
            detail=f"t = {cont['tv']['t']:.2f}")

        # (ii) The section's claim, under both pressure measures.
        for measure in ("participation", "capitalisation"):
            prop = block["pressure"][measure]["proportionality"]
            assert_true(
                block["pressure"][measure]["verdict_rejected"],
                claim=f"subsec:appendix_flow_continuation says eq:proportionality is "
                      f"rejected for the {cls} class under {measure} pressure, so "
                      f"the persistence-then-impact channel cannot reproduce both "
                      f"filters' return loadings 'with one factor of proportionality, the "
                      f"day's price impact, for both filters', "
                      f"which is what subsec:surviving_channels states as 'the data "
                      f"reject that proportionality for both classes'",
                detail=f"cross-contrast {prop['cross_contrast_bps']:+.3f} bp, "
                       f"95% CI [{prop['ci_lower_bps']:+.3f}, "
                       f"{prop['ci_upper_bps']:+.3f}], z = {prop['boot_z']:+.2f}")
        part = block["pressure"]["participation"]["proportionality"]
        for value, macro in ((part["cross_contrast_bps"], f"prop{tag}Cross"),
                             (part["ci_lower_bps"], f"prop{tag}CrossLo"),
                             (part["ci_upper_bps"], f"prop{tag}CrossHi")):
            assert_prose_close(value, parse_num(m[macro]), tol=0.01,
                               claim=f"\\{macro} ({m[macro]}) is the {cls} class's "
                                     f"eq:proportionality contrast in basis points")
