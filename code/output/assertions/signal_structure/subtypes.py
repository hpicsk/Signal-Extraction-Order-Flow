"""Checks on order flow by KRX-disclosed class and institutional sub-type,
measured by the scripts of code/empirical/signal_structure/."""

from __future__ import annotations

import json

from shared_utils import CRIT_5PCT, CRIT_10PCT

from output._formatters import BPS_PER_RETURN
from output.assertions.common import (
    MANUSCRIPTS,
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def check_prose_subtype_advantage(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:subtype_dose_response; the
    sentence qf_paper.tex subsec:boundary_extent carries.

    The section reports the matched-filter advantage for every disclosed
    category the panel can estimate. Its claims are signs and counts over
    that set, so the set is read off the results file rather than off the
    classes the prose names."""
    dr = json.loads((RESULTS / "subtype_advantage_results.json").read_text())
    classes = dr["classes"]
    # The primary specification of subsec:return_prediction.
    ARM = "univariate"

    assert_close(m["doseClasses"], len(classes), tol=0.5, label="doseClasses")
    # "the eight institutional sub-types with foreign and individual flow"
    # (subsec:subtype_dose_response), read off the categories the script ran.
    names = set(classes) | set(dr["excluded"])
    assert_true({"foreign", "individual"} <= names
                and len(names - {"foreign", "individual"}) == 8,
                claim="the panel's investor categories are the eight institutional "
                      "sub-types with foreign and individual flow",
                detail=f"categories: {sorted(names)}")
    # "The comparison over classes covers \doseClasses investor categories: foreign
    # flow, individual flow, and seven of the eight institutional sub-types",
    # the eighth being the government account (subsec:boundary_extent).
    assert_true(set(dr["excluded"]) == {"government"}
                and {"foreign", "individual"} <= set(classes),
                claim="the estimable categories are foreign flow, individual flow, "
                      "and every institutional sub-type except the government account",
                detail=f"estimable: {sorted(classes)}, excluded: {sorted(dr['excluded'])}")
    # "the government account carries no disclosed flow in the panel"
    assert_true(all(e["panel_size"] == 0 for e in dr["excluded"].values()),
                claim="the government account carries no disclosed flow in the panel "
                      "at all",
                detail=f"excluded: { {k: v['panel_size'] for k, v in dr['excluded'].items()} }")

    foreign = classes["foreign"]
    # "Foreign flow carries the only significantly negative advantage"
    negatives = [k for k, c in classes.items()
                 if c[ARM]["advantage"]["mean"] < 0
                 and abs(c[ARM]["advantage"]["t"]) > CRIT_5PCT]
    assert_true(negatives == ["foreign"],
                claim="foreign flow carries the only significantly negative advantage",
                detail=f"significantly negative classes: {negatives}")

    # "All seven institutional sub-types carry a positive advantage, which is
    # distinguishable from zero in all but the bank sub-type (t = \\doseBankT)."
    institutional = {k: c for k, c in classes.items()
                     if k not in ("foreign", "individual")}
    non_positive = [k for k, c in institutional.items()
                    if c[ARM]["advantage"]["mean"] <= 0]
    assert_true(len(institutional) == 7 and not non_positive,
                claim="all seven institutional sub-types carry a positive advantage",
                detail=f"{len(institutional)} sub-types, non-positive: {non_positive}")
    assert_true(abs(classes["bank"][ARM]["advantage"]["t"]) < CRIT_10PCT,
                claim="the bank sub-type's advantage is indistinguishable from zero",
                detail=f"t={classes['bank'][ARM]['advantage']['t']:+.2f}")
    # "distinguishable from zero in all but one of them"
    # (subsec:boundary_extent) is a count over the sub-types, so it is read off
    # every sub-type the script ran and not off the one the sentence names.
    indistinct = [k for k, c in institutional.items()
                  if abs(c[ARM]["advantage"]["t"]) <= CRIT_5PCT]
    assert_true(indistinct == ["bank"],
                claim="the institutional sub-types' positive advantage is 'distinguishable "
                      "from zero in all but the bank sub-type'",
                detail=f"indistinguishable at the 5% level: {indistinct}")
    assert_close(m["doseBankT"], classes["bank"][ARM]["advantage"]["t"], tol=0.005,
                 label="doseBankT")

    # Individual flow: "has an advantage of ... indistinguishable from zero" and
    # "Both of its normalized flows predict next-day returns negatively"
    individual = classes["individual"][ARM]
    assert_true(abs(individual["advantage"]["t"]) < CRIT_5PCT,
                claim="individual flow's advantage is indistinguishable from zero",
                detail=f"t={individual['advantage']['t']:+.2f}")
    assert_true(individual["b_MC"]["mean"] < 0 and individual["b_TV"]["mean"] < 0,
                claim="'Both of its normalized flows predict next-day returns "
                      "negatively' (individual flow)",
                detail=f"b_MC={individual['b_MC']['mean'] * BPS_PER_RETURN:+.2f}, "
                       f"b_TV={individual['b_TV']['mean'] * BPS_PER_RETURN:+.2f} bps")
    assert_close(m["doseIndAdv"], individual["advantage"]["mean"] * BPS_PER_RETURN,
                 tol=0.005, label="doseIndAdv")
    assert_close(m["doseIndMC"], individual["b_MC"]["mean"] * BPS_PER_RETURN,
                 tol=0.005, label="doseIndMC")
    assert_close(m["doseIndTV"], individual["b_TV"]["mean"] * BPS_PER_RETURN,
                 tol=0.005, label="doseIndTV")
    assert_close(m["doseIndAdvT"], individual["advantage"]["t"], tol=0.005,
                 label="doseIndAdvT")


def check_prose_capacity_concentration(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:appendix_aggregate_alignment, concentration
    paragraph; qf_paper.tex subsec:boundary_extent.

    "Insurance flow alone has an advantage of S^MC over S^TV of \\doseInsAdv
    basis points at t = \\doseInsAdvT, on \\doseInsObsApprox stock-days, and
    pension flow alone \\dosePenAdv basis points at t = \\dosePenAdvT on
    \\dosePenObsApprox. ... so the advantage for capacity-bound flow does not
    depend on either category."

    The paragraph answers a concentration objection, so the assertion that
    carries it is that *each* category clears the level on its own. A future
    panel in which one category stopped doing so would leave both macros
    correct and the paragraph's conclusion false, which is what this gates."""
    classes = json.loads(
        (RESULTS / "subtype_advantage_results.json").read_text())["classes"]
    for key, pref in (("pension", "Pen"), ("insurance", "Ins")):
        adv = classes[key]["univariate"]["advantage"]
        assert_prose_close(adv["mean"] * BPS_PER_RETURN, parse_num(m[f"dose{pref}Adv"]),
                           tol=0.02,
                           claim=f"{key} advantage = \\dose{pref}Adv "
                                 f"({m[f'dose{pref}Adv']}) bps")
        assert_prose_close(adv["t"], parse_num(m[f"dose{pref}AdvT"]), tol=0.02,
                           claim=f"{key} advantage t = \\dose{pref}AdvT "
                                 f"({m[f'dose{pref}AdvT']})")
        assert_prose_close(classes[key]["panel_size"] / 1e6,
                           parse_num(m[f"dose{pref}ObsApprox"].split("~")[0]), tol=0.005,
                           claim=f"{key} panel = \\dose{pref}ObsApprox "
                                 f"({m[f'dose{pref}ObsApprox']}) stock-days")
        assert_true(adv["t"] > CRIT_5PCT,
                    claim=f"'the advantage for capacity-bound flow does not depend on "
                          f"either category', because each category's advantage is "
                          f"significant on its own ({key})",
                    detail=f"{key} advantage t={adv['t']:+.2f} against {CRIT_5PCT:.2f}")


def check_prose_subinvestor_sharpest(_m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:appendix_aggregate_alignment.

    ``Sharpest at the capacity-bound subset'' is a superlative over the KRX
    sub-types, and no table of the paper ranks the sub-types, so the
    subsection may not make it. This check therefore guards the prose alone, because a test that costs more
    than the claim adds cuts or softens the claim instead.
    """
    src = "\n".join(tex.read_text() for tex in MANUSCRIPTS)
    for banned in ("Sharpest at the Capacity-Bound Subset", "sharpens this to"):
        assert_true(
            banned not in src,
            claim="the aggregate-alignment subsection does not call the "
                  "capacity-bound subset the sharpest, a superlative over "
                  "sub-types that no table of the paper ranks",
            detail=f"found {banned!r}",
        )
