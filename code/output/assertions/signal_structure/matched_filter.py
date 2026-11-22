"""Checks on the return-free restriction and the normalisation exponent,
measured on the panel by the scripts of code/empirical/signal_structure/."""

from __future__ import annotations

import json

from shared_utils import CRIT_1PCT

from output._formatters import BPS_PER_RETURN

from output.assertions.common import RESULTS, assert_prose_close, assert_true, parse_num


def check_prose_free_exponent(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:free_exponent, and the verdict
    sentences qf_paper.tex subsec:boundary_extent carries.

    The section estimates the normalisation exponent over the whole
    scale-invariant family instead of testing its two endpoints. Four claims
    carry it. (i) Each class's level profile turns over inside the grid, so
    each class has a best exponent and the gap between them is defined. (ii)
    The gap is positive and its
    bootstrap interval excludes zero on each class's own panel. (iii) The
    assigned endpoint beats the unassigned one in every comparison the section
    prints, in ranks as well as levels. (iv) On common support the foreign
    profile does not locate a maximum and the gap's interval contains zero,
    which is the concession the section makes and which a future panel could
    turn into a pass without anyone noticing.
    """
    fe = json.loads((RESULTS / "free_exponent_profile.json").read_text())
    own = fe["arms"]["own"]

    # (i) Each class has a best exponent, so the gap is defined.
    cap_theta = own["capacity_bound"]["levels"]["theta_hat"]
    for_theta = own["foreign"]["levels"]["theta_hat"]
    assert_true(
        cap_theta is not None and for_theta is not None,
        claim="subsec:free_exponent reports a free exponent for each class, which "
              "requires both level profiles to turn over inside the grid",
        detail=f"capacity {cap_theta}, foreign {for_theta}")

    # (ii) The gap, which is the identified quantity.
    gap = own["gap_levels"]
    assert_true(
        gap["gap"] > 0 and gap["ci"][0] > 0,
        claim="subsec:free_exponent says the gap between the two classes' free "
              "exponents is positive with an interval clear of zero, the ordering "
              "cor:regimes predicts",
        detail=f"gap {gap['gap']:+.2f}, 95% CI [{gap['ci'][0]:+.2f}, "
               f"{gap['ci'][1]:+.2f}], {gap['share_positive'] * 100:.1f}% of draws positive")

    # (iii) The endpoint comparison, in every comparison the section prints.
    printed = [("own", cls, kind) for cls in ("capacity_bound", "foreign")
               for kind in ("levels", "ranks")] + [("common", "foreign", "levels")]
    for arm, cls, kind in printed:
        node = fe["arms"][arm][cls][kind]
        assert_true(
            node["assigned_beats_other"],
            claim=f"subsec:free_exponent says the endpoint cor:regimes "
                  f"assigns each class leads the other endpoint, here for "
                  f"{cls} on the {arm} sample in {kind}",
            detail=f"assigned {node['assigned_share_of_max'] * 100:.1f}% of "
                   f"the grid maximum")
    assert_prose_close(own["capacity_bound"]["levels"]["assigned_share_of_max"] * 100,
                       parse_num(m["fexCapAssignedPct"]), tol=0.05,
                       claim=f"\\fexCapAssignedPct ({m['fexCapAssignedPct']})\\% is the "
                             f"share of its own optimum the capacity endpoint attains, "
                             f"the number subsec:free_exponent prints as 'the "
                             f"capacity endpoint captures' of its optimum")

    # (iv) The concession. Common support does not locate the foreign optimum.
    com_for = fe["arms"]["common"]["foreign"]["levels"]
    grid_lo = fe["config"]["theta_grid"]["lo"]
    assert_true(
        com_for["theta_ci"][0] <= grid_lo,
        claim="subsec:free_exponent says common support does not locate the foreign "
              "maximum, its interval running to the end of the grid",
        detail=f"foreign interval [{com_for['theta_ci'][0]:+.2f}, "
               f"{com_for['theta_ci'][1]:+.2f}] against a grid floor of {grid_lo:+.1f}")
    com_gap = fe["arms"]["common"]["gap_levels"]
    assert_true(
        com_gap["ci"][0] <= 0.0,
        claim="subsec:free_exponent reports the common-support gap as an interval "
              "reaching zero rather than as a second confirmation",
        detail=f"gap {com_gap['gap']:+.2f}, 95% CI [{com_gap['ci'][0]:+.2f}, "
               f"{com_gap['ci'][1]:+.2f}]")


def check_prose_accounting_bound(_m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:threshold_estimate.

    "Net buying cannot exceed traded value, so |S^TV| <= 1": the accounting
    identity |D| <= V holds on every stock-day of both classes' panels, so the
    bound from which the section argues holds in the data."""
    st = json.loads((RESULTS / "accounting_bound.json").read_text())
    for regime in ("capacity", "volume"):
        bound = st[regime]
        assert_true(bound["share_abs_stv_above_one"] == 0.0 and bound["max_abs_stv"] <= 1.0,
                    claim=f"|D| <= V holds as an accounting identity on every "
                          f"{regime}-panel stock-day, capping |S^TV| at one",
                    detail=f"max|S^TV|={bound['max_abs_stv']:.4f}, "
                           f"share>1={bound['share_abs_stv_above_one']:.6f}")


