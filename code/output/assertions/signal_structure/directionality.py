"""Checks on the persistence and directionality of order flow, measured by the
scripts of code/empirical/signal_structure/."""

from __future__ import annotations

import json

from output.assertions.common import (
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def check_flow_persistence(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:regime_assignment (tab:directionality_kr);
    qf_paper.tex sec:appendix_scaling
    (round-trip-execution vs net-accumulation decomposition)."""
    fp = json.loads((RESULTS / "flow_persistence_decomposition.json").read_text())
    kr, krp = fp["korea"], fp["korea_preperiod"]

    def d(blk: dict) -> float:
        return blk["directionality_median"]

    # subsec:regime_assignment, "The raw ratios repeat the pattern (Panel B of
    # tab:directionality_kr)": the pre-period medians carry no macro, so the
    # ordering the sentence states is asserted on the data the table prints.
    assert_true(d(krp["capacity_bound"]) > d(krp["foreign"]),
                claim="the raw directionality ratios repeat the class ordering on the "
                      "pre-period (subsec:regime_assignment)",
                detail=f"capacity {d(krp['capacity_bound']):.3f}, "
                       f"foreign {d(krp['foreign']):.3f}")
    assert parse_num(m["persistMinDays"]) == float(fp["config"]["min_days"]), (
        f"persistMinDays={m['persistMinDays']} vs config={fp['config']['min_days']}")


def check_directionality_frequency_null(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:regime_assignment (the null and pre-period
    paragraphs), qf_paper.tex subsec:appendix_directionality_frequency,
    tab:directionality_frequency.

    Three guards. (1) Fixed point against drift: the null is measured on the
    stocks and days the published ratio is measured on, so the observed ratio in
    this JSON must equal the one in results/flow_persistence_decomposition.json. (2) The
    directional claim that the ordering survives the null, which is what the
    main-paper sentence spends. (3) The macros the two sections quote.

    The excess ordering is load-bearing rather than descriptive: the corroboration
    rests on it, not on the raw ratio, because the raw ratio's zero point moves
    with the day count and the two classes do not trade equally often."""
    dfn = json.loads((RESULTS / "directionality_frequency_null.json").read_text())
    fp = json.loads((RESULTS / "flow_persistence_decomposition.json").read_text())

    for window in ("korea", "korea_preperiod"):
        for cls in ("capacity_bound", "foreign"):
            assert_prose_close(
                dfn[window][cls]["observed"],
                fp[window][cls]["directionality_median"],
                tol=5e-6,
                claim=(f"the frequency null is measured on the published ratio's own "
                       f"sample ({window}/{cls} must match, no script drift)"),
            )

        cap, for_ = dfn[window]["capacity_bound"], dfn[window]["foreign"]
        ordering = dfn[window]["ordering"]
        # "Capacity-bound flow has an excess directionality of X and foreign flow
        # of Y, so the two classes fall on opposite sides of the value one, as
        # the maintained hypothesis predicts." The unit boundary is what
        # "opposite sides" asserts,
        # so it is asserted rather than read.
        assert_true(cap["excess"] > 1.0 > for_["excess"],
                    claim=("the two classes fall on opposite sides of their own "
                           "trading-frequency null, the ordering their mandates "
                           "predict"),
                    detail=f"{window}: cap={cap['excess']:.3f}, foreign={for_['excess']:.3f}")
        # "the capacity-bound class leads by a factor of N once each class is
        # measured against its own day count. By comparison, the capacity-bound
        # null exceeds the foreign null by a factor of M on day counts alone" —
        # the null must not reproduce either the published or the surviving
        # ordering.
        assert_true(ordering["null"] < ordering["observed"],
                    claim=("trading frequency alone does not deliver the published "
                           "directionality ordering"),
                    detail=(f"{window}: null={ordering['null']:.3f}x, "
                            f"observed={ordering['observed']:.3f}x"))
        assert_true(ordering["excess"] > ordering["null"],
                    claim=("the ordering that survives the day count exceeds the one "
                           "the day count delivers on its own"),
                    detail=(f"{window}: excess={ordering['excess']:.3f}x, "
                            f"null={ordering['null']:.3f}x"))
        # Each cell separates from its null at the randomisation resolution
        # floor, which is what the appendix's "p <= 0.002" states. The macro
        # itself is checked once below, because the emitter refuses to write it
        # unless all four cells agree on the value.
        floor = 2.0 / (dfn["config"]["n_draws"] + 1)
        for cls in ("capacity_bound", "foreign"):
            assert_true(dfn[window][cls]["p_two_sided"] <= floor + 1e-12,
                        claim="every cell rejects its null at the randomisation floor",
                        detail=f"{window}/{cls}: p={dfn[window][cls]['p_two_sided']:.4f}")

    assert_close(m["dirNullP"], dfn["korea"]["capacity_bound"]["p_two_sided"],
                 tol=5e-4, label="dirNullP")

    # Taiwan (sec:taiwan, Panel C of tab:directionality_frequency). The ratio
    # orders the two classes as in Korea, but SITC does not sit above its own
    # null in either measurement space, so the Korean corroboration reaches
    # the Taiwanese assignment in its ordering and not in its level. Both
    # halves are asserted, because the section reports both.
    tw, tws = dfn["taiwan"], dfn["taiwan_share_space"]
    floor = 2.0 / (dfn["config"]["n_draws"] + 1)
    for label, blk in (("value", tw), ("share", tws)):
        assert_true(blk["institutional"]["excess"] > blk["foreign"]["excess"],
                    claim="sec:taiwan: the directionality ratio against its own null "
                          f"orders SITC above foreign flow ({label} space), as the "
                          "Korean classes are ordered",
                    detail=(f"SITC={blk['institutional']['excess']:.3f}, "
                            f"foreign={blk['foreign']['excess']:.3f}"))
        assert_true(blk["institutional"]["excess"] < 1.0,
                    claim="sec:taiwan: SITC does not sit above its own sign-flip null "
                          f"({label} space), where Korean capacity-bound flow does",
                    detail=f"excess={blk['institutional']['excess']:.3f}")
    assert_true(tw["institutional"]["p_two_sided"] > 0.05,
                claim="sec:taiwan: in currency units SITC does not separate from its "
                      "own null",
                detail=f"p_value_space={tw['institutional']['p_two_sided']:.4f}")
    assert_true(tws["institutional"]["p_two_sided"] <= 0.05,
                claim="sec:taiwan: in share counts SITC falls below its own null",
                detail=f"p_share_space={tws['institutional']['p_two_sided']:.4f}")
    # The dagger in tab:directionality_frequency is generated from this same
    # comparison, so the table and the claim cannot part.
    assert_true(tw["institutional"]["p_two_sided"] > floor + 1e-12,
                claim="Taiwan SITC in currency units is the cell whose randomisation p-value "
                      "stands above the resolution floor the draw count sets",
                detail=f"p={tw['institutional']['p_two_sided']:.4f} vs floor={floor:.4f}")
    assert_close(m["dirNullTwSitcP"], tw["institutional"]["p_two_sided"], tol=5e-4,
                 label="dirNullTwSitcP")
    assert_close(m["dirNullTwSitcPShare"], tws["institutional"]["p_two_sided"],
                 tol=5e-5, label="dirNullTwSitcPShare")
    assert_close(m["dirNullTwSitcExcessShare"], tws["institutional"]["excess"],
                 tol=5e-3, label="dirNullTwSitcExcessShare")
    assert_close(m["dirNullTwForExcessShare"], tws["foreign"]["excess"],
                 tol=5e-3, label="dirNullTwForExcessShare")


def check_prose_flow_persistence(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:regime_assignment (tab:directionality_kr);
    qf_paper.tex sec:appendix_scaling.

    The directionals that subsec:regime_assignment rests on:

    (1) Net accumulation is M-dominant for every Korean class, so the
        net-accumulation magnitude alone does not separate the two classes.
        subsec:regime_assignment states the fact, "The net accumulation scales
        more with market capitalization for both classes", and takes the step
        from an accumulating class to a signal proportional to market
        capitalization as a maintained hypothesis.
    (2) Gross flow is M-dominant for Korean capacity-bound flow and
        V-dominant for Korean foreign flow, which subsec:regime_assignment
        states as "The gross flow, which the round trips dominate, has the
        larger elasticity on market capitalization for capacity-bound
        institutions and on traded value for foreign flow in a cross-stock
        regression on both characteristics". Both column pairs of
        tab:directionality_kr therefore carry a claim, and both are asserted
        here.
    (3) The directionality ratio separates capacity-bound from foreign flow
        on the analysis window and on the non-overlapping Korean pre-period,
        the return-free statistic that corroborates the assignment.
    """
    fp = json.loads((RESULTS / "flow_persistence_decomposition.json").read_text())
    kr, krp = fp["korea"], fp["korea_preperiod"]

    # (1) net accumulation is cap (M)-dominant for every Korean class.
    for market_name, market in [("korea", kr), ("korea_preperiod", krp)]:
        for cls, blk in market.items():
            if not isinstance(blk, dict) or "net_accum" not in blk:
                continue
            na = blk["net_accum"]
            assert_true(abs(na["beta_M"]) > abs(na["beta_V"]),
                        claim="'The net accumulation scales more with market capitalization "
                              "for both classes', so 'the net-accumulation magnitude alone "
                              "does not' separate them",
                        detail=f"{market_name}/{cls}: net beta_M={na['beta_M']:+.3f}, beta_V={na['beta_V']:+.3f}")

    # (2) gross flow: Korean capacity-bound M-dominant (cap-scaled execution);
    #     volume-executing classes V-dominant. The pre-period rows are
    #     displayed in tab:directionality_kr Panel B, so their dominance
    #     pattern (bolded cells) is asserted alongside the analysis window.
    for label, blk in [("Korea capacity-bound", kr["capacity_bound"]),
                       ("Korea capacity-bound (pre-period)", krp["capacity_bound"])]:
        g = blk["gross_flow"]
        assert_true(abs(g["beta_M"]) > abs(g["beta_V"]),
                    claim="capacity-bound gross flow 'has the larger elasticity on market "
                          "capitalization', so its round trips 'scale with the characteristic "
                          "of its scaling role'",
                    detail=f"{label}: gross beta_M={g['beta_M']:+.3f}, beta_V={g['beta_V']:+.3f}")
    for label, blk in [("Korea foreign", kr["foreign"]),
                       ("Korea foreign (pre-period)", krp["foreign"])]:
        g = blk["gross_flow"]
        assert_true(abs(g["beta_V"]) > abs(g["beta_M"]),
                    claim="foreign gross flow has the larger elasticity 'on traded value for "
                          "foreign flow', so its round trips 'scale with the characteristic "
                          "of its scaling role'",
                    detail=f"{label}: gross beta_M={g['beta_M']:+.3f}, beta_V={g['beta_V']:+.3f}")

    # (3) directionality ratio discriminates capacity-bound from foreign,
    #     in-sample, on the pre-period holdout, and out-of-market.
    assert_true(kr["capacity_bound"]["directionality_median"] > kr["foreign"]["directionality_median"],
                claim="Korea: capacity-bound flow is more directional than foreign",
                detail=f"cap={kr['capacity_bound']['directionality_median']:.3f}, "
                       f"foreign={kr['foreign']['directionality_median']:.3f}")
    assert_true(krp["capacity_bound"]["directionality_median"] > krp["foreign"]["directionality_median"],
                claim="Korean 2015-2019 pre-period holdout: capacity-bound is more directional than "
                      "foreign without consulting returns (return-free corroboration)",
                detail=f"cap={krp['capacity_bound']['directionality_median']:.3f}, "
                       f"foreign={krp['foreign']['directionality_median']:.3f}")
