"""Checks on the common-stock universe and on the boundaries of the capacity-bound
and foreign classes, measured by the scripts of code/empirical/robustness/."""

from __future__ import annotations

import json

import pandas as pd

from shared_utils import CAPACITY_BOUND_COMPONENTS, CRIT_1PCT

from output._formatters import BPS_PER_RETURN
from output.assertions.common import (
    RESULTS,
    assert_close,
    assert_prose_close,
    assert_true,
    parse_num,
)


def _run_kind(sessions: pd.DataFrame, runs: pd.DataFrame) -> pd.Series:
    """The kind of the universe run holding each ``(code, date)`` session, NaN
    where no run of that code covers the date. Runs of one code do not
    overlap, so a session falls in at most one."""
    s = sessions.reset_index(drop=True)
    hit = s.reset_index().merge(runs[["code", "kind", "first_date", "last_date"]],
                                on="code")
    hit = hit[(hit["date"] >= hit["first_date"]) & (hit["date"] <= hit["last_date"])]
    return hit.set_index("index")["kind"].reindex(s.index)


def check_prose_universe_composition(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:appendix_data, the sentence that
    defines common stock.

    The sentence names the security types the universe excludes, and the
    preferred test is the one that reads straight off the code: KRX gives a
    preferred line a terminal character other than '0', so a panel that
    admitted one shows it without any external file being consulted.

    code/data/kr_universe.py takes the issues that FnGuide's all-codes export
    prices as the common stocks, so the universe is checked against that
    export in both directions, over the span the two panels cover. Every
    session the export prices must sit in a common-stock, SPAC or KONEX run:
    a listed-name rule once filed 메리츠금융지주 as a REIT and 율호 as a fund,
    and this is the direction that caught it. Every session of a common-stock
    run must be one the export prices, which is the direction a fund whose
    name escapes every rule fails. The export also prices some issues before
    their KRX listing and after their delisting, and marcap lists no session
    on those dates, so no run covers them and they are not counted.

    The sentence also states the export's own premise, that it prices no REIT
    or listed investment company. No artifact here carries a security type, so
    the premise is read the one way the names allow: no common-stock run
    carries a REIT's listed name, which ends in 리츠."""
    from paths import (DAILY_DATA_ALL_INVESTORS, DAILY_DATA_PRE2020,
                       FNGUIDE_PRICE_CACHE, KR_UNIVERSE_PANEL)

    runs = pd.read_parquet(KR_UNIVERSE_PANEL)
    runs["code"] = runs["code"].astype(str).str.zfill(6)
    common = runs[runs["kind"] == "common"]

    bounds = []
    for path, label in ((DAILY_DATA_ALL_INVESTORS, "main"),
                        (DAILY_DATA_PRE2020, "pre-period")):
        panel = pd.read_parquet(path, columns=["ticker", "date"])
        codes = panel["ticker"].str.lstrip("A")
        # An absent panel would leave nothing to find and report as a pass, so
        # the population is asserted before the property is. Its size is pinned
        # elsewhere (\\panelTotalObsApprox); non-emptiness is what this one owes.
        assert_true(len(codes) > 0,
                    claim=f"the {label} panel exists, so the exclusion below is "
                          f"tested against stocks rather than against nothing",
                    detail=f"{path} loaded 0 rows")
        preferred = sorted(codes[codes.str[-1] != "0"].unique())
        assert_true(not preferred,
                    claim=f"the {label} panel is 'the common stocks of KOSPI and "
                          f"KOSDAQ, which excludes preferred lines'",
                    detail=f"{len(preferred)} preferred-shaped codes among "
                           f"{codes.nunique():,} scanned (terminal character "
                           f"not '0'): {preferred[:5]}")
        kind = _run_kind(pd.DataFrame({"code": codes, "date": panel["date"]}), runs)
        stray = codes[kind.ne("common").to_numpy()]
        assert_true(stray.empty,
                    claim=f"the {label} panel is drawn from 'the common stocks of "
                          f"KOSPI and KOSDAQ' (sec:appendix_data)",
                    detail=f"{len(stray):,} stock-days of {stray.nunique()} codes "
                           f"lie outside every common-stock run of "
                           f"{KR_UNIVERSE_PANEL.name}: {sorted(stray.unique())[:5]}")
        bounds += [panel["date"].min(), panel["date"].max()]
    lo, hi = min(bounds), max(bounds)

    priced = (pd.read_parquet(FNGUIDE_PRICE_CACHE, columns=["date", "ticker"])
                .rename(columns={"ticker": "code"}))
    window = priced[(priced["date"] >= lo) & (priced["date"] <= hi)]
    kind = _run_kind(window, runs)
    bad = (kind.notna() & ~kind.isin(["common", "spac", "konex"])).to_numpy()
    misfiled = window[bad].assign(kind=kind[bad].to_numpy())
    assert_true(misfiled.empty,
                claim="'the universe admits an issue on the sessions for which the "
                      "export gives a price', SPACs and KONEX listings aside "
                      "(sec:appendix_data)",
                detail=f"{len(misfiled):,} sessions of {misfiled['code'].nunique()} "
                       f"codes that FnGuide's all-codes export prices between "
                       f"{lo.date()} and {hi.date()} are filed under another kind: "
                       f"{misfiled.groupby('code')['kind'].first().head(8).to_dict()}")

    live = common[(common["last_date"] >= lo) & (common["first_date"] <= hi)]
    served = priced.merge(live[["code", "first_date", "last_date"]], on="code")
    served = served[(served["date"] >= served["first_date"])
                    & (served["date"] <= served["last_date"])]
    n_served = served.groupby(["code", "first_date"]).size()
    runs_n = live.set_index(["code", "first_date"])["n_days"]
    unserved = runs_n[runs_n > n_served.reindex(runs_n.index, fill_value=0)]
    assert_true(unserved.empty,
                claim="the universe 'excludes preferred lines, SPACs, REITs, other "
                      "listed investment companies, ETFs and KONEX listings' "
                      "(sec:appendix_data)",
                detail=f"{len(unserved)} common-stock runs hold sessions FnGuide's "
                       f"all-codes export does not price: "
                       f"{[c for c, _ in unserved.index[:8]]}")

    reit_named = sorted(common.loc[common["name"].str.endswith("리츠"), "code"].unique())
    assert_true(not reit_named,
                claim="'The DataGuide export gives no price for a preferred line, REIT, "
                      "other listed investment company or ETF' (sec:appendix_data)",
                detail=f"common-stock runs under a REIT's listed name, ending in "
                       f"리츠: {reit_named[:5]}")


def check_prose_foreign_definition(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:appendix_data and its
    subsec:appendix_panel_filters restrictions.

    results/foreign_definition_share.json — expected values read from \\foreignOther*
    macros. The merge guard is asserted alongside them because a failed load
    reports as a 0% share, which reads as the small leg the section claims
    rather than as the absent data it would be. The nonzero-flow assertions cover
    a different sentence: it reads an exact zero as non-participation, and the
    reloaded legs of each class measure how often the class in fact traded to a
    balancing amount. "Most" in that sentence claims more than half of each
    class's zero stock-days, so both shares are held to that bar as well as to
    their macros.
    """
    fd = json.loads((RESULTS / "foreign_definition_share.json").read_text())["panel"]
    assert_prose_close(fd["gross_other_share_pct"], parse_num(m["foreignOtherGrossPct"]),
                       tol=0.05,
                       claim=f"the excluded other-foreigner leg carries "
                             f"\\foreignOtherGrossPct\\% ({m['foreignOtherGrossPct']}%) "
                             f"of foreign value traded")
    assert_prose_close(fd["sign_flip_share_pct"], parse_num(m["foreignOtherFlipPct"]),
                       tol=0.05,
                       claim=f"adding the residual back flips the sign of daily net "
                             f"foreign flow on \\foreignOtherFlipPct\\% "
                             f"({m['foreignOtherFlipPct']}%) of stock-days")
    assert_true(fd["max_registered_mismatch_krw"] == 0.0,
                claim="the reloaded registered leg reproduces the panel's own D_foreign, "
                      "so the excluded-leg shares measure a leg rather than a failed load",
                detail=f"max |diff| = {fd['max_registered_mismatch_krw']:,.0f} KRW")
    absent = 100.0 * (fd["n_zero_net"] - fd["n_zero_net_with_gross"]) / fd["n_zero_net"]
    assert_prose_close(absent, parse_num(m["foreignZeroAbsentPct"]), tol=0.05,
                       claim=f"foreign investors do not trade the stock 'on "
                             f"\\foreignZeroAbsentPct\\% ({m['foreignZeroAbsentPct']}%) "
                             f"of theirs', their stock-days with D = 0 "
                             f"(subsec:appendix_panel_filters)")
    cap = fd["capacity_bound"]
    assert_true(cap["max_net_mismatch_krw"] == 0.0,
                claim="the reloaded pension and insurance legs reproduce the panel's own "
                      "D_capacity_bound, so the zero-net counts measure the class rather "
                      "than a failed load",
                detail=f"max |diff| = {cap['max_net_mismatch_krw']:,.0f} KRW")
    cap_absent = (100.0 * (cap["n_zero_net"] - cap["n_zero_net_with_gross"])
                  / cap["n_zero_net"])
    assert_prose_close(cap_absent, parse_num(m["capZeroAbsentPct"]), tol=0.05,
                       claim=f"'Capacity-bound institutions do not trade the stock at all "
                             f"on \\capZeroAbsentPct\\% ({m['capZeroAbsentPct']}%) of their "
                             f"stock-days with D = 0' (subsec:appendix_panel_filters)")
    for cls, share in (("capacity-bound", cap_absent), ("foreign", absent)):
        assert_true(share > 50.0,
                    claim=f"'a zero mostly records that the class did not trade the "
                          f"stock that day' (subsec:data) holds for the {cls} class",
                    detail=f"no-trade share {share:.1f}% of its zero stock-days")


def check_investment_trust_inclusion(m: dict[str, str]) -> None:
    """Covers: the criteria of code/empirical/robustness/investment_trust_inclusion.py;
    qf_paper.tex subsec:appendix_aggregate_alignment, the paragraph on the
    investment-trust sub-type; the sentence of subsec:data that cites it.

    The criteria were committed before the run, and both sentences report their
    verdict as the contrast staying significant at the 1% level with the
    sub-type in the class. (i) The run adds the sub-type, and only the sub-type,
    to the published class. (ii) The printed values are the run's. (iii) Both
    criteria hold, recomputed from the values rather than read off the stored
    verdicts. (iv) The common support grows, the reason the paragraph gives for
    the advantage for foreign flow moving.
    """
    it = json.loads((RESULTS / "investment_trust_inclusion.json").read_text())
    # (i) "The class leaves out the investment-trust sub-type"
    assert_true("investment_trust" not in CAPACITY_BOUND_COMPONENTS
                and it["config"]["components"] == [*CAPACITY_BOUND_COMPONENTS, "investment_trust"],
                claim="the capacity-bound class leaves out the investment-trust sub-type, "
                      "and the run adds it to the published class alone "
                      "(subsec:appendix_aggregate_alignment)",
                detail=f"class {CAPACITY_BOUND_COMPONENTS}, run {it['config']['components']}")
    # (ii)
    common = it["common"]
    fm = common["fama_macbeth_univariate"]
    for macro, value in (("revTrustRev", fm["reversal"]["mean"] * BPS_PER_RETURN),
                         ("revTrustT", fm["reversal"]["t"]),
                         ("revTrustCapEdge", fm["advantage_capacity"]["mean"] * BPS_PER_RETURN),
                         ("revTrustCapEdgeT", fm["advantage_capacity"]["t"]),
                         ("revTrustForEdge", fm["advantage_foreign"]["mean"] * BPS_PER_RETURN),
                         ("revTrustForEdgeT", fm["advantage_foreign"]["t"])):
        assert_close(m[macro], value, tol=0.005, label=macro)
    rows = set(common["n_rows_by_class"].values())
    assert_true(len(rows) == 1,
                claim="the common-support stacking holds one row count for both classes",
                detail=f"{common['n_rows_by_class']}")
    assert_close(m["revTrustStockDays"], rows.pop(), tol=0.5, label="revTrustStockDays")
    # (iii) "The Korean contrast stays significant at the 1% level" (sec:taiwan)
    for key, pack in (("T1", "reversal"), ("T2", "advantage_capacity")):
        assert_true(fm[pack]["mean"] > 0 and fm[pack]["t"] > CRIT_1PCT,
                    claim="with the investment-trust sub-type in the class, the contrast "
                          "stays significant at the 1% level (subsec:data; "
                          f"subsec:appendix_aggregate_alignment), criterion {key} of "
                          "investment_trust_inclusion.py",
                    detail=f"{pack}: mean {fm[pack]['mean']:.6f}, t {fm[pack]['t']:.2f}")
    # (iv) "the common support ... grows from \revCommonStockDays to \revTrustStockDays"
    assert_true(parse_num(m["revTrustStockDays"]) > parse_num(m["revCommonStockDays"]),
                claim="the common support grows when the class gains the sub-type's flow "
                      "(subsec:appendix_aggregate_alignment)",
                detail=f"{m['revCommonStockDays']} -> {m['revTrustStockDays']}")
