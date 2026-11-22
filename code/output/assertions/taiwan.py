"""Checks on the results the scripts of code/empirical/taiwan/ write."""

from __future__ import annotations

import json

from output.assertions.common import (
    RESULTS,
    assert_prose_close,
    assert_true,
    parse_num,
)


def check_taiwan(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:taiwan (panel size and window)."""
    twr = json.loads((RESULTS / "twn_return_prediction_results.json").read_text())
    # sec:taiwan states the Taiwanese window as the Korean panel's own, so the
    # two macros are compared rather than the claim left to the reader. The
    # window is also read off the panel the return prediction ran on, which is
    # what makes \twnYears a measurement rather than a restatement of
    # \sampleYears.
    assert_true(m["twnYears"] == m["sampleYears"],
                claim="sec:taiwan: the Taiwan panel spans \\twnYears, the Korean "
                      "panel's window",
                detail=f"\\twnYears {m['twnYears']}, \\sampleYears {m['sampleYears']}")
    assert_true(m["twnYears"] == f"{twr['panel']['start_date'][:4]}--"
                                 f"{twr['panel']['end_date'][:4]}",
                claim="\\twnYears is the window the Taiwan panel spans",
                detail=f"\\twnYears {m['twnYears']}, panel "
                       f"{twr['panel']['start_date']}..{twr['panel']['end_date']}")
    assert parse_num(m["twnPanelRows"]) == float(twr["panel"]["n_rows"]), (
        f"twnPanelRows={m['twnPanelRows']} vs {twr['panel']['n_rows']}")
    assert parse_num(m["twnPanelStocks"]) == float(twr["panel"]["n_stocks"]), (
        f"twnPanelStocks={m['twnPanelStocks']} vs {twr['panel']['n_stocks']}")


def check_prose_taiwan(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:taiwan, the verdicts subsec:second_market
    states from it, and the limit subsec:limitations draws from them.

    The directional claims the section spends and the construction facts it
    discloses. The numeric values behind the return prediction are covered by
    check_taiwan; what is asserted here is the shape of the result the section
    reads off them, which is what a regenerated panel could break while every
    macro still matched its JSON.
    """
    tw = json.loads((RESULTS / "twn_return_prediction_results.json").read_text())["by_investor"]
    sitc, fr = tw["institutional"]["horse_race"], tw["foreign"]["horse_race"]
    assert_true(abs(sitc["signal_TV"]["t_stat"]) < 1.96,
                claim="Taiwan SITC: 'the investment trusts' matched filter predicts "
                      "returns better than their mismatched filter' "
                      "(subsec:second_market), whose S^TV carries no significant "
                      "loading in the horse race",
                detail=f"|t_TV|={abs(sitc['signal_TV']['t_stat']):.2f}")
    assert_true(sitc["signal_MC"]["t_stat"] > 2.576,
                claim="Taiwan SITC: the matched filter S^MC clears the 1% level in the "
                      "horse race, so the collapse of S^TV is a contrast and not a "
                      "jointly weak cell",
                detail=f"t_MC={sitc['signal_MC']['t_stat']:.2f}")
    assert_true(fr["signal_MC"]["t_stat"] > 2.576 and fr["signal_TV"]["t_stat"] > 2.576,
                claim="Taiwan foreign: both horse-race t-statistics that "
                      "tab:twn_return_prediction prints for foreign flow clear the 1% "
                      "level, the cells behind 'foreign flow's matched filter has "
                      "neither the larger univariate slope nor the larger horse-race "
                      "t-statistic' (subsubsec:kospi200_falsification)",
                detail=f"t_MC={fr['signal_MC']['t_stat']:.2f}, t_TV={fr['signal_TV']['t_stat']:.2f}")

    # The survivorship exposure the live registry leaves. The vendor's delisting
    # record is what bounds it, and sec:taiwan states the bound rather than
    # claiming the panel has none.
    cov = json.loads((RESULTS / "twn_universe_coverage.json").read_text())
    assert_prose_close(len(cov["survivorship_missing_stocks"]),
                       parse_num(m["twnSurvivorStocks"]), tol=0.5,
                       claim=f"the delisting record names \\twnSurvivorStocks "
                             f"({m['twnSurvivorStocks']}) stocks that traded inside the "
                             f"window and are absent from the live registry")
    assert_prose_close(cov["survivorship_missing_sessions"],
                       parse_num(m["twnSurvivorSessions"]), tol=0.5,
                       claim=f"those stocks carry \\twnSurvivorSessions "
                             f"({m['twnSurvivorSessions']}) stock-days, which the panel "
                             f"therefore does not hold")

    # The rule that keeps the emerging stock board out of the panel, asserted on
    # the vendor's own shares and price trees rather than through the pipeline,
    # so a rebuild that changed the rule fails here.
    from pathlib import Path

    import pandas as pd

    from data._finmind import emerging_exits
    from paths import (DAILY_DATA_TAIWAN, FINMIND_OHLCV_DIR, FINMIND_SHARES_DIR,
                       TWN_STOCK_INFO)

    panel = pd.read_parquet(DAILY_DATA_TAIWAN, columns=["stock_id", "date"])
    off = {}
    for sid, dates in panel.groupby(panel["stock_id"].astype(str))["date"]:
        shares = pd.to_datetime(pd.read_parquet(
            Path(FINMIND_SHARES_DIR) / f"{sid}.parquet", columns=["date"])["date"])
        sessions = pd.to_datetime(pd.read_parquet(
            Path(FINMIND_OHLCV_DIR) / f"{sid}.parquet", columns=["date"])["date"])
        sessions = sessions.sort_values().reset_index(drop=True)
        carried = sessions.isin(shares)
        kept = sessions[carried & carried.shift(fill_value=True)]
        n = int((~dates.isin(kept)).sum())
        if n:
            off[sid] = n
    assert_true(not off,
                claim="sec:taiwan: a session enters the panel only if the vendor "
                      "reports the stock's shares outstanding on it and on the "
                      "stock's previous session",
                detail=f"{sum(off.values()):,} rows in {len(off)} stocks break "
                       f"the rule: {dict(list(off.items())[:5])}")

    # What the rule removes, read off a record the rule does not use: the
    # registry dates each emerging row it retired, and no panel row of a stock
    # that moved from the emerging stock board inside the window precedes that
    # date.
    first = panel.groupby(panel["stock_id"].astype(str))["date"].min()
    exits = pd.Series(emerging_exits(TWN_STOCK_INFO))
    moved = exits[exits.between(panel["date"].min(), panel["date"].max())]
    moved = moved[moved.index.isin(first.index)]
    assert_true(len(moved) > 0,
                claim="sec:taiwan names the sessions on which a stock traded on "
                      "the emerging stock board before its listing",
                detail="no panel stock's emerging row is retired inside the window")
    early = first[moved.index][first[moved.index] < moved]
    assert_true(early.empty,
                claim="sec:taiwan: the rule removes the sessions on which a stock "
                      "traded on the emerging stock board before its listing",
                detail=f"{len(early)} of {len(moved)} such stocks have panel rows "
                       f"before the registry retires their emerging row: "
                       f"{sorted(early.index)[:5]}")

    ca = json.loads((RESULTS / "twn_return_prediction_results.json").read_text())["corporate_action_sensitivity"]
    assert_prose_close(ca["max_abs_drift"], parse_num(m["twnCorpActionDrift"]),
                       tol=0.005,
                       claim=f"dropping the stock-days whose forward window crosses a "
                             f"disclosed corporate action moves every horse-race t by at "
                             f"most \\twnCorpActionDrift ({m['twnCorpActionDrift']})")
    drifts = {(k, arm): v[arm]["abs_drift"]
              for k, v in ca["by_investor"].items()
              for arm in ("signal_MC", "signal_TV")}
    assert_true(max(drifts.values()) == ca["max_abs_drift"],
                claim="\\twnCorpActionDrift is the largest drift over every horse-race "
                      "cell sec:taiwan reports",
                detail={f"{k}/{a}": round(v, 3) for (k, a), v in drifts.items()})
    for k in ca["by_investor"]:
        for arm in ("signal_MC", "signal_TV"):
            cell = ca["by_investor"][k][arm]
            assert_true(cell["t_stat"] * cell["published_t_stat"] > 0,
                        claim="no Taiwan horse-race sign changes when the "
                              "corporate-action stock-days are dropped",
                        detail=f"{k}/{arm}: {cell['published_t_stat']:.2f} -> "
                               f"{cell['t_stat']:.2f}")


def check_twn_contrast(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:second_market, subsec:size_restricted,
    subsec:limitations, sec:taiwan and tab:twn_contrast.

    The contrast of eq:reversal_contrast on the Taiwan panel (code/empirical/taiwan/twn_contrast.py).
    Every value tab:twn_contrast prints is bound to its JSON. The two
    directional sentences are asserted on the estimates themselves:
    subsec:second_market and subsec:limitations say the contrast on common
    support is positive and significant, the comparison the reading rule of
    code/empirical/taiwan/twn_contrast.py decides, and subsec:size_restricted and subsec:limitations
    say foreign flow in Taiwan does not rank the normalizations as the volume
    regime predicts, which holds while the advantage for foreign flow is not
    significantly negative on either stacking.
    """
    from shared_utils.regression import CRIT_5PCT

    tc = json.loads((RESULTS / "twn_contrast_results.json").read_text())
    for sample, tag in (("common", "Common"), ("union", "Union")):
        uni = tc[sample]["fama_macbeth_univariate"]
        for field, half in (("advantage_capacity", "CapEdge"),
                            ("advantage_foreign", "ForEdge"), ("reversal", "Rev")):
            name = f"twn{tag}{half}"
            t_name = f"twn{tag}{'' if half == 'Rev' else half}T"
            assert_prose_close(uni[field]["mean"] * 1e4, parse_num(m[name]), tol=0.005,
                               claim=f"\\{name} ({m[name]}) is the {field} of "
                                     f"tab:twn_contrast on the {sample} stacking")
            assert_prose_close(uni[field]["t"], parse_num(m[t_name]), tol=0.005,
                               claim=f"\\{t_name} ({m[t_name]}) is its t-statistic")

    common = tc["common"]["fama_macbeth_univariate"]["reversal"]
    assert_true(common["mean"] > 0 and common["t"] > CRIT_5PCT and tc["verdict"]["holds"],
                claim="subsec:second_market says the advantage of S^MC over S^TV is larger "
                      "for the investment trusts than for foreign flow on the stock-days "
                      "that both classes trade, and subsec:limitations that the contrast "
                      "is positive and significant in Taiwan",
                detail=f"contrast {common['mean'] * 1e4:+.2f} bp, t = {common['t']:+.2f}")
    # sec:taiwan says each union advantage is the difference between the
    # class's two univariate slopes in tab:twn_return_prediction, so the union
    # row adds t-statistics and no new point estimate.
    tw = json.loads((RESULTS / "twn_return_prediction_results.json").read_text())["by_investor"]
    for key, field in (("institutional", "advantage_capacity"), ("foreign", "advantage_foreign")):
        slopes = tw[key]["univariate_MC"]["coefficient"] - tw[key]["univariate_TV"]["coefficient"]
        union = tc["union"]["fama_macbeth_univariate"][field]["mean"]
        assert_true(abs(union - slopes) * 1e4 < 0.005,
                    claim="sec:taiwan says the union advantage of each class is the "
                          "difference between its two univariate slopes in "
                          "tab:twn_return_prediction",
                    detail=f"{key}: union {union * 1e4:+.4f} bp against "
                           f"{slopes * 1e4:+.4f} bp from the slopes")
    foreign_t = {s: tc[s]["fama_macbeth_univariate"]["advantage_foreign"]["t"]
                 for s in ("common", "union")}
    assert_true(all(t > -CRIT_5PCT for t in foreign_t.values())
                and tc["common"]["fama_macbeth_univariate"]["advantage_foreign"]["mean"] > 0,
                claim="subsec:size_restricted says the advantage for foreign flow in Taiwan "
                      "is positive on common support, and subsec:limitations that foreign "
                      "flow there does not rank the normalizations as the volume regime "
                      "predicts",
                detail=f"foreign advantage t by stacking: {foreign_t}")
