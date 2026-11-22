"""Checks on the daily panels code/data/ builds and on the normalisation
shared_utils applies to them."""

from __future__ import annotations

import pandas as pd

from output.assertions.common import (
    PAPER_TEX,
    RESULTS,
    ROOT,
    assert_close,
    assert_true,
    parse_num,
)


def check_prose_panel_survivorship(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:data ("including the stocks delisted during
    that period").

    It is a premise about the panel, stated as verified in prose and
    re-verified by nothing. A survivorship-filtered rebuild would satisfy every
    other assertion in the suite, because every downstream statistic would move
    together. The observable signature is names whose last observation precedes
    the sample end, so that is what is checked. It is a floor on retention, not
    a bound on what is missing.
    """
    for label, name, col, floor in (
            ("Korea", "full_daily_data_all_investors.parquet", "ticker", 100),):
        df = pd.read_parquet(RESULTS / name, columns=[col, "date"])
        end = df["date"].max()
        # A month's grace, so a name that simply stops trading in the final
        # week is not counted as a delisting.
        delisted = int((df.groupby(col)["date"].max()
                        < end - pd.Timedelta(days=30)).sum())
        assert_true(delisted >= floor,
                    claim=f"the {label} panel retains names delisted within the window, so it is not survivorship-filtered",
                    detail=f"{delisted} of {df[col].nunique()} names last observed "
                           f"more than 30 days before {end.date()}")


def check_prose_no_trade_sessions(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:data ("A session in which a stock did not
    trade is dropped, so an $h$-day forward return spans the stock's next $h$
    sessions in the panel") and sec:taiwan ("sessions in which a stock did not
    trade are dropped").

    Both halves are read off the published panels: no row carries a session
    without volume, and every stored forward return is the return to the
    stock's h-th next row, so a dropped session lies inside a return rather
    than ending it.
    """
    import numpy as np

    from paths import DAILY_DATA_ALL_INVESTORS, DAILY_DATA_TAIWAN

    for label, path, key in (("Korea", DAILY_DATA_ALL_INVESTORS, "ticker"),
                             ("Taiwan", DAILY_DATA_TAIWAN, "stock_id")):
        df = pd.read_parquet(path, columns=[key, "date", "close", "volume",
                                            "R_t1", "R_t5", "R_t20"])
        untraded = int((df["volume"] <= 0).sum())
        assert_true(untraded == 0,
                    claim=f"the {label} panel drops every session in which a stock did not trade",
                    detail=f"{untraded:,} rows carry no volume")
        df = df.sort_values([key, "date"])
        close = df.groupby(key)["close"]
        for h in (1, 5, 20):
            to_row = close.shift(-h) / df["close"] - 1
            stored = df[f"R_t{h}"]
            same = np.isclose(stored, to_row, rtol=1e-9, atol=1e-12, equal_nan=True)
            assert_true(bool(same.all()),
                        claim=f"a {h}-day forward return in the {label} panel spans the stock's next {h} sessions in the panel (subsec:data)",
                        detail=f"{int((~same).sum()):,} of {len(df):,} rows store an R_t{h} "
                               f"other than the return to the stock's {h}-th next row")


def check_matched_filter_algebra(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:theory (matched-filter definition, governance).

    Toy DataFrame round-trip through ``compute_normalisations``.
    Verifies S^MC = D/M, S^TV = D/V with V=0 → NaN mask applied identically."""
    import math
    from shared_utils.data import compute_normalisations
    df = pd.DataFrame({"D": [100.0, -200.0, 0.0, 50.0, -10.0],
                       "M": [1e9, 2e9, 5e9, 1e10, 5e8],
                       "V": [1e7, 0.0, 1e8, 1e9, 1e6]})
    compute_normalisations(df)
    expected_smc = df["D"] / df["M"]
    assert (abs(df["S_MC"] - expected_smc) < 1e-15).all(), "S_MC algebra mismatch"
    assert math.isnan(df["S_TV"][1]), "V=0 row must produce NaN in S_TV"
    keep = df["V"] > 0
    assert (abs(df.loc[keep, "S_TV"] - (df.loc[keep, "D"] / df.loc[keep, "V"])) < 1e-15).all(), "S_TV algebra mismatch"


def check_turnover_descriptive_is_the_panel(m: dict[str, str]) -> None:
    """Covers: a count prints its own population's unit noun.

    \\descTurnMean and \\descTurnStd are printed in sec:appendix_data beside the noun
    "the panel". They were emitted over the stock-days with nonzero
    capacity-bound flow, 34.3% of it, so the sentence named one population and
    the macros measured another and every binding stayed correct. Recompute
    both from the panel itself, and check the noun beside them is still the
    one the emitter reads.
    """
    from paths import DAILY_DATA_ALL_INVESTORS
    turn = pd.read_parquet(DAILY_DATA_ALL_INVESTORS, columns=["turnover"])["turnover"]
    assert_close(m["descTurnMean"], turn.mean() * 100, tol=0.05, label="descTurnMean")
    assert_close(m["descTurnStd"], turn.std(ddof=0) * 100, tol=0.05, label="descTurnStd")

    sentence = next(
        (ln for ln in PAPER_TEX.read_text().splitlines()
         if "\\descTurnMean" in ln), "")
    span = sentence[max(0, sentence.find("\\descTurnMean") - 120):
                    sentence.find("\\descTurnMean")]
    assert_true(
        "panel" in span,
        claim="the sentence printing \\descTurnMean names the population the "
              "macro is measured over",
        detail=f"the 120 characters before the macro read: {span!r}")


def check_min_stocks_floor_matches_its_appendix(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:appendix_panel_filters, the
    sentence printing \\minStocksPerDay, and the $T$ column of
    qf_paper.tex tab:liquidity_stratified, which records what that floor removes.

    The sentence states a rule that applies everywhere and a bite that does
    not, and nothing else in the suite reads either half. Every macro around
    the floor binds whether it removed one day or nine hundred, so a sentence
    generalising from the unstratified panel to the stratified cells stays
    green, because the quantifier ranges over cells that the paper enumerates
    itself.

    Both halves are asserted against the panel rather than against the members
    the sentence names. The class list is derived from the panel's own flow
    columns, and each stratified cell's surviving day count is recomputed here
    under the same floor the scripts apply, then matched to the count the
    appendix prints and to the figure the table body renders.
    """
    import json

    import numpy as np
    import pyarrow.parquet as pq

    from shared_utils.config import MIN_STOCKS_PER_DAY_DEFAULT as FLOOR
    from shared_utils.data import within_day_quantile

    panel_path = RESULTS / "full_daily_data_all_investors.parquet"
    names = pq.ParquetFile(panel_path).schema.names
    classes = sorted(c[2:] for c in names if c.startswith("D_")
                     and f"S_MC_{c[2:]}" in names and f"S_TV_{c[2:]}" in names)
    assert_true(len(classes) > 1,
                claim="the panel carries the investor classes the floor is asserted over",
                detail=f"derived classes: {classes}")

    # --- Half one: the floor does not bind on any class's full cross-section.
    cols = ["date", "ticker", "M", "V", "R_t", "R_t1"]
    cols += [f"{p}_{c}" for c in classes for p in ("D", "S_MC", "S_TV")]
    df = pd.read_parquet(panel_path, columns=cols)
    thin = {}
    for c in classes:
        sub = df[df[f"D_{c}"] != 0].dropna(
            subset=[f"S_MC_{c}", f"S_TV_{c}", "R_t1"])
        if sub.empty:
            continue
        n = sub.groupby("date").size()
        if int((n < FLOOR).sum()):
            thin[c] = (int((n < FLOOR).sum()), int(n.min()))
    assert_true(
        not thin,
        claim="the cross-section floor does not bind on the full cross-sections "
              "(subsec:appendix_panel_filters)",
        detail=f"classes with a day under {FLOOR}: {thin}")

    # --- Half two: it does bind inside the stratified cells, and every day
    # count it leaves there is the one the appendix prints.
    published = json.loads(
        (RESULTS / "liquidity_stratified_results.json").read_text())
    labels = [f"Q{i}" for i in range(1, 6)]
    df["size_q"] = within_day_quantile(df, "M", labels)

    body = (ROOT / "tables"
            / "liquidity_stratified_body.tex").read_text()
    rendered = {t.replace("\\,", "").replace(",", "")
                for t in body.replace("&", " ").split()}
    surviving: dict[str, list[int]] = {}
    for axis, block in (("size_q", "by_size"),):
        for cls, arm in (("capacity_bound", "MC"), ("foreign", "TV")):
            for q in labels:
                sub = df[(df[axis] == q) & (df[f"D_{cls}"] != 0)].dropna(
                    subset=[f"S_{arm}_{cls}", "R_t1"])
                size = sub.groupby("date").size()
                spread = sub.groupby("date")[f"S_{arm}_{cls}"].std()
                got = int(((size >= FLOOR) & (spread > 0)
                           & np.isfinite(spread)).sum())
                want = published[block][cls][q][f"univariate_{arm}"]["n_days"]
                assert_true(
                    got == int(want),
                    claim=f"the {block} {cls} {q} cell keeps the days the floor "
                          f"leaves it (tab:liquidity_stratified, column $T$)",
                    detail=f"recomputed under a floor of {FLOOR}: {got}; "
                           f"published: {want}")
                assert_true(
                    str(want) in rendered,
                    claim=f"the {block} {cls} {q} day count reaches the reader, "
                          f"as every cell that the results data hold does",
                    detail=f"{want} is absent from liquidity_stratified_body.tex")
                surviving.setdefault(block, []).append(got)

    # A cell the floor never reached keeps every day the axis offers, so the
    # bite is the shortfall against each axis's own maximum rather than against
    # a day count named here.
    short = {block: [n for n in counts if n < max(counts)]
             for block, counts in surviving.items()}
    assert_true(
        any(short.values()),
        claim="the floor binds inside the stratified cells, which is what the "
              "appendix's $T$ column reports (subsec:appendix_panel_filters)",
        detail=f"cells short of their axis maximum: "
               f"{ {b: len(v) for b, v in short.items()} }")


def check_union_stacking_is_the_panels_with_a_return(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:reversal_test, "The union stacking gives a
    sample of \\revUnionRows{} rows", the stacking "which stacks each class on
    its own nonzero-flow stock-days with a next-day return", and the common
    support it names; qf_paper.tex subsec:data, the
    two panel sizes; appendix A's two sentences that run on common support.

    The panel sizes that subsec:data prints count every nonzero-flow stock-day
    of the window, and the stacked regressions keep those with a next-day
    return, so a reader summing the two sizes lands above the stacked row
    count. The sentence names the stacking's population, and this check
    recomputes it from the panel class by class, so a stacking that drops rows
    for any other reason, or a panel size printed under another filter, fails
    here instead of leaving the reader's sum short.
    """
    import json

    from shared_utils import END_DATE_MAIN, START_DATE_MAIN

    ri = json.loads((RESULTS / "reversal_interaction_results.json").read_text())
    by_class = ri["union"]["n_rows_by_class"]
    df = pd.read_parquet(RESULTS / "full_daily_data_all_investors.parquet",
                         columns=["date", "R_t1"] + [f"D_{c}" for c in by_class])
    df = df[(df["date"] >= START_DATE_MAIN) & (df["date"] <= END_DATE_MAIN)]
    # subsec:data prints the capacity-bound size from a count file and the
    # foreign size as a macro.
    printed = {
        "capacity_bound": parse_num((ROOT / "tables" / "obs_count.tex").read_text()),
        "foreign": parse_num(m["sampForeignObs"]),
    }
    for cls, rows in by_class.items():
        flow = df[df[f"D_{cls}"].fillna(0) != 0]
        with_return = int(flow["R_t1"].notna().sum())
        assert_true(with_return == rows,
                    claim=f"the union stacking holds the {cls} panel's stock-days "
                          f"that carry a next-day return (subsec:reversal_test)",
                    detail=f"{rows} stacked rows against {with_return} such stock-days")
        assert_true(printed[cls] == len(flow),
                    claim=f"the {cls} panel size subsec:data prints counts every "
                          f"nonzero-flow stock-day of the window",
                    detail=f"printed {printed[cls]:.0f} against {len(flow)}")
    assert_close(m["revUnionRows"], sum(by_class.values()), tol=0.5,
                 label="revUnionRows")
    # Common support: both classes' nonzero-flow stock-days with a next-day
    # return.
    both = df[(df["D_capacity_bound"].fillna(0) != 0) & (df["D_foreign"].fillna(0) != 0)]
    common = int(both["R_t1"].notna().sum())
    assert_true(2 * common == ri["common"]["n_rows"],
                claim="common support is both classes' nonzero-flow stock-days "
                      "with a next-day return (subsec:reversal_test)",
                detail=f"{common} such stock-days against "
                       f"{ri['common']['n_rows']} stacked rows")
    assert_close(m["revCommonStockDays"], common, tol=0.5, label="revCommonStockDays")


def check_hml_floors_do_not_bind(m: dict[str, str]) -> None:
    """Covers: the two non-degeneracy floors of code/data/construct_hml_factor.py,
    whose comments say they do not bind on the sample window.

    A floor that binds drops its formation year, or its day, out of the daily
    series, so a complete series over the window is what makes the comment a
    measurement instead of an assertion. The factor feeds the factor-neutrality alpha of
    subsec:appendix_j_factor, which is asserted separately; this check reads
    the series' coverage rather than its values.
    """
    hml = pd.read_csv(RESULTS / "hml_factor_daily.csv", parse_dates=["date"])
    panel = pd.read_parquet(RESULTS / "full_daily_data_all_investors.parquet",
                            columns=["date"])
    dates = pd.Index(panel["date"].unique()).sort_values()
    # The series carries a return, so it cannot reach the panel's final date.
    expected = dates[:-1]
    missing = expected.difference(pd.Index(hml["date"]))
    assert_true(
        missing.empty and not hml["HML"].isna().any(),
        claim="the HML tail and split floors leave every formation year and "
              "every trading day of the window in the factor series",
        detail=f"{len(missing)} of {len(expected)} dates absent, "
               f"{int(hml['HML'].isna().sum())} null returns")


def check_threshold_grid_summary_floors_do_not_bind(m: dict[str, str]) -> None:
    """Covers: the two summary-statistic floors of
    code/empirical/robustness/threshold_sensitivity.py, whose comments say they do
    not bind across the grid.

    Each floor suppresses its whole summary when it bites, so a grid point
    reporting every long-short and rank-IC statistic is one neither floor
    reached. Reading the grid derived from the results file rather than the
    points a comment names keeps the claim true of a grid someone extends.
    """
    import json

    grid = json.loads(
        (RESULTS / "threshold_sensitivity_results.json").read_text())["results"]
    summaries = ("ls_sharpe_S_MC", "ls_mean_bps_S_MC", "ls_tstat_S_MC",
                 "ls_sharpe_S_TV", "ls_mean_bps_S_TV", "ls_tstat_S_TV",
                 "ic_mean_S_MC", "ic_tstat_S_MC", "ic_mean_S_TV", "ic_tstat_S_TV")
    suppressed = {
        point["threshold_label"]: [k for k in summaries if point.get(k) is None]
        for point in grid
    }
    suppressed = {k: v for k, v in suppressed.items() if v}
    assert_true(
        not suppressed and len(grid) > 1,
        claim="the long-short and rank-IC day floors leave every grid point of "
              "the size sweep summarised (subsec:appendix_j_marketcap)",
        detail=f"{len(grid)} grid points; suppressed summaries: {suppressed}")
