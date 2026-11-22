"""``\\newcommand`` emission helpers for ``tables/numbers.tex``.

Each ``_emit_*(cmd)`` function takes the ``cmd(name, value)`` callback supplied
by ``generate_numbers_tex`` and emits its block of ``\\newcommand`` definitions.
Helpers are called in the order their macros appear in ``tables/numbers.tex``, so the
rendered file is byte-identical across refactors of this layer.

This module is the macro-emission counterpart to ``code/output/_macro_specs.py`` (which
holds the declarative ``(name, dotted_path, fmt)`` rows that ``_emit_from_spec``
walks). The body-fragment generators (table ``*_body.tex`` files) live in
``code/output/generate_table_bodies.py``; this module only writes into ``tables/numbers.tex``.
"""

import inspect
import math

import numpy as np
import pandas as pd

from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR
from shared_utils import (
    END_DATE_MAIN,
    END_DATE_PRE,
    MIN_STOCKS_PER_DAY_DEFAULT,
    START_DATE_MAIN,
    START_DATE_PRE,
    winsorize,
)

from output._formatters import (
    BPS_PER_RETURN,
    CRIT_1PCT,
    CRIT_5PCT,
    _emit_from_spec,
    _emit_yearly_subperiod_extrema,
    fmt_comma,
    fmt_sci,
    load_json,
)
from output._macro_specs import (
    EXT_FN_REST_MACROS,
    EXT_LS_MACROS,
    EXT_PP_MACROS,
    FOREIGN_FN_REST_MACROS,
    LIQ_STRAT_MACROS,
    MI_IH_MACROS,
    MULTICOLLINEARITY_MACROS,
    WIN_NW_MACROS,
)


def _emit_flow_persistence_macros(cmd):
    """Round-trip-execution vs net-accumulation decomposition
    (tab:directionality_kr in subsec:regime_assignment).

    The directionality ratio (net accumulation / gross flow, median across
    stocks) is the return-free structural discriminator: capacity-bound flow
    is ~2--3x more directional than foreign, on the analysis window and on the
    non-overlapping pre-period holdout.
    """
    fp_path = RESULTS_DIR / "flow_persistence_decomposition.json"
    if not fp_path.exists():
        raise FileNotFoundError(
            f"{fp_path} not found. Run flow_persistence_decomposition.py "
            f"to produce it."
        )
    fp = load_json(fp_path)
    krp = fp["korea_preperiod"]

    def d(blk):
        return blk["directionality_median"]

    # The medians themselves are read off tab:directionality_kr and carry no macro.
    cmd("persistMinDays", f"{fp['config']['min_days']}")



def _emit_directionality_frequency_macros(cmd):
    """Directionality ratio against a trading-frequency null
    (subsec:regime_assignment, subsec:appendix_directionality_frequency).

    Source is ``results/directionality_frequency_null.json``. The observed ratios there
    are the ones tab:directionality_kr prints, re-derived on the same stocks and days,
    so they are not re-emitted here; what is emitted is the sign-flip null each
    ratio is measured against, the excess over it, and the ordering the null
    delivers on its own.
    """
    dfn = load_json(RESULTS_DIR / "directionality_frequency_null.json")
    cmd("dirNullDraws", fmt_comma(dfn["config"]["n_draws"]))

    # The published window is quoted in full. The pre-period window enters
    # subsec:regime_assignment as its excess over the same null and nothing
    # else, so its null level and day counts are not emitted.
    for suffix, window in (("", "korea"), ("Pre", "korea_preperiod")):
        blk = dfn[window]
        published = not suffix
        for cls, key in (("Cap", "capacity_bound"), ("For", "foreign")):
            c = blk[key]
            # Nulls are quoted at the three decimals the observed ratios use,
            # so a reader can difference the two as displayed.
            # The capacity-bound null is the one sec:appendix_scaling sets
            # against the Gaussian null; the table prints the rest.
            if published and cls == "Cap":
                cmd(f"dirNull{cls}", f"{c['null_mean']:.3f}")
            cmd(f"dirExcess{cls}{suffix}", f"{c['excess']:.2f}")
        # The null ordering and the excess ordering are emitted; the observed
        # one is already printed by tab:directionality_kr, which rounds it to
        # the precision its two-significant-figure components support, and a
        # second macro for the same quantity at a different precision would
        # publish both. The excess ordering is the one the corroboration rests
        # on (subsec:regime_assignment), since it measures each class against
        # its own day count.
        if published:
            cmd("dirNullRatio", f"{blk['ordering']['null']:.1f}")
        cmd(f"dirExcessRatio{suffix}", f"{blk['ordering']['excess']:.1f}")

    # The randomisation p-value floors at 2/(n_draws + 1) when no draw reaches
    # the observed ratio, which is the case in every Korean cell, so one macro
    # carries all four and the prose states it as a bound. The decimal count is
    # set by the floor rather than fixed: at 10,000 draws the floor is 2e-4, and
    # a display that rounds it to three places prints a p-value of zero.
    p_values = {
        c["p_two_sided"]
        for window in ("korea", "korea_preperiod")
        for c in (dfn[window]["capacity_bound"], dfn[window]["foreign"])
    }
    if len(p_values) != 1:
        raise ValueError(f"cells disagree on the randomisation p-value: {p_values}")
    p_floor = p_values.pop()
    digits = max(3, -int(np.floor(np.log10(p_floor))))
    cmd("dirNullP", f"{p_floor:.{digits}f}")

    # The Gaussian null at the same day counts, the closed-form alternative the
    # sign flip replaces. Emitted for the capacity-bound cell, the one the
    # appendix contrasts.
    cmd("dirGaussianCap",
        f"{dfn['korea']['capacity_bound']['null_gaussian_median']:.3f}")

    # Taiwan (sec:taiwan, Panel C of tab:directionality_frequency). The table
    # prints the value-space cells of both classes; sec:taiwan quotes the SITC
    # cell's p-value, because it is the one cell off the resolution floor, and
    # both classes' share-space excess, the reading the Taiwanese disclosure
    # reports flow in. The Taiwan p-values stay out of the ``dirNullP`` floor
    # above, which asserts a single shared value and holds for the Korean cells
    # alone.
    tw, tws = dfn["taiwan"], dfn["taiwan_share_space"]
    cmd("dirNullTwSitcP",           f"{tw['institutional']['p_two_sided']:.3f}")
    cmd("dirNullTwSitcExcessShare", f"{tws['institutional']['excess']:.2f}")
    cmd("dirNullTwForExcessShare",  f"{tws['foreign']['excess']:.2f}")
    # Four decimals, because this cell's p sits at the randomisation floor the
    # draw count sets and three would print it as zero, which is the one value
    # a randomisation p-value cannot take.
    cmd("dirNullTwSitcPShare",      f"{tws['institutional']['p_two_sided']:.4f}")

def _emit_taiwan_macros(cmd):
    """The Taiwan panel and its return prediction (sec:taiwan).

    Two sources, one section. ``results/twn_return_prediction_results.json`` sizes the
    panel and carries the corporate-action sensitivity that the section
    quotes. ``results/twn_universe_coverage.json`` carries what the live listing
    registry omits.
    """
    twr = load_json(RESULTS_DIR / "twn_return_prediction_results.json")
    cmd("twnPanelRows",   fmt_comma(twr["panel"]["n_rows"]))
    cmd("twnPanelStocks", fmt_comma(twr["panel"]["n_stocks"]))
    # The Taiwanese window is read off that panel's own dates, because the two
    # markets' panels are built separately and need not span the same years.
    cmd("twnYears", f"{twr['panel']['start_date'][:4]}--{twr['panel']['end_date'][:4]}")

    cmd("twnCorpActionDrift",
        f"{twr['corporate_action_sensitivity']['max_abs_drift']:.2f}")

    # The contrast of eq:reversal_contrast on the Taiwan panel, univariate, on
    # both stackings (code/empirical/taiwan/twn_contrast.py), in basis points as the Korean ones are.
    tc = load_json(RESULTS_DIR / "twn_contrast_results.json")
    for sample, tag in (("common", "Common"), ("union", "Union")):
        uni = tc[sample]["fama_macbeth_univariate"]
        for field, half in (("advantage_capacity", "CapEdge"),
                            ("advantage_foreign", "ForEdge"), ("reversal", "Rev")):
            cmd(f"twn{tag}{half}", f"{uni[field]['mean'] * 1e4:.2f}")
            cmd(f"twn{tag}{'' if half == 'Rev' else half}T", f"{uni[field]['t']:.2f}")

    # What the live listing registry costs. The universe is drawn from it, so a
    # delisted name it no longer lists cannot reach the panel however many
    # sessions it traded; the vendor's own delisting record puts a number on
    # that, and sec:taiwan states it as the panel's survivorship exposure.
    cov = load_json(RESULTS_DIR / "twn_universe_coverage.json")
    cmd("twnSurvivorStocks", fmt_comma(len(cov["survivorship_missing_stocks"])))
    cmd("twnSurvivorSessions", fmt_comma(cov["survivorship_missing_sessions"]))


def _emit_vwap_formation_macros(cmd):
    """VWAP formation-price robustness (subsec:vwap_formation).

    Matched- and mismatched-filter horse-race t-statistics for the two Korean
    cells under the close-formation market cap and under the VWAP-formation cap
    M^vwap = M x (VWAP/close). The cap denominator is the only change; the
    return is the standard close-to-close R_{t+1}. The reversal is unchanged
    across the two formation prices, ruling out a stale-closing-print artifact
    in the signal denominator.
    """
    path = RESULTS_DIR / "vwap_formation_robustness_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run vwap_formation_robustness.py to produce it."
        )
    vf = load_json(path)
    cells = {"Cap": vf["korea"]["capacity_bound"], "For": vf["korea"]["foreign"]}
    for tag, c in cells.items():
        cmd(f"vwap{tag}MatchedClose", f"{c['close']['hr_matched_t']:.2f}")
        cmd(f"vwap{tag}MatchedVwap", f"{c['vwap']['hr_matched_t']:.2f}")
        cmd(f"vwap{tag}MismatchClose", f"{c['close']['hr_mismatched_t']:.2f}")
        cmd(f"vwap{tag}MismatchVwap", f"{c['vwap']['hr_mismatched_t']:.2f}")


def _emit_fm_and_returns_macros(cmd):
    """Return prediction (subsec:return_prediction inline /
    tab:skip_day horizon profile), and the collinearity of the two filters
    (subsec:return_prediction)."""
    # Next-day (R_t1) horse-race t-stats of the two filters (qf_paper.tex
    # subsec:return_prediction). The longer-horizon decay profile lives in the
    # hz* macros (tab:skip_day), emitted by the horizon-analysis
    # block.
    rp = load_json(RESULTS_DIR / "return_prediction_results.json")
    d = rp["R_t1"]
    cmd("rpHRMCtone", f"{d['horse_race']['t_stat_S_MC']:.2f}")
    cmd("rpHRTVtone", f"{d['horse_race']['t_stat_S_TV']:.2f}")

    mc = load_json(RESULTS_DIR / "multicollinearity_test_results.json")
    _emit_from_spec(cmd, mc, MULTICOLLINEARITY_MACROS)


def _emit_extended_robustness_and_multi_investor_macros(cmd):
    """Out-of-sample, long-short, and factor-neutrality
    (sec:appendix_robustness_battery tables), investor heterogeneity
    (main tab:investor_heterogeneity), horizon analysis
    (tab:skip_day horizon profile)."""
    ext = load_json(RESULTS_DIR / "extended_robustness_results.json")
    _emit_from_spec(cmd, ext, EXT_PP_MACROS)
    _emit_yearly_subperiod_extrema(cmd, ext["yearly_subperiods"], prefix="ry",
                                   axes=("MC",))
    _emit_from_spec(cmd, ext, EXT_LS_MACROS)
    cmd("fnAlpha", f"{ext['factor_neutrality']['alpha']['coef']*BPS_PER_RETURN:.2f}")
    _emit_from_spec(cmd, ext, EXT_FN_REST_MACROS)

    mi = load_json(RESULTS_DIR / "multi_investor_robustness_results.json")
    # The subsec:appendix_aggregate_alignment paragraph quotes the \ihAggregate*
    # macros (MI_IH_MACROS).
    _emit_from_spec(cmd, mi, MI_IH_MACROS)


def _emit_liquidity_stratified_macros(cmd):
    """Return prediction inside market-cap quintiles.

    Source is ``results/liquidity_stratified_results.json``, backing
    sec:appendix_liquidity_stratified and the main-paper sentences that cite
    it on the size axis.
    """
    path = RESULTS_DIR / "liquidity_stratified_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run liquidity_stratified_return_prediction.py "
            f"to produce it."
        )
    liq = load_json(path)
    _emit_from_spec(cmd, liq, LIQ_STRAT_MACROS)
    # The day count reads the capacity-bound null in the smallest quintile as
    # a power boundary.
    cmd("liqCapSizeQOneDays",
        fmt_comma(liq["by_size"]["capacity_bound"]["Q1"]["univariate_MC"]["n_days"]))
    # Top-quintile share of each sample's stock-days, the size-boundary
    # paragraph of subsec:limitations placing the common-support stacking on
    # the axis the quintiles above cut. Stored as fractions.
    for name, sample in (("liqCommonSizeQFivePct", "common_support"),
                         ("liqForSizeQFivePct",    "foreign_nonzero")):
        cmd(name, f"{100 * liq['size_composition'][sample]['share']['Q5']:.1f}")


def _emit_robustness_inline_and_foreign_macros(cmd):
    """Winsorized + Newey-West inline robustness, plus foreign extended robustness."""
    ext = load_json(RESULTS_DIR / "extended_robustness_results.json")
    _emit_from_spec(cmd, ext, WIN_NW_MACROS)

    fext_path = RESULTS_DIR / "extended_robustness_foreign_results.json"
    if not fext_path.exists():
        raise FileNotFoundError(
            f"{fext_path} not found. Run robustness_tests_extended.py "
            f"--investor foreign to produce it."
        )
    fext = load_json(fext_path)
    # The foreign column of tab:robustness_summary quotes the horse race
    # throughout, because the claim on each row is the foreign matched filter's
    # advantage over the mismatched one rather than either filter's standalone
    # loading.
    fpp = fext["pre_period_oos"]
    for sig in ["MC", "TV"]:
        cmd(f"ffPP{sig}hrt", f"{fpp['horse_race'][sig]['t_stat']:.2f}")

    _emit_yearly_subperiod_extrema(cmd, fext["yearly_subperiods"], prefix="fry",
                                   axes=("TV",), n_years=False)

    # Both arms of the winsorised horse race, unlike the rows around it. The
    # mismatched arm is where the two classes part: subsec:return_prediction
    # claims a negative mismatched loading in both, and trimming the foreign
    # cross-section removes the foreign one rather than preserving it, so the
    # row that establishes the qualifier has to print the coefficient it
    # qualifies.
    fout = fext["outlier_robustness"]
    for regime, prefix in [("baseline", "fBase"), ("winsorized", "fWin")]:
        for sig in ["MC", "TV"]:
            cmd(f"f{prefix}{sig}hrt",
                f"{fout[regime]['horse_race'][sig]['t_stat']:.2f}")
    cmd("fNwTenTVt", f"{fext['alternative_se']['TV']['nw10_t']:.2f}")
    fls = fext["long_short"]["S_TV"]
    cmd("fLsTVbps", f"{fls['mean_bps']:.2f}")
    cmd("fLsTVt", f"{fls['t_stat']:.2f}")
    cmd("fFnAlpha", f"{fext['factor_neutrality']['alpha']['coef']*BPS_PER_RETURN:.2f}")
    _emit_from_spec(cmd, fext, FOREIGN_FN_REST_MACROS)


def _emit_turnover_descriptive(cmd):
    """Turnover mean/std (%) over the whole stock-day panel.

    The sentence these two feed (sec:appendix_data) says turnover varies
    widely across the panel, so the population is the panel, not the
    subset a capacity-bound institution trades. Restricting to nonzero capacity-bound flow -- which is the subset
    generate_count_files uses -- printed that class's moments under the
    panel's noun.
    """
    panel = pd.read_parquet(DAILY_DATA_ALL_INVESTORS, columns=["turnover"])
    turn_arr = panel["turnover"].to_numpy()
    cmd("descTurnMean", f"{turn_arr.mean() * 100:.1f}")
    cmd("descTurnStd", f"{turn_arr.std() * 100:.1f}")


def _emit_reversal_interaction_macros(cmd):
    """Stacked class-by-normalisation interaction test (subsec:reversal_test).

    Source is ``results/reversal_interaction_results.json``. Coefficients are daily
    returns and are emitted in basis points per standard deviation of the normalized flow,
    matching tab:reversal_interaction. The common-support stacking is the
    primary identification sample and the union stacking the sample extension,
    so only the statistics the prose of subsec:reversal_test names are emitted
    here; the full four-column grid lives in the table body. The two
    ``revPre*`` families are the same two stackings on the disjoint pre-period
    window, which subsec:contrast_halves reports as the out-of-sample row.
    """
    ri = load_json(RESULTS_DIR / "reversal_interaction_results.json")
    common, union, zero = ri["common"], ri["union"], ri["zero_included"]

    # Common support intersects the two classes' nonzero-flow stock-days, so
    # both classes must contribute the identical row count. The prose calls
    # that shared count the sample, so a drift would silently misreport it.
    per_class = set(common["n_rows_by_class"].values())
    if len(per_class) != 1:
        raise ValueError(
            f"common-support stacking is not balanced across classes: "
            f"{common['n_rows_by_class']}"
        )
    cmd("revCommonStockDays", fmt_comma(per_class.pop()))

    # The univariate arm is the primary specification (subsec:return_prediction)
    # and the only one the prose quotes; the joint arm, the conditioning check,
    # is read off tab:reversal_interaction. The per-class advantages carry their
    # t where a table of the body prints them from these macros, which
    # tab:contrast_halves does for common support and the zero-included
    # stacking. The union contrast and advantages are quoted as values alone
    # (subsec:appendix_zero_flow, subsec:skip_day) and their t is read off
    # tab:reversal_interaction.
    for stacking, res in (("Common", common), ("Union", union), ("Zero", zero)):
        fm = res["fama_macbeth_univariate"]
        cmd(f"rev{stacking}Rev", f"{fm['reversal']['mean'] * BPS_PER_RETURN:.2f}")
        if stacking != "Union":
            cmd(f"rev{stacking}T", f"{fm['reversal']['t']:.2f}")
        # Each class's matched-filter advantage over its mismatched filter, the
        # two quantities eq:reversal_contrast differences.
        for cls, key in (("Cap", "advantage_capacity"), ("For", "advantage_foreign")):
            adv = fm[key]
            cmd(f"rev{stacking}{cls}Edge", f"{adv['mean'] * BPS_PER_RETURN:.2f}")
            if stacking != "Union":
                cmd(f"rev{stacking}{cls}EdgeT", f"{adv['t']:.2f}")

    cmd("revCommonClusterP", fmt_sci(common["pooled_cluster_date"]["reversal"]["p_value"]))
    cmd("revUnionRows", fmt_comma(union["n_rows"]))
    cmd("revUnionClusterT",
        f"{union['pooled_cluster_date']['reversal']['t_stat']:.2f}")

    # The same two stackings on the pre-period panel, which subsec:contrast_halves
    # reads as the out-of-sample row of tab:contrast_halves. Only the univariate
    # arm is emitted, matching the published rows of that table.
    pre_common, pre_union = ri["pre_common"], ri["pre_union"]
    for res, window in ((pre_common, "pre_common"), (pre_union, "pre_union")):
        if res["window"] != "pre_period":
            raise ValueError(f"{window} arm is not stamped as the pre-period window: "
                             f"{res['window']}")
    # Disjointness is what makes the arm out of sample, so it is checked here
    # rather than assumed from the file the panel was read out of.
    if pre_common["date_last"] >= common["date_first"]:
        raise ValueError(
            f"pre-period window ends at {pre_common['date_last']}, on or after the "
            f"published window's first date {common['date_first']}"
        )
    cmd("revPreCommonStockDays", fmt_comma(set(pre_common["n_rows_by_class"].values()).pop()))
    for stacking, res in (("PreCommon", pre_common), ("PreUnion", pre_union)):
        fm = res["fama_macbeth_univariate"]
        cmd(f"rev{stacking}Rev", f"{fm['reversal']['mean'] * BPS_PER_RETURN:.2f}")
        cmd(f"rev{stacking}T", f"{fm['reversal']['t']:.2f}")
        # The advantage for foreign flow is emitted on both stackings and the
        # advantage for capacity-bound flow on common support alone, because
        # the two carry claims of different shape on this window.
        # subsec:contrast_halves establishes that the advantage for
        # capacity-bound flow reproduces, which one stacking does, and that
        # the advantage for foreign flow reaches the level on neither, which
        # is a negative claim and owes both.
        classes = [("For", "advantage_foreign")]
        if stacking == "PreCommon":
            classes.insert(0, ("Cap", "advantage_capacity"))
        for cls, key in classes:
            adv = fm[key]
            cmd(f"rev{stacking}{cls}Edge", f"{adv['mean'] * BPS_PER_RETURN:.2f}")
            cmd(f"rev{stacking}{cls}EdgeT", f"{adv['t']:.2f}")

    # The same two stackings on the return over [t+1, t+2], which
    # subsec:skip_day reads as the end of the reversal. The contrast is emitted
    # on both stackings, because its absence is a negative claim and owes both.
    # The advantage for capacity-bound flow is emitted on common support, where the prose
    # compares it with the next day's, and the t of the advantage for foreign flow on both, where
    # the prose says it turns positive.
    for stacking, key in (("SkipCommon", "common_second_day"), ("SkipUnion", "union_second_day")):
        res = ri[key]
        if res["return_col"] == ri["config"]["return_horizon"]:
            raise ValueError(f"{key} is estimated on the next-day return, not the second day's")
        fm = res["fama_macbeth_univariate"]
        cmd(f"rev{stacking}Rev", f"{fm['reversal']['mean'] * BPS_PER_RETURN:.2f}")
        cmd(f"rev{stacking}T", f"{fm['reversal']['t']:.2f}")
        cmd(f"rev{stacking}ForEdgeT", f"{fm['advantage_foreign']['t']:.2f}")
        if stacking == "SkipCommon":
            for cls, pack in (("Cap", "advantage_capacity"), ("For", "advantage_foreign")):
                cmd(f"rev{stacking}{cls}Edge", f"{fm[pack]['mean'] * BPS_PER_RETURN:.2f}")
            cmd(f"rev{stacking}CapEdgeT", f"{fm['advantage_capacity']['t']:.2f}")

    # Rows are counted after the per-day cell drops, which is what the
    # regressions of subsec:appendix_zero_flow actually fit.
    cmd("revZeroRows", fmt_comma(zero["n_rows"]))


def _emit_investment_trust_inclusion_macros(cmd):
    """The common-support contrast with the investment-trust sub-type in the
    capacity-bound class (subsec:appendix_aggregate_alignment).

    Source is ``results/investment_trust_inclusion.json``, in the units and the
    univariate arm of the ``revCommon*`` family it is printed beside. The
    stock-day count is emitted because the advantage for foreign flow moves with the common
    support, which the added flow widens.
    """
    it = load_json(RESULTS_DIR / "investment_trust_inclusion.json")["common"]
    per_class = set(it["n_rows_by_class"].values())
    if len(per_class) != 1:
        raise ValueError(f"common-support stacking is not balanced across classes: "
                         f"{it['n_rows_by_class']}")
    cmd("revTrustStockDays", fmt_comma(per_class.pop()))
    fm = it["fama_macbeth_univariate"]
    cmd("revTrustRev", f"{fm['reversal']['mean'] * BPS_PER_RETURN:.2f}")
    cmd("revTrustT", f"{fm['reversal']['t']:.2f}")
    for cls, key in (("Cap", "advantage_capacity"), ("For", "advantage_foreign")):
        cmd(f"revTrust{cls}Edge", f"{fm[key]['mean'] * BPS_PER_RETURN:.2f}")
        cmd(f"revTrust{cls}EdgeT", f"{fm[key]['t']:.2f}")


def _emit_flow_continuation_macros(cmd):
    """Flow persistence and the pressure it would exert (subsec:appendix_flow_continuation).

    Source is ``results/flow_continuation_results.json``. The participation-rate
    pressure measure is the one the body reports, because execution-cost
    models make impact a function of it, and the capital-scaled arm enters only
    as the cross-contrast and its bootstrap z, which show that the rejection
    does not turn on the choice.

    Continuation coefficients are slopes of one standardised signal on another
    and are dimensionless, so they carry no unit suffix. The cross-contrast is
    already in basis points in the JSON, because the return loadings entering
    it are.
    """
    fc = load_json(RESULTS_DIR / "flow_continuation_results.json")
    cmd("propBootDraws", fmt_comma(fc["config"]["n_bootstrap"]))

    for cls, tag in (("capacity_bound", "Cap"), ("foreign", "For")):
        block = fc["classes"][cls]
        cont = block["continuation"]["all"]
        cmd(f"cont{tag}MC", f"{cont['mc']['mean']:+.3f}")
        cmd(f"cont{tag}TV", f"{cont['tv']['mean']:+.3f}")
        cmd(f"cont{tag}MCT", f"{cont['mc']['t']:+.2f}")
        cmd(f"cont{tag}TVT", f"{cont['tv']['t']:+.2f}")

        press = block["pressure"]["participation"]
        prop = press["proportionality"]
        cmd(f"prop{tag}Cross", f"{prop['cross_contrast_bps']:+.2f}")
        cmd(f"prop{tag}CrossLo", f"{prop['ci_lower_bps']:+.2f}")
        cmd(f"prop{tag}CrossHi", f"{prop['ci_upper_bps']:+.2f}")
        cmd(f"prop{tag}CrossZ", f"{prop['boot_z']:+.2f}")
        cmd(f"prop{tag}QMC", f"{press['pressure']['mc']['mean']:+.3f}")
        cmd(f"prop{tag}QTV", f"{press['pressure']['tv']['mean']:+.3f}")
        cmd(f"prop{tag}QMCT", f"{press['pressure']['mc']['t']:+.2f}")
        cmd(f"prop{tag}QTVT", f"{press['pressure']['tv']['t']:+.2f}")

        cap_press = block["pressure"]["capitalisation"]
        cmd(f"prop{tag}CrossCap",
            f"{cap_press['proportionality']['cross_contrast_bps']:+.2f}")
        cmd(f"prop{tag}CrossCapZ", f"{cap_press['proportionality']['boot_z']:+.2f}")


def _emit_free_exponent_macros(cmd):
    """Free normalisation exponent theta (subsec:free_exponent).

    Source is ``results/free_exponent_profile.json``. Only the levels profiles carry a
    theta, because the rank profiles have no maximum: ranking
    $S^{MC}\\tau^{-\\theta}$ keeps an ordering that converges as $|\\theta|$
    grows, so the script records ``argmax_at_edge`` for them and this emitter
    takes their endpoint shares alone. ``assigned`` is whichever endpoint
    cor:regimes gives the class, so the same macro name carries the market-cap
    endpoint for one class and the traded-value endpoint for the other.
    """
    fe = load_json(RESULTS_DIR / "free_exponent_profile.json")
    grid = fe["config"]["theta_grid"]
    cmd("fexGridLo", f"{grid['lo']:+.1f}")
    cmd("fexGridHi", f"{grid['hi']:+.1f}")
    cmd("fexGridStep", f"{grid['step']:g}")
    cmd("fexBootDraws", fmt_comma(fe["config"]["n_bootstrap"]))

    def shares(node):
        assigned = node["assigned_endpoint"]
        other = "tv" if assigned == "mc" else "mc"
        return (node["endpoints"][assigned]["share_of_max"] * 100,
                node["endpoints"][other]["share_of_max"] * 100)

    # No best exponent or its interval is emitted: subsec:free_exponent reads
    # the gap between the two classes, because turnover's own return effect
    # enters both levels.
    for cls, tag in (("capacity_bound", "Cap"), ("foreign", "For")):
        own = fe["arms"]["own"][cls]["levels"]
        assigned, other = shares(own)
        cmd(f"fex{tag}AssignedPct", f"{assigned:.1f}")
        cmd(f"fex{tag}OtherPct", f"{other:.1f}")
        assigned, other = shares(fe["arms"]["own"][cls]["ranks"])
        cmd(f"fexRank{tag}AssignedPct", f"{assigned:.1f}")
        cmd(f"fexRank{tag}OtherPct", f"{other:.1f}")

    gap = fe["arms"]["own"]["gap_levels"]
    cmd("fexGap", f"{gap['gap']:+.2f}")
    cmd("fexGapLo", f"{gap['ci'][0]:+.2f}")
    cmd("fexGapHi", f"{gap['ci'][1]:+.2f}")

    # Common support is where subsec:reversal_test estimates the reversal, and
    # where the foreign profile flattens. Neither its theta nor its bounds are
    # emitted: the bootstrap interval reaches the grid end, so the point would
    # be read as located when the interval says it is not.
    com_for = fe["arms"]["common"]["foreign"]["levels"]
    assigned, other = shares(com_for)
    cmd("fexCommonForAssignedPct", f"{assigned:.1f}")
    cmd("fexCommonForOtherPct", f"{other:.1f}")
    cmd("fexForRhoMax", f"{fe['arms']['own']['foreign']['levels']['rho_max']:.4f}")
    cmd("fexCommonForRhoMax", f"{com_for['rho_max']:.4f}")
    com_gap = fe["arms"]["common"]["gap_levels"]
    cmd("fexCommonGap", f"{com_gap['gap']:+.2f}")
    cmd("fexCommonGapLo", f"{com_gap['ci'][0]:+.2f}")
    cmd("fexCommonGapHi", f"{com_gap['ci'][1]:+.2f}")


def _emit_size_restricted_macros(cmd):
    """Common-support perturbations inside the size distribution (subsec:size_restricted).

    Source is ``results/size_restricted_common.json``. The ``all`` arm reproduces the
    cells already carried by the ``rank*``, ``cem*`` and ``rev*`` families and
    takes no macros of its own, so nothing here can disagree with them; what is
    emitted is the two size arms, the paired contrast between them, and the
    count-matched control that says whether the difference is size or sample
    size. Advantages are in basis points per standard deviation of the normalized flow, as in
    subsec:reversal_test. The exponent block carries the volume endpoint's
    correlation rather than a theta, because the restricted profiles do not
    locate one.
    """
    sz = load_json(RESULTS_DIR / "size_restricted_common.json")
    cmd("sizSmallQuintiles", str(sz["config"]["n_small_quintiles"]))
    # The large arm is the small arm's complement, so the prose follows
    # N_SMALL_QUINTILES rather than carrying a second copy of the split.
    cmd("sizLargeQuintiles", str(len(sz["config"]["size_arms"]["large"])))

    for arm, tag in (("small", "Small"), ("large", "Large")):
        comp = sz["composition"][arm]
        cmd(f"siz{tag}StockDays", fmt_comma(comp["n_stock_days"]))
        cmd(f"siz{tag}PerDay", fmt_comma(int(comp["median_stocks_per_day"])))
    # One share only: the two arms partition common support, so the second is
    # the complement and would be a second name for the same fact.
    cmd("sizSmallSharePct",
        f"{sz['composition']['small']['share_of_common'] * 100:.1f}")

    # (source block, key inside it) for each of the table's three rows, so the
    # same three names index the size arms and the paired contrast alike.
    rows = (("Lv", "reversal", "zscore"), ("Rk", "reversal", "rank"),
            ("Cem", "matching", "matched"))
    halves = (("Cap", "advantage_capacity"), ("For", "advantage_foreign"),
              ("Rev", "reversal"))

    for arm, tag in (("small", "Small"), ("large", "Large")):
        for label, block, key in rows:
            node = sz[block][arm][key]["fama_macbeth_univariate"]
            for half, field in halves:
                cmd(f"siz{tag}{label}{half}", f"{node[field]['mean'] * 1e4:+.2f}")
                cmd(f"siz{tag}{label}{half}T", f"{node[field]['t']:+.2f}")

    res = sz["resample"]["advantage_foreign"]
    cmd("sizResampleDraws", str(sz["resample"]["n_draws"]))
    cmd("sizResampleMean", f"{res['mean_of_means'] * 1e4:+.2f}")
    cmd("sizResampleSd", f"{res['sd_of_means'] * 1e4:.2f}")
    # subsec:appendix_size_restricted says the bottom subsample lies this many
    # draw standard deviations below the draws' mean, so the magnitude prints.
    cmd("sizResampleZ", f"{abs(res['small_arm_z']):.2f}")
    cmd("sizResampleSigPct", f"{res['share_significant'] * 100:.1f}")

    # subsec:appendix_size_restricted says the profile locates the gap in
    # neither subsample, so it prints the lower end of each arm's gap interval.
    for arm, tag in (("small", "Small"), ("large", "Large")):
        cmd(f"siz{tag}GapLo",
            f"{sz['free_exponent'][arm]['gap_levels']['ci'][0]:+.2f}")


def _emit_rank_transform_macros(cmd):
    """Ordering claims re-estimated on within-day ranks (subsec:appendix_j_rank).

    Source is ``results/rank_transform_robustness.json``. Only the rank arm is emitted:
    its level counterparts carry macros (``\\rpHRMCtone``, the ``\\rev*``
    family) or are read off tab:investor_heterogeneity, and the full
    level-versus-rank grid lives in the table body. Advantages are daily returns and are emitted
    in basis points per standard deviation of the normalized flow, matching
    subsec:reversal_test.
    """
    rk = load_json(RESULTS_DIR / "rank_transform_robustness.json")

    # Individual flow appears in the table only, so it takes no macros.
    for cls, tag in (("capacity_bound", "Cap"), ("foreign", "For")):
        for spec, suffix in (("univariate_MC", "MCt"), ("univariate_TV", "TVt"),
                             ("horse_race_MC", "HRMCt"), ("horse_race_TV", "HRTVt")):
            cmd(f"rank{tag}{suffix}",
                f"{rk['per_class'][cls]['rank'][spec]['t_stat']:.2f}")

    for stacking in ("common", "union"):
        fm = rk["stacked"][stacking]["rank"]["fama_macbeth_univariate"]
        tag = stacking.capitalize()
        cmd(f"rank{tag}Rev", f"{fm['reversal']['mean'] * BPS_PER_RETURN:.2f}")
        cmd(f"rank{tag}T", f"{fm['reversal']['t']:.2f}")
        for cls, key in (("Cap", "advantage_capacity"), ("For", "advantage_foreign")):
            adv = fm[key]
            cmd(f"rank{tag}{cls}Edge", f"{adv['mean'] * BPS_PER_RETURN:.2f}")
            cmd(f"rank{tag}{cls}EdgeT", f"{adv['t']:.2f}")

    # How much of the turnover rescaling the transform keeps. The rank arm can
    # only be a lower bound on the advantage, because two filters with
    # identical ranks would separate by exactly zero; this pair of correlations
    # is what the prose reads that bound off.
    corr = rk["stacked"]["common"]
    for cls, tag in (("capacity_bound", "Cap"), ("foreign", "For")):
        for arm, suffix in (("zscore", "Z"), ("rank", "Rank")):
            cmd(f"rankCorr{tag}{suffix}",
                f"{corr[arm]['filter_correlation'][cls]:.2f}")


def _emit_lagged_adv_macros(cmd):
    """Reversal under a lagged traded-value denominator (sec:appendix_nonuniform_channels).

    Source is ``results/lagged_adv_normalisation_results.json``. Coefficients are daily
    returns and are emitted in basis points per standard deviation of the normalized flow,
    matching subsec:reversal_test. Every arm is estimated on the one sample
    that clears the longest window, so the ``same_day`` baseline emitted here
    is *not* the ``revCommon*`` figure of the full-sample specification and the
    two are not interchangeable in prose.

    The univariate arm is the primary specification (subsec:return_prediction),
    so every ``\\adv*`` macro reads it except the ``\\advJoint*`` pair, which
    carries the conditioning check.
    """
    adv = load_json(RESULTS_DIR / "lagged_adv_normalisation_results.json")
    cfg = adv["config"]
    common = adv["common"]

    windows = cfg["adv_windows"]
    headline = cfg["headline_window"]
    cmd("advWindow", str(headline))
    cmd("advWindowMin", str(min(windows)))
    cmd("advWindowMax", str(max(windows)))
    cmd("advSampleShare",
        f"{100 * cfg['n_rows_restricted'] / cfg['n_rows_panel']:.1f}")

    def rev(arm):
        return arm["fama_macbeth_univariate"]["reversal"]

    base, head = common["same_day"], common[f"adv_{headline}"]
    cmd("advBaseRev", f"{rev(base)['mean'] * BPS_PER_RETURN:.2f}")
    cmd("advBaseT", f"{rev(base)['t']:.2f}")
    cmd("advRev", f"{rev(head)['mean'] * BPS_PER_RETURN:.2f}")
    cmd("advT", f"{rev(head)['t']:.2f}")
    cmd("advClusterT", f"{head['pooled_cluster_date']['reversal']['t_stat']:.2f}")
    cmd("advJointRev",
        f"{head['fama_macbeth']['reversal']['mean'] * BPS_PER_RETURN:.2f}")
    cmd("advJointT", f"{head['fama_macbeth']['reversal']['t']:.2f}")
    # Each class's matched-filter advantage over its mismatched filter. The
    # prose reads the sign change off the pair, so each carries its own t.
    # Both arms are emitted: this row's advantages are only interpretable against
    # the same-day baseline on the same sample, per the warning above, and
    # subsec:contrast_halves counts the rows where the advantage for capacity-bound flow moves.
    for cls, key in (("Cap", "advantage_capacity"), ("For", "advantage_foreign")):
        edge = head["fama_macbeth_univariate"][key]
        cmd(f"adv{cls}Edge", f"{edge['mean'] * BPS_PER_RETURN:.2f}")
        cmd(f"adv{cls}EdgeT", f"{edge['t']:.2f}")
        # Baseline level only, matching the float paragraph beside it: the prose
        # reads the pair as a direction, and tab:contrast_halves carries the t.
        cmd(f"adv{cls}EdgeBase",
            f"{base['fama_macbeth_univariate'][key]['mean'] * BPS_PER_RETURN:.2f}")
    # The prose reads this gap as an upper bound on the shared component's
    # contribution, not as its value, because the lagged denominator also
    # swaps realised volume for its recent average.
    cmd("advGap", f"{(rev(base)['mean'] - rev(head)['mean']) * BPS_PER_RETURN:.2f}")

    # Range over the window sweep, the discharge for having chosen one window.
    sweep = [common[f"adv_{w}"] for w in windows]
    cmd("advSweepRevLo", f"{min(rev(a)['mean'] for a in sweep) * BPS_PER_RETURN:.2f}")
    cmd("advSweepRevHi", f"{max(rev(a)['mean'] for a in sweep) * BPS_PER_RETURN:.2f}")
    cmd("advSweepTLo", f"{min(rev(a)['t'] for a in sweep):.2f}")

    union_head = adv["union"][f"adv_{headline}"]
    cmd("advUnionRev", f"{rev(union_head)['mean'] * BPS_PER_RETURN:.2f}")
    cmd("advUnionT", f"{rev(union_head)['t']:.2f}")


def _emit_participation_matching_macros(cmd):
    """Reversal on matched net participation rates (subsec:appendix_participation_matching).

    Source is ``results/participation_matching_results.json``. Coefficients are daily
    returns and are emitted in basis points per standard deviation of the normalized flow,
    matching subsec:reversal_test. The ``full`` arm reproduces the published
    specification exactly, so the unmatched figures the prose reads against are
    the ``\\revCommon*`` family and take no macros of their own here.

    The univariate arm is the primary specification (subsec:return_prediction).
    Balance enters as the
    standardised class difference on the participation rate, the quantity the
    matching is there to equalise, and the two class means enter as geometric
    mean participation rates because the matching variable is their log.
    """
    pm = load_json(RESULTS_DIR / "participation_matching_results.json")
    common, union = pm["common"], pm["union"]
    grid, fine = common["quintile"], common["decile"]

    def rev(arm):
        return arm["fama_macbeth_univariate"]["reversal"]

    # The imbalance the design closes, and its direction. Exponentiating the
    # class mean of log |D|/V returns a participation rate, so the pair states
    # which class takes the larger share of the day's trading.
    gap = common["balance_full"]["logIntensity"]
    cmd("cemIntensityCap", f"{100 * math.exp(gap['capacity_mean']):.1f}")
    cmd("cemIntensityFor", f"{100 * math.exp(gap['foreign_mean']):.1f}")
    cmd("cemBalanceBar", f"{pm['config']['balance_bar']:.1f}")
    cmd("cemGapFull", f"{gap['std_diff']:.2f}")
    cmd("cemGapMatched", f"{grid['balance_matched']['logIntensity']['std_diff']:.2f}")
    cmd("cemGapDecile", f"{fine['balance_matched']['logIntensity']['std_diff']:.2f}")

    # The capacity filter's magnitude, which the cells pin through the identity
    # |S^MC| = |S^TV| tau rather than by conditioning on it. Its balance is what
    # says the design is neutral between the two filters in the coarsened cells
    # and not only in the identity.
    cmd("cemMagGapFull", f"{common['balance_full']['logCapMagnitude']['std_diff']:.2f}")
    cmd("cemMagGapMatched",
        f"{grid['balance_matched']['logCapMagnitude']['std_diff']:.2f}")
    cmd("cemMagGapDecile",
        f"{fine['balance_matched']['logCapMagnitude']['std_diff']:.2f}")

    # What the re-weighting costs in precision. The Kish count is the number of
    # unit-weight rows that carry the same precision, so its share of the raw
    # row count is what a raw count would overstate.
    for cls, tag in (("capacity_bound", "Cap"), ("foreign", "For")):
        ess = grid["effective_size"][cls]
        cmd(f"cemEss{tag}Pct", f"{100 * ess['effective_n'] / ess['n_rows']:.0f}")

    cmd("cemRev", f"{rev(grid['matched'])['mean'] * BPS_PER_RETURN:.2f}")
    cmd("cemT", f"{rev(grid['matched'])['t']:.2f}")
    cmd("cemOverlapRev", f"{rev(grid['overlap'])['mean'] * BPS_PER_RETURN:.2f}")
    cmd("cemOverlapT", f"{rev(grid['overlap'])['t']:.2f}")

    # Each class's matched-filter advantage on the re-balanced cells, which
    # Table 5 prints for the quintile grid.
    for cls, key in (("Cap", "advantage_capacity"), ("For", "advantage_foreign")):
        edge = grid["matched"]["fama_macbeth_univariate"][key]
        cmd(f"cem{cls}Edge", f"{edge['mean'] * BPS_PER_RETURN:.2f}")
        cmd(f"cem{cls}EdgeT", f"{edge['t']:.2f}")


def _emit_subtype_advantage_macros(cmd):
    """The matched-filter advantage for every disclosed class.

    Source is ``results/subtype_advantage_results.json``, backing
    subsec:subtype_dose_response and subsec:appendix_aggregate_alignment.
    Advantages are daily returns and are emitted in basis points per standard
    deviation of the normalized flow, matching subsec:reversal_test. The univariate arm is
    the primary specification (subsec:return_prediction).
    """
    dr = load_json(RESULTS_DIR / "subtype_advantage_results.json")
    classes = dr["classes"]

    cmd("doseClasses", str(len(classes)))

    # The class the prose describes by name without assigning it a regime.
    individual = classes["individual"]
    cmd("doseIndAdv",
        f"{individual['univariate']['advantage']['mean'] * BPS_PER_RETURN:.2f}")
    cmd("doseIndMC", f"{individual['univariate']['b_MC']['mean'] * BPS_PER_RETURN:.2f}")
    cmd("doseIndTV", f"{individual['univariate']['b_TV']['mean'] * BPS_PER_RETURN:.2f}")
    cmd("doseIndAdvT", f"{individual['univariate']['advantage']['t']:.2f}")
    cmd("doseBankT", f"{classes['bank']['univariate']['advantage']['t']:.2f}")

    # subsec:appendix_aggregate_alignment drops each half of the capacity-bound class
    # in turn, so both halves' advantages are emitted from the same file the
    # class-level advantages come from.
    for key, pref in (("pension", "Pen"), ("insurance", "Ins")):
        c = classes[key]
        cmd(f"dose{pref}Adv",
            f"{c['univariate']['advantage']['mean'] * BPS_PER_RETURN:.2f}")
        cmd(f"dose{pref}AdvT", f"{c['univariate']['advantage']['t']:.2f}")
        cmd(f"dose{pref}ObsApprox", f"{c['panel_size'] / 1e6:.2f}~million")


def _emit_sample_size_macros(cmd):
    """Day counts and per-investor nonzero-flow observation counts."""
    if not DAILY_DATA_ALL_INVESTORS.exists():
        raise FileNotFoundError(
            f"{DAILY_DATA_ALL_INVESTORS} not found. Run "
            f"full_daily_data_pipeline.py to produce it."
        )
    panel = pd.read_parquet(
        DAILY_DATA_ALL_INVESTORS,
        columns=["date", "D_capacity_bound", "D_foreign", "D_individual"],
    )
    panel_total_rows = len(panel)
    cap_nonzero = panel["D_capacity_bound"] != 0
    total_days = panel.loc[cap_nonzero, "date"].nunique()
    cmd("sampTotalDays", fmt_comma(total_days))
    cmd("sampTotalDaysMinusOne", fmt_comma(total_days - 1))
    cmd("sampFactorDays", fmt_comma(total_days - 1))
    # Two decimals. subsec:appendix_panel_filters prints the panel total and
    # subsec:data the foreign estimation sample, and at one decimal the two
    # rendered as one number quoted twice rather than two populations 22,169
    # stock-days apart.
    cmd("panelTotalObsApprox", f"{panel_total_rows / 1e6:.2f}~million")

    inst_obs = int(cap_nonzero.sum())
    foreign_obs = int((panel["D_foreign"] != 0).sum())
    cmd("sampForeignObs", fmt_comma(foreign_obs))
    # Each class's share of the panel it is drawn from.
    # subsec:appendix_panel_filters quotes both so that the panel total and the
    # two estimation samples cannot be read as the same number: the capacity
    # class is absent from two-thirds of the panel while foreign flow covers
    # nearly all of it.
    cmd("sampInstPanelPct",    f"{100.0 * inst_obs / panel_total_rows:.1f}")
    cmd("sampForeignPanelPct", f"{100.0 * foreign_obs / panel_total_rows:.1f}")


def _emit_foreign_definition_macros(cmd):
    """Size of the 기타외국인 leg the foreign definition excludes
    (sec:appendix_data).

    Source is ``results/foreign_definition_share.json``. sec:appendix_data quotes the
    gross share to size the leg and the sign-flip share to say what including
    it would do to the signal, which is the quantity the contrast is estimated
    on rather than the flow it is built from.

    The last two macros serve the nonzero-flow restriction rather than the
    class boundary. subsec:appendix_panel_filters reads an exact zero as a
    record of non-participation and prints, for the foreign and the
    capacity-bound class, the share of that class's zero stock-days on which
    the same script finds no trade at all.
    """
    fd = load_json(RESULTS_DIR / "foreign_definition_share.json")["panel"]
    cmd("foreignOtherGrossPct", f"{fd['gross_other_share_pct']:.1f}")
    cmd("foreignOtherFlipPct",  f"{fd['sign_flip_share_pct']:.1f}")
    cmd("foreignZeroAbsentPct",
        f"{100.0 * (fd['n_zero_net'] - fd['n_zero_net_with_gross']) / fd['n_zero_net']:.1f}")
    cap = fd["capacity_bound"]
    cmd("capZeroAbsentPct",
        f"{100.0 * (cap['n_zero_net'] - cap['n_zero_net_with_gross']) / cap['n_zero_net']:.1f}")


def _emit_supplementary_macros(cmd):
    """Foreign-flow return regressions across KOSPI 200 membership."""
    k200_path = RESULTS_DIR / "foreign_kospi200_split_results.json"
    if not k200_path.exists():
        raise FileNotFoundError(
            f"{k200_path} not found. Run foreign_kospi200_split.py "
            f"to produce it."
        )
    k200 = load_json(k200_path)
    # The three primary subsets' t-statistics, and the size control's, are read
    # off tab:foreign_kospi200_split, so the prose carries their sample sizes alone.
    for key, prefix in [("full", "All"), ("kospi200", "In"), ("non_kospi200", "Out")]:
        cmd(f"retForeignKospi{prefix}Obs", fmt_comma(k200[key]['panel_size']))
    cmd("retForeignKospiInDailyAvg",
        f"{k200['kospi200']['horse_race']['avg_n_stocks']:.0f}")
    cmd("retForeignKospiOutDailyAvg",
        f"{k200['non_kospi200']['horse_race']['avg_n_stocks']:.0f}")


# ---------------------------------------------------------------------------
# Manual-region macros
# ---------------------------------------------------------------------------

def _emit_manual_section_macros(cmd):
    """Emit the prose-body macros, so the running text types no number.

    Numerical primitives in the manuscript body (the
    log-flow shift constant ``c``, the cross-section size filter, the
    sample-window and pre-period year ranges (and the pre-period's full ISO
    start/end bounds), the turnover-
    volatility regime percentiles, the winsorisation cut points, the
    size-threshold grid, the yearly-stability
    year lists, and the IH sub-type row shares) are emitted here so that
    the manuscript inherits the single source of truth from
    ``shared_utils.config`` (and the result JSONs), rather than carrying
    duplicated literals.
    """
    # --- Cross-section size filter (subsec:appendix_panel_filters) ---
    cmd("minStocksPerDay", f"{MIN_STOCKS_PER_DAY_DEFAULT}")

    # --- Sample-window year ranges (data-description prose). Derived from the
    #     period bounds in shared_utils.config so the body text, the analysis
    #     window, and the pre-period holdout all read one source of truth. ---
    cmd("sampleYears", f"{START_DATE_MAIN[:4]}--{END_DATE_MAIN[:4]}")
    cmd("preYears",    f"{START_DATE_PRE[:4]}--{END_DATE_PRE[:4]}")

    # --- Newey-West lag conventions (sec:appendix_robustness_battery robustness checks) ---
    cmd("nwLagsFive", "5")
    cmd("nwLagsTen",  "10")

    # --- Winsorisation cut points (subsec:appendix_j_outlier prose).
    #     Sourced from the shared_utils.data.winsorize signature defaults. ---
    winsor_lower = inspect.signature(winsorize).parameters["lower"].default
    winsor_upper = inspect.signature(winsorize).parameters["upper"].default
    cmd("winsorLowerPct", f"{round(winsor_lower * 100)}")
    cmd("winsorUpperPct", f"{round(winsor_upper * 100)}")

    # --- Market-cap threshold grid (subsec:appendix_j_marketcap).
    #     Read off the sweep's own results file, so the grid quoted in prose
    #     cannot drift from the grid the sweep ran. The top column's coverage
    #     is emitted alongside it, because the prose's claim is about how much
    #     of the universe the grid removes, not about the cutoffs themselves. ---
    ts_path = RESULTS_DIR / "threshold_sensitivity_results.json"
    if ts_path.exists():
        ts_rows = sorted(load_json(ts_path)["results"], key=lambda r: r["threshold"])
        bns = [f"{r['threshold'] / 1e9:g}" for r in ts_rows]
        cmd("threshGridKRW", ", ".join(bns[:-1]) + f", and {bns[-1]}")
        base, top = ts_rows[0], ts_rows[-1]
        cmd("threshTopRemovedPct", f"{top['pct_stock_days_removed']:.0f}")
        cmd("threshTopTickers", str(int(top["n_tickers"])))
        cmd("threshTopMCt", f"{top['fm_hr_tstat_S_MC_t+1']:.2f}")
        cmd("threshTopTVt", f"{top['fm_hr_tstat_S_TV_t+1']:+.2f}")
        # The last threshold at which BOTH horse-race arms clear the 1% level,
        # the floor that the robustness summary table reports.
        held = [r for r in ts_rows
                if r["fm_hr_tstat_S_MC_t+1"] > CRIT_1PCT
                and r["fm_hr_tstat_S_TV_t+1"] < -CRIT_1PCT][-1]
        cmd("threshHeldRemovedPct", f"{held['pct_stock_days_removed']:.0f}")
        cmd("threshHeldMCt", f"{held['fm_hr_tstat_S_MC_t+1']:.2f}")
        # The matched filter alone survives past that point, so the appendix
        # reports the matched arm's last 1% threshold.
        mc_last = [r for r in ts_rows
                   if r["fm_hr_tstat_S_MC_t+1"] > CRIT_1PCT][-1]
        cmd("threshMCLastPct", f"{mc_last['pct_stock_days_removed']:.0f}")
        cmd("threshMCLastT", f"{mc_last['fm_hr_tstat_S_MC_t+1']:.2f}")
        # Coefficient against standard error at the top of the grid, which is
        # what separates an attenuating effect from a thinner cross-section.
        cmd("threshTopCoefRatio",
            f"{top['fm_hr_coef_S_MC_t+1'] / base['fm_hr_coef_S_MC_t+1']:.2f}")
        cmd("threshTopSeRatio",
            f"{top['fm_hr_se_S_MC_t+1'] / base['fm_hr_se_S_MC_t+1']:.2f}")
        # Long-short and rank-IC ranges over the whole grid. The sweep's other
        # two statistics reach the reader as these ranges rather than as the
        # per-threshold table the appendix used to print.
        for tag, col in (("MC", "ls_mean_bps_S_MC"), ("TV", "ls_mean_bps_S_TV")):
            bps = [r[col] for r in ts_rows]
            cmd(f"threshLsBps{tag}Min", f"{min(bps):.1f}")
            cmd(f"threshLsBps{tag}Max", f"{max(bps):.1f}")
        cmd("threshIcMCmaxT",
            f"{max(abs(r['ic_tstat_S_MC']) for r in ts_rows):.2f}")

    # --- Yearly-stability year lists (auto-generated from data) ---
    ext_path = RESULTS_DIR / "extended_robustness_results.json"
    if ext_path.exists():
        ext = load_json(ext_path)
        years = ext.get("yearly_subperiods", {})
        # The sign of a coefficient that is not significant is not reported,
        # so the years split on significance alone. A year significantly
        # positive falls in neither list, and check_extended_robustness_yearly
        # fails on it.
        sig_neg, insignif = [], []
        for y in sorted(years):
            tv = years[y].get("horse_race", {}).get("TV", {})
            t = tv.get("t_stat")
            if t is None:
                continue
            if t <= -CRIT_5PCT:
                sig_neg.append(y)
            elif abs(t) < CRIT_5PCT:
                insignif.append(y)
        cmd("rySignifNegYears", _format_year_list(sig_neg))
        cmd("ryInsignifYears",  _format_year_list(insignif))

def _format_year_list(years):
    """Format a list of year strings as English-style enumeration.

    Examples: [] → "none"; ["2021"] → "2021"; ["2021","2023"] → "2021 and 2023";
    ["2020","2022","2024"] → "2020, 2022, and 2024".
    """
    if not years:
        return "none"
    if len(years) == 1:
        return years[0]
    if len(years) == 2:
        return f"{years[0]} and {years[1]}"
    return ", ".join(years[:-1]) + f", and {years[-1]}"
