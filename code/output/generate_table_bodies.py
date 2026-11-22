#!/usr/bin/env python3
"""
Generate LaTeX numbers from JSON result files.

Produces two kinds of output in tables/:

1. tables/numbers.tex — \\newcommand macros for every key value in the paper.
   The paper \\input{tables/numbers} in the preamble and uses macros like
   \\retMCuniTone inside hardcoded table structures. This avoids
   \\input inside tabulars (which breaks with endfloat + \\multicolumn).

2. *_body.tex — table body fragments for tables that don't use
   \\multicolumn at the start (e.g. investor_heterogeneity).

Usage:
    cd code && PYTHONPATH=. python output/generate_table_bodies.py
"""

import csv

import numpy as np
import pandas as pd

from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR, TABLES_DIR
from shared_utils import (
    END_DATE_MAIN,
    END_DATE_PRE,
    START_DATE_MAIN,
    START_DATE_PRE,
)

# Body-fragment table generators use the formatters directly; the
# ``\newcommand`` emission layer (``_emit_*`` helpers + declarative
# ``*_MACROS`` specs) lives in ``_macro_emitters`` / ``_macro_specs`` and is
# only imported below for ``generate_numbers_tex``.
from output._formatters import (
    BPS_PER_RETURN,
    TRADING_DAYS_PER_YEAR,
    _bold_max,
    _p_from_t,
    _with_botrule,
    _wrap_negative_in_math,
    fmt_coef,
    fmt_coef_star,
    fmt_comma,
    fmt_r2_from_pct,
    fmt_sci,
    fmt_t,
    fmt_t_paren,
    load_json,
    sweep_stale_bodies,
    write_body,
)
from output._macro_emitters import (
    _emit_turnover_descriptive,
    _emit_reversal_interaction_macros,
    _emit_investment_trust_inclusion_macros,
    _emit_lagged_adv_macros,
    _emit_participation_matching_macros,
    _emit_rank_transform_macros,
    _emit_flow_continuation_macros,
    _emit_free_exponent_macros,
    _emit_size_restricted_macros,
    _emit_subtype_advantage_macros,
    _emit_vwap_formation_macros,
    _emit_sample_size_macros,
    _emit_foreign_definition_macros,
    _emit_taiwan_macros,
    _emit_extended_robustness_and_multi_investor_macros,
    _emit_directionality_frequency_macros,
    _emit_flow_persistence_macros,
    _emit_fm_and_returns_macros,
    _emit_liquidity_stratified_macros,
    _emit_manual_section_macros,
    _emit_robustness_inline_and_foreign_macros,
    _emit_supplementary_macros,
)


def generate_liquidity_stratified_body():
    """Generate body for the size-stratified return table (sec:appendix_liquidity_stratified).

    Five quintile rows. Every row carries both classes: each class's matched
    filter first, its mismatched filter second, then the number of trading
    days that cleared the Fama-MacBeth minimum for that class in that bin. The
    day counts are the column the reader needs to tell a null from an
    unmeasured cell, since capacity-bound flow is absent from most small
    cross-sections.
    """
    path = RESULTS_DIR / "liquidity_stratified_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run "
            f"liquidity_stratified_return_prediction.py to produce it."
        )
    data = load_json(path)
    matched = data["config"]["matched_filter"]

    def cell(bin_result, investor, arm):
        r = bin_result[f"univariate_{arm}"][f"S_{arm}_{investor}"]
        return fmt_t(r["t_stat"], r["p_value"])

    def panel_rows(axis):
        lines = []
        for q in sorted(data[axis]["capacity_bound"]):
            cols = []
            for investor, arm in matched.items():
                mis = "TV" if arm == "MC" else "MC"
                b = data[axis][investor][q]
                cols += [cell(b, investor, arm), cell(b, investor, mis),
                         f"{b[f'univariate_{arm}']['n_days']:,}".replace(",", "\\,")]
            lines.append(f"{q} & " + " & ".join(cols) + " \\\\")
        return lines

    lines = panel_rows("by_size")
    _with_botrule(lines)
    write_body("liquidity_stratified_body.tex", "\n".join(lines) + "\n")


def generate_foreign_kospi200_split_body():
    """Generate body for the KOSPI 200 membership-split table.

    Four rows (Full / KOSPI 200 / Non-KOSPI 200 / Non-KOSPI 200 largest
    quintile) × four columns: univariate t_TV, horse-race t_TV, horse-race
    t_MC, average daily N stocks.
    """
    path = RESULTS_DIR / "foreign_kospi200_split_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run foreign_kospi200_split.py to produce it."
        )
    data = load_json(path)

    rows = [
        ("Full foreign panel", "full"),
        ("KOSPI 200 subset", "kospi200"),
        ("Non-KOSPI 200 subset", "non_kospi200"),
        ("\\quad largest size quintile", "non_kospi200_largest"),
    ]

    lines = []
    for label, key in rows:
        d = data[key]
        tv_uni = d["univariate_TV"]
        hr = d["horse_race"]
        avg_n = hr["avg_n_stocks"]
        lines.append(
            f"{label} & "
            f"{fmt_t(tv_uni['t_stat'], tv_uni['p_value'])} & "
            f"{fmt_t(hr['signal_TV']['t_stat'], hr['signal_TV']['p_value'])} & "
            f"{fmt_t(hr['signal_MC']['t_stat'], hr['signal_MC']['p_value'])} & "
            f"{avg_n:,.0f} \\\\"
        )
    # Add botrule on last row
    _with_botrule(lines)

    write_body("foreign_kospi200_split_body.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: investor_heterogeneity (main tab:investor_heterogeneity)
# Source: results/multi_investor_robustness_results.json (return_predictions section)
# ---------------------------------------------------------------------------

def generate_investor_heterogeneity_body():
    """Generate body for the investor heterogeneity return prediction table."""
    data = load_json(RESULTS_DIR / "multi_investor_robustness_results.json")

    rp = data["return_predictions"]

    # Main-paper tab:investor_heterogeneity panels. Panel A is Capacity-Bound Institutions
    # (= pension + insurance), the matched-filter-relevant subset of the KRX
    # aggregate. The aggregate appears only inline in the subsec:appendix_aggregate_alignment
    # paragraph and in the sec:appendix_subinvestor_decomposition decomposition table.
    panels = [
        ("capacity_bound", "Capacity-bound institutions", "A"),
        ("foreign",        "Foreign investors",           "B"),
        ("individual",     "Individual investors",        "C"),
    ]

    # First panel header is in the paper (not here) to avoid
    # \multicolumn as first token with endfloat.
    lines = []
    for i, (inv_key, title, panel_letter) in enumerate(panels):
        d = rp[inv_key]
        mc_uni = d["univariate_MC"]
        tv_uni = d["univariate_TV"]
        hr_mc = d["horse_race_MC"]
        hr_tv = d["horse_race_TV"]
        hr_r2 = d["horse_race_R2"]

        if i > 0:
            lines.append(
                f"\\multicolumn{{4}}{{@{{}}l@{{}}}}{{\\textbf{{Panel {panel_letter}: {title}}}}} \\\\\\colrule"
            )

        # S^MC rows
        lines.append(
            f"$S^{{\\mathrm{{MC}}}}$ & {fmt_coef_star(mc_uni['mean'], mc_uni['p_value'])} "
            f"& --- & {fmt_coef_star(hr_mc['mean'], hr_mc['p_value'])} \\\\"
        )
        lines.append(
            f"& {fmt_t_paren(mc_uni['t_stat'])} & & {fmt_t_paren(hr_mc['t_stat'])} \\\\"
        )

        # S^TV rows
        lines.append(
            f"$S^{{\\mathrm{{TV}}}}$ & --- & {fmt_coef_star(tv_uni['mean'], tv_uni['p_value'])} "
            f"& {fmt_coef_star(hr_tv['mean'], hr_tv['p_value'])} \\\\"
        )
        lines.append(
            f"& & {fmt_t_paren(tv_uni['t_stat'])} & {fmt_t_paren(hr_tv['t_stat'])} \\\\"
        )

        # R² row (multi_investor JSON stores R² already in percentage units)
        mc_r2 = fmt_r2_from_pct(mc_uni["mean_R2"])
        tv_r2 = fmt_r2_from_pct(tv_uni["mean_R2"])
        hr_r2_str = fmt_r2_from_pct(hr_r2)
        lines.append(
            f"$R^2$ (\\%) & {mc_r2} & {tv_r2} & {hr_r2_str} \\\\\\colrule"
        )

    # Remove trailing \colrule from last panel, replace with \botrule
    lines[-1] = lines[-1].replace("\\\\\\colrule", "\\\\\\botrule")

    write_body("investor_heterogeneity_body.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: reversal_interaction (main tab:reversal_interaction, subsec:reversal_test)
# Source: results/reversal_interaction_results.json
# ---------------------------------------------------------------------------

# Row order of eq:reversal_interaction, paired with the LaTeX label the paper
# uses for each term. The two interaction rows carry the contrast that
# eq:reversal_contrast defines, so they are the last of the four slopes.
_REVERSAL_ROWS = [
    ("zMC", "$z(S^{\\mathrm{MC}})$"),
    ("zTV", "$z(S^{\\mathrm{TV}})$"),
    ("F_zMC", "$F \\times z(S^{\\mathrm{MC}})$"),
    ("F_zTV", "$F \\times z(S^{\\mathrm{TV}})$"),
]

# The three contrasts eq:reversal_contrast builds from the four slopes: each
# class's matched-minus-mismatched advantage, then their difference.
_REVERSAL_CONTRAST_ROWS = [
    ("advantage_capacity", "Advantage of $S^{\\mathrm{MC}}$ over $S^{\\mathrm{TV}}$, capacity-bound"),
    ("advantage_foreign", "Advantage of $S^{\\mathrm{MC}}$ over $S^{\\mathrm{TV}}$, foreign"),
    ("reversal", "Contrast ($b_4 - b_3$)"),
]

# Column order: the common-support stacking is the primary identification
# sample (subsec:reversal_test), the union stacking the sample extension.
_REVERSAL_SAMPLES = ("common", "union")

# Within each stacking, the univariate arm is the primary specification
# (subsec:return_prediction) and the joint arm the conditioning check.
_REVERSAL_ARMS = ("fama_macbeth_univariate", "fama_macbeth")


def generate_reversal_interaction_body():
    """Generate the body for the stacked class-by-normalisation interaction table.

    Four columns, one per (stacking, specification) pair, all Fama-MacBeth over
    a per-day coefficient series. Coefficients are daily returns and are
    reported in basis points per standard deviation of the normalized flow, both signals
    being day-by-class standardised in eq:reversal_interaction. The
    cluster-by-date estimates of the joint arm are quoted in the prose of
    subsec:reversal_test rather than given columns here, because the univariate
    contrast spans two regressions per day and so has no counterpart under a
    single fitted cross-sectional covariance.
    """
    data = load_json(RESULTS_DIR / "reversal_interaction_results.json")

    def cells(getter):
        """Apply ``getter(spec, n)`` across the four (stacking, specification) columns."""
        return [getter(data[sample][arm], data[sample][arm]["n_days"])
                for sample in _REVERSAL_SAMPLES for arm in _REVERSAL_ARMS]

    lines = []
    for key, label in _REVERSAL_ROWS:
        def coef_cell(spec, n, key=key):
            c = spec["coefficients"][key]
            return fmt_coef_star(c["mean"] * BPS_PER_RETURN,
                                 _p_from_t(c["t"], n), decimals=2)

        def t_cell(spec, n, key=key):
            return fmt_t_paren(spec["coefficients"][key]["t"])

        lines.append(f"{label} & " + " & ".join(cells(coef_cell)) + " \\\\")
        lines.append("& " + " & ".join(cells(t_cell)) + " \\\\")

    lines.append("\\colrule")
    for key, label in _REVERSAL_CONTRAST_ROWS:
        def contrast_coef_cell(spec, n, key=key):
            c = spec[key]
            return fmt_coef_star(c["mean"] * BPS_PER_RETURN,
                                 _p_from_t(c["t"], n), decimals=2)

        def contrast_t_cell(spec, n, key=key):
            return fmt_t_paren(spec[key]["t"])

        lines.append(f"{label} & " + " & ".join(cells(contrast_coef_cell)) + " \\\\")
        lines.append("& " + " & ".join(cells(contrast_t_cell)) + " \\\\")
    lines.append("\\colrule")

    rows = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{fmt_comma(data[s]['n_rows'])}}}"
        for s in _REVERSAL_SAMPLES
    )
    days = " & ".join(
        f"\\multicolumn{{2}}{{c}}{{{fmt_comma(data[s]['n_dates'])}}}"
        for s in _REVERSAL_SAMPLES
    )
    lines.append(f"Stacked rows & {rows} \\\\")
    lines.append(f"Trading days & {days} \\\\\\botrule")

    write_body("reversal_interaction_body.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Tables: rank_transform_class / rank_transform_reversal
# (tab:rank_transform_class, tab:rank_transform_reversal,
#  subsec:appendix_j_rank)
# Source: results/rank_transform_robustness.json
# ---------------------------------------------------------------------------

# The three disclosed classes of tab:investor_heterogeneity, in panel order,
# paired with the column heading each takes here.
_RANK_CLASSES = [
    ("capacity_bound", "Capacity-bound inst."),
    ("foreign", "Foreign"),
    ("individual", "Individual"),
]

# The four specifications of tab:investor_heterogeneity, as (JSON key, label).
_RANK_SPECS = [
    ("univariate_MC", "$S^{\\mathrm{MC}}$ only"),
    ("univariate_TV", "$S^{\\mathrm{TV}}$ only"),
    ("horse_race_MC", "Horse race, $S^{\\mathrm{MC}}$"),
    ("horse_race_TV", "Horse race, $S^{\\mathrm{TV}}$"),
]

# Column order within each stacking: the published level arm first, the rank
# arm second, so the table reads as a perturbation of the published estimate.
_RANK_ARMS = ("zscore", "rank")


def generate_rank_transform_bodies():
    """Generate both bodies for the within-day rank-transform robustness check.

    ``tab:rank_transform_class`` reports $t$-statistics only: the question
    tab:investor_heterogeneity's specifications are asked here is which filter
    predicts and whether it stays significant, not by how much. The reversal
    table carries coefficients as well, because subsec:appendix_j_rank reads
    the attenuation between the two arms as a bound rather than as an estimate.
    """
    data = load_json(RESULTS_DIR / "rank_transform_robustness.json")

    per_class = data["per_class"]
    lines = []
    for spec, label in _RANK_SPECS:
        cells = [fmt_t_paren(per_class[cls][arm][spec]["t_stat"]).strip("()")
                 for cls, _ in _RANK_CLASSES for arm in _RANK_ARMS]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines[-1] += "\\botrule"
    write_body("rank_transform_class_body.tex", "\n".join(lines) + "\n")

    stacked = data["stacked"]

    def contrast_rows(arm, rows):
        """Emit coefficient and $t$ lines for ``rows`` of one specification arm."""
        out = []
        for key, label in rows:
            cells, ts = [], []
            for sample in ("common", "union"):
                for transform in _RANK_ARMS:
                    fm = stacked[sample][transform][arm]
                    c = fm[key]
                    cells.append(fmt_coef_star(
                        c["mean"] * BPS_PER_RETURN,
                        _p_from_t(c["t"], fm["n_days"]), decimals=2))
                    ts.append(fmt_t_paren(c["t"]))
            out.append(f"{label} & " + " & ".join(cells) + " \\\\")
            out.append("& " + " & ".join(ts) + " \\\\")
        return out

    lines = contrast_rows("fama_macbeth_univariate", _REVERSAL_CONTRAST_ROWS)
    lines.append("\\colrule")
    lines += contrast_rows("fama_macbeth", [("reversal", "Contrast, horse race")])
    lines[-1] += "\\botrule"
    write_body("rank_transform_reversal_body.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: participation_matching
# (tab:participation_matching, subsec:appendix_participation_matching)
# Source: results/participation_matching_results.json
# ---------------------------------------------------------------------------

# Column order within each stacking: the unmatched arm first, then the two
# matching grids from coarse to fine, so the table reads as a perturbation of
# the published estimate. The overlap arms are not columns here because they
# differ from the unmatched one by almost nothing, which the prose states.
_CEM_COLUMNS = (("full", None), ("quintile", "matched"), ("decile", "matched"))


def generate_participation_matching_body():
    """Generate the body for the net-participation matching robustness check.

    Coefficients as well as $t$-statistics, because
    subsec:appendix_participation_matching reads which class's advantage the
    matching moves and by how much, not only whether each advantage stays
    significant.
    """
    data = load_json(RESULTS_DIR / "participation_matching_results.json")

    def arm(sample, grid, name):
        return data[sample][grid] if name is None else data[sample][grid][name]

    def contrast_rows(spec, rows):
        """Emit coefficient and $t$ lines for ``rows`` of one specification arm."""
        out = []
        for key, label in rows:
            cells, ts = [], []
            for sample in ("common", "union"):
                for grid, name in _CEM_COLUMNS:
                    fm = arm(sample, grid, name)[spec]
                    c = fm[key]
                    cells.append(fmt_coef_star(
                        c["mean"] * BPS_PER_RETURN,
                        _p_from_t(c["t"], fm["n_days"]), decimals=2))
                    ts.append(fmt_t_paren(c["t"]))
            out.append(f"{label} & " + " & ".join(cells) + " \\\\")
            out.append("& " + " & ".join(ts) + " \\\\")
        return out

    lines = contrast_rows("fama_macbeth_univariate", _REVERSAL_CONTRAST_ROWS)
    lines.append("\\colrule")
    lines += contrast_rows("fama_macbeth", [("reversal", "Contrast, horse race")])
    lines[-1] += "\\botrule"
    write_body("participation_matching_body.tex", "\n".join(lines) + "\n")


def generate_flow_persistence_kr_main_body():
    """Generate body for the main-text Korea regime-assignment table
    (qf_paper.tex subsec:regime_assignment, tab:directionality_kr).

    Korea rows of the round-trip-vs-accumulation decomposition, on the
    analysis window (Panel A) and the pre-period holdout (Panel B).
    Columns: gross-flow ($\\beta_M$, $\\beta_V$), net-accumulation
    ($\\beta_M$, $\\beta_V$), directionality ratio, $N$ stocks.
    The Panel A header lives in the paper skeleton because endfloat breaks
    when a body starts with \\multicolumn; the Panel B header is emitted
    here with the pre-period window read from the results JSON.
    """
    path = RESULTS_DIR / "flow_persistence_decomposition.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run flow_persistence_decomposition.py "
            f"to produce it."
        )
    fp = load_json(path)

    def pair(stage):
        bm, tm, bv, tv = (stage["beta_M"], stage["t_M"],
                          stage["beta_V"], stage["t_V"])
        m_s, v_s = _bold_max(f"{bm:+.3f} ({tm:+.1f})", f"{bv:+.3f} ({tv:+.1f})",
                             bm, bv)
        return _wrap_negative_in_math(m_s), _wrap_negative_in_math(v_s)

    def rows_for(block):
        out = []
        for label, blk in [("Capacity-bound institutions", block["capacity_bound"]),
                           ("Foreign", block["foreign"])]:
            gm, gv = pair(blk["gross_flow"])
            nm, nv = pair(blk["net_accum"])
            out.append(
                f"{label} & {gm} & {gv} & {nm} & {nv} & "
                f"{blk['directionality_median']:.3f} & {fmt_comma(blk['n_stocks'])} \\\\"
            )
        return out

    pre_years = fp["korea_preperiod"]["window"].replace("-", "--")
    lines = rows_for(fp["korea"])
    lines.append("\\colrule")
    lines.append(
        f"\\multicolumn{{7}}{{@{{}}l@{{}}}}"
        f"{{\\textbf{{Panel B: Pre-period ({pre_years})}}}} \\\\\\colrule"
    )
    lines.extend(rows_for(fp["korea_preperiod"]))
    _with_botrule(lines)
    write_body("flow_persistence_kr_main_body.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: directionality ratio against a trading-frequency null
# (qf_paper.tex subsec:appendix_directionality_frequency)
# Source: results/directionality_frequency_null.json
# ---------------------------------------------------------------------------

def generate_directionality_frequency_body():
    """Generate body for the sign-flip null on the directionality ratio.

    Each class's published ratio beside what its own trading frequency and
    execution magnitudes deliver with the direction randomised away, on the
    Korean analysis window (Panel A), the Korean pre-period holdout (Panel B),
    and the Taiwan panel (Panel C). Columns: median nonzero-flow days, observed
    ratio, sign-flip null, excess over it, $N$ stocks. The Panel A header lives
    in the paper skeleton because endfloat breaks when a body starts with
    \\multicolumn.

    A cell whose two-sided randomisation p-value stands above the resolution
    floor the draw count sets carries a dagger on its excess. Every Korean cell
    sits at the floor and Taiwan SITC does not, so the note states the rejection
    per cell rather than over all of them.
    """
    path = RESULTS_DIR / "directionality_frequency_null.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run directionality_frequency_null.py "
            f"to produce it."
        )
    dfn = load_json(path)
    floor = 2.0 / (dfn["config"]["n_draws"] + 1)

    # Row labels in the paper's case; the JSON's labels serve the analysis
    # script's console output.
    labels = {"capacity_bound": "Capacity-bound institutions",
              "institutional": "Investment trust (SITC)", "foreign": "Foreign"}

    def rows_for(block, keys=("capacity_bound", "foreign")):
        out = []
        for key in keys:
            c = block[key]
            mark = "$^{\\dagger}$" if c["p_two_sided"] > floor + 1e-12 else ""
            out.append(
                f"{labels[key]} & {fmt_comma(round(c['median_days']))} & "
                f"{c['observed']:.3f} & {c['null_mean']:.3f} & "
                f"{c['excess']:.2f}{mark} & {fmt_comma(c['n_stocks'])} \\\\"
            )
        return out

    def panel_header(letter, title):
        return (f"\\multicolumn{{6}}{{@{{}}l@{{}}}}"
                f"{{\\textbf{{Panel {letter}: {title}}}}} \\\\\\colrule")

    pre_years = dfn["korea_preperiod"]["window"].replace("-", "--")
    twn_years = dfn["taiwan"]["window"].replace("-", "--")
    lines = rows_for(dfn["korea"])
    lines.append("\\colrule")
    lines.append(panel_header("B", f"Korean pre-period ({pre_years})"))
    lines.extend(rows_for(dfn["korea_preperiod"]))
    lines.append("\\colrule")
    lines.append(panel_header("C", f"Taiwan ({twn_years})"))
    lines.extend(rows_for(dfn["taiwan"], keys=("institutional", "foreign")))
    _with_botrule(lines)
    write_body("directionality_frequency_body.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: twn_return_prediction (sec:taiwan)
# Source: results/twn_return_prediction_results.json
# The Taiwan analogue of tab:investor_heterogeneity: one panel per assigned
# class, univariate S^MC, univariate S^TV, and the horse race that puts them
# against each other on the same cross-sections.
# ---------------------------------------------------------------------------

def generate_twn_return_prediction_body():
    """Generate body for the Taiwan return-prediction table (sec:taiwan).

    One panel per class the assignment reaches: the capacity-bound SITC cell
    that replicates Korea and the foreign cell that does not. The Panel A
    header lives in the paper skeleton, as tab:directionality_frequency's does.
    """
    data = load_json(RESULTS_DIR / "twn_return_prediction_results.json")["by_investor"]
    panels = [
        ("Investment trust (SITC)", "institutional", "A"),
        ("Foreign investor",        "foreign",       "B"),
    ]
    lines = []
    for i, (label, key, panel) in enumerate(panels):
        d = data[key]
        if i > 0:
            lines.append(
                f"\\multicolumn{{4}}{{@{{}}l@{{}}}}{{\\textbf{{Panel {panel}: {label}}}}} \\\\\\colrule"
            )
        mc_uni, tv_uni, hr = d["univariate_MC"], d["univariate_TV"], d["horse_race"]
        lines.append(
            f"$S^{{\\mathrm{{MC}}}}$ & "
            f"{fmt_coef_star(mc_uni['coefficient'], mc_uni['p_value'])} "
            f"& --- & "
            f"{fmt_coef_star(hr['signal_MC']['coefficient'], hr['signal_MC']['p_value'])} \\\\"
        )
        lines.append(
            f"& {fmt_t_paren(mc_uni['t_stat'])} "
            f"& & {fmt_t_paren(hr['signal_MC']['t_stat'])} \\\\"
        )
        lines.append(
            f"$S^{{\\mathrm{{TV}}}}$ & --- & "
            f"{fmt_coef_star(tv_uni['coefficient'], tv_uni['p_value'])} & "
            f"{fmt_coef_star(hr['signal_TV']['coefficient'], hr['signal_TV']['p_value'])} \\\\"
        )
        lines.append(
            f"& & {fmt_t_paren(tv_uni['t_stat'])} "
            f"& {fmt_t_paren(hr['signal_TV']['t_stat'])} \\\\"
        )
    _with_botrule(lines)
    write_body("twn_return_prediction_body.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Table: skip_day (subsec:skip_day, return prediction on the first and second
# day after the flow)
# Source: results/skip_day_return_prediction_results.json
# Both normalisations on the returns over [t,t+1] and [t+1,t+2], both classes.
# Each class's MATCHED filter loses significance on the second day's return and
# both mismatched filters load on it, which is the ordering subsec:skip_day
# reads.
# ---------------------------------------------------------------------------

def generate_skip_day_body():
    """Generate body for the second-day-after-the-flow table (subsec:skip_day).

    Four rows: the two classes, each with its matched and its mismatched
    normalisation. Columns: filter, univariate
    $t$ at $[t,t+1]$ and $[t+1,t+2]$, horse-race $t$ at the same two horizons.
    Both normalisations are in the table body in every cell, because the
    second day's ordering between them is what subsec:skip_day reports.
    """
    path = RESULTS_DIR / "skip_day_return_prediction_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run skip_day_return_prediction.py to produce it."
        )
    sk = load_json(path)
    rows = [
        ("Korea capacity-bound inst.", sk["korea"]["capacity_bound"]),
        ("Korea Foreign",              sk["korea"]["foreign"]),
    ]

    def t_cell(t):
        # Normal-approximation stars on the t-stat; math-wrap negatives with the
        # stars outside the math block (the fmt_coef_star convention), so a
        # starred negative renders as $-3.94$***.
        a = abs(t)
        star = "***" if a > 2.58 else "**" if a > 1.96 else "*" if a > 1.65 else ""
        return (f"${t:.2f}$" if t < 0 else f"{t:.2f}") + star

    lines = []
    for label, c in rows:
        matched = c["matched_arm"]
        mismatched = "TV" if matched == "MC" else "MC"
        for arm, role, name in ((matched, "matched", label),
                                (mismatched, "mismatched", "")):
            lines.append(
                f"{name} & $S^{{\\mathrm{{{arm}}}}}$ ({role}) & "
                f"{t_cell(c['baseline'][f'univariate_{arm}']['t_stat'])} & "
                f"{t_cell(c['skip'][f'univariate_{arm}']['t_stat'])} & "
                f"{t_cell(c['baseline'][f'horse_race_{arm}']['t_stat'])} & "
                f"{t_cell(c['skip'][f'horse_race_{arm}']['t_stat'])} \\\\"
            )
    _with_botrule(lines)
    write_body("skip_day_body.tex", "\n".join(lines) + "\n")



# ---------------------------------------------------------------------------
# Table: long_short (sec:appendix_robustness_battery)
# Source: results/extended_robustness_results.json
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Table: factor_neutrality (sec:appendix_robustness_battery)
# Source: results/extended_robustness_results.json
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Count files: tables/obs_count.tex, tables/ticker_count.tex, tables/day_count.tex
# ---------------------------------------------------------------------------

def generate_count_files():
    """Regenerate observation/ticker/day count files from the unified panel.

    Reads the all-investors panel and counts only stock-days where the
    canonical Capacity-Bound Institutional flow is nonzero — the same subset
    that ``shared_utils.load_domestic_panel`` returns to the analysis scripts
    (D = D_capacity_bound after the subsec:appendix_aggregate_alignment rewrite).
    """
    panel = pd.read_parquet(
        DAILY_DATA_ALL_INVESTORS,
        columns=["date", "ticker", "D_capacity_bound"],
    )
    panel = panel[panel["D_capacity_bound"] != 0]
    n_rows = len(panel)

    # ``{,}`` rather than the table bodies' ``\,``: subsec:data now quotes this
    # count in the same paragraph as \sampForeignObs, which fmt_comma renders
    # with ``{,}`` like every other prose count.
    obs_str = f"{n_rows:,}".replace(",", "{,}")
    write_body("obs_count.tex", obs_str)
    write_body("ticker_count.tex", str(panel["ticker"].nunique()))
    write_body("day_count.tex", str(panel["date"].nunique()))


# ---------------------------------------------------------------------------
# tables/numbers.tex — orchestration
# ---------------------------------------------------------------------------

def generate_numbers_tex():
    """Generate tables/numbers.tex with \\newcommand macros for all key values.

    The paper \\input{tables/numbers} in the preamble; prose then uses e.g.
    \\fmMCt instead of hardcoded -16.75. Macros are organised by paper section
    via the ``_emit_*`` helpers in ``_macro_emitters``.
    """
    cmds = []

    def cmd(name, val):
        cmds.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    _emit_flow_persistence_macros(cmd)
    _emit_directionality_frequency_macros(cmd)
    _emit_fm_and_returns_macros(cmd)
    _emit_extended_robustness_and_multi_investor_macros(cmd)
    _emit_robustness_inline_and_foreign_macros(cmd)
    _emit_liquidity_stratified_macros(cmd)
    _emit_turnover_descriptive(cmd)
    _emit_reversal_interaction_macros(cmd)
    _emit_investment_trust_inclusion_macros(cmd)
    _emit_lagged_adv_macros(cmd)
    _emit_participation_matching_macros(cmd)
    _emit_flow_continuation_macros(cmd)
    _emit_free_exponent_macros(cmd)
    _emit_size_restricted_macros(cmd)
    _emit_rank_transform_macros(cmd)
    _emit_subtype_advantage_macros(cmd)
    _emit_vwap_formation_macros(cmd)
    _emit_sample_size_macros(cmd)
    _emit_foreign_definition_macros(cmd)
    _emit_taiwan_macros(cmd)
    _emit_supplementary_macros(cmd)
    _emit_manual_section_macros(cmd)

    write_body("numbers.tex", "% Auto-generated by code/output/generate_table_bodies.py\n"
               "% Do not edit — regenerate with: make tables\n"
               + "\n".join(cmds) + "\n")


def main():
    print("Generating table bodies...")

    generate_flow_persistence_kr_main_body()
    generate_directionality_frequency_body()
    generate_twn_return_prediction_body()
    generate_skip_day_body()
    generate_liquidity_stratified_body()
    generate_foreign_kospi200_split_body()
    generate_count_files()
    generate_numbers_tex()
    generate_investor_heterogeneity_body()
    generate_reversal_interaction_body()
    generate_rank_transform_bodies()
    generate_participation_matching_body()

    print(f"\nDone. Generated files in: {TABLES_DIR}")


if __name__ == "__main__":
    main()
    # Only after a real run: check_generator_idempotent calls main() with
    # write_body patched out to capture in memory, so nothing is written and a
    # sweep there would delete every body on disk.
    sweep_stale_bodies()
