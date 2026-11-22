"""Declarative spec tables for ``\\newcommand`` macros emitted by
``code/output/generate_table_bodies.py``.

Each spec is a list of ``(macro_name, dotted_path, fmt_string)`` rows that
``_emit_from_spec`` walks against a loaded JSON document. A row whose path
resolves to ``None`` stops the generator with the macro's name and the path.

Specs are grouped by *consumer* — the ``_emit_*`` function that uses them —
because that is the order in which macros appear in ``tables/numbers.tex``. A new
macro added in-place inside the right spec preserves the byte-identical
contract; reordering rows or moving them across specs changes the output.
"""


# ---------------------------------------------------------------------------
# _emit_fm_and_returns_macros — results/multicollinearity_test_results.json
# ---------------------------------------------------------------------------

# subsec:return_prediction prints the two diagnostics in prose. Each is the
# mean over the daily cross-sections of the capacity-bound contemporaneous
# panel, the population \sampTotalDays counts.
MULTICOLLINEARITY_MACROS = [
    ("mcollCorr", "correlation.mean",      "{:.3f}"),
    ("mcollVif",  "vif.mean",              "{:.3f}"),
]


# ---------------------------------------------------------------------------
# _emit_extended_robustness_and_multi_investor_macros — two source files
# (extended_robustness, multi_investor_robustness). Order below mirrors
# emission order; do not lump these
# together — byte-identity depends on the interleaving with inline emissions.
# ---------------------------------------------------------------------------

EXT_PP_MACROS = [
    ("oosPPMChr",  "pre_period_oos.horse_race.MC.t_stat", "{:.2f}"),
    ("oosPPTVhr",  "pre_period_oos.horse_race.TV.t_stat", "{:.2f}"),
    # Regression-day count (drops the final no-forward-return day), read off
    # the horse race that the pre-period row prints, not the +1 panel-day count.
    ("sampPPDays", "pre_period_oos.horse_race.n_days",     "{}"),
]
EXT_LS_MACROS = [
    ("lsMCbps",    "long_short.S_MC.mean_bps", "{:.2f}"),
    ("lsMCt",      "long_short.S_MC.t_stat",   "{:.2f}"),
    ("lsTVbps",    "long_short.S_TV.mean_bps", "{:.2f}"),
]
# fnAlpha needs *BPS_PER_RETURN (return → bps); stays inline. Other fn macros are pure extracts.
EXT_FN_REST_MACROS = [
    ("fnAlphaT",   "factor_neutrality.alpha.t_stat", "{:.2f}"),
    ("fnSMB",      "factor_neutrality.SMB.coef",     "{:.3f}"),
    ("fnSMBt",     "factor_neutrality.SMB.t_stat",   "{:.2f}"),
    # \fnHML/\fnHMLt retired: the HML loading is an incidental completeness
    # detail, not a load-bearing prose claim.
]

MI_IH_MACROS = [
    ("ihAggregateMCt",   "return_predictions.institutional_aggregate.univariate_MC.t_stat", "{:.2f}"),
    ("ihAggregateTVt",   "return_predictions.institutional_aggregate.univariate_TV.t_stat", "{:.2f}"),
    ("ihAggregateHRMCt", "return_predictions.institutional_aggregate.horse_race_MC.t_stat", "{:.2f}"),
    ("ihAggregateHRTVt", "return_predictions.institutional_aggregate.horse_race_TV.t_stat", "{:.2f}"),
]

# ---------------------------------------------------------------------------
# _emit_robustness_inline_and_foreign_macros — extended_robustness (winsor/NW
# block) + extended_robustness_foreign (factor-neutrality)
# ---------------------------------------------------------------------------

WIN_NW_MACROS = [
    ("winMCt",     "outlier_robustness.winsorized.horse_race.MC.t_stat", "{:.2f}"),
    ("winTVt",     "outlier_robustness.winsorized.horse_race.TV.t_stat", "{:.2f}"),
    ("nwFiveMCt",  "alternative_se.MC.nw5_t",  "{:.2f}"),
    ("nwTenMCt",   "alternative_se.MC.nw10_t", "{:.2f}"),
    ("nwFiveTVt",  "alternative_se.TV.nw5_t",  "{:.2f}"),
    ("nwTenTVt",   "alternative_se.TV.nw10_t", "{:.2f}"),
]
# fFnAlpha has *BPS_PER_RETURN scaling, kept inline. Other fFn macros are pure extracts.
FOREIGN_FN_REST_MACROS = [
    ("fFnAlphaT", "factor_neutrality.alpha.t_stat", "{:.2f}"),
]


# ---------------------------------------------------------------------------
# _emit_liquidity_stratified_macros — results/liquidity_stratified_results.json
# ---------------------------------------------------------------------------

# Return prediction inside market-cap quintiles
# (sec:appendix_liquidity_stratified). Only the quintiles prose quotes appear
# here; the table body reads the same file for all five. ``Mis`` names the
# mismatched filter of the class in question, so liqForSizeQFiveT is foreign
# S^TV and liqForSizeQFiveMisT is foreign S^MC in the same cell. The three
# day-count macros are emitted inline instead, for LaTeX-safe thousands.
_LS_Q = "by_{axis}.{inv}.{q}.univariate_{arm}.S_{arm}_{inv}.t_stat"
LIQ_STRAT_MACROS = [
    # Size axis: the two cells where the advantage is not established.
    ("liqCapSizeQOneT",    _LS_Q.format(axis="size", inv="capacity_bound", q="Q1", arm="MC"), "{:.2f}"),
    ("liqCapSizeQTwoT",    _LS_Q.format(axis="size", inv="capacity_bound", q="Q2", arm="MC"), "{:.2f}"),
    ("liqForSizeQFiveT",   _LS_Q.format(axis="size", inv="foreign", q="Q5", arm="TV"),        "{:.2f}"),
    ("liqForSizeQFiveMisT",_LS_Q.format(axis="size", inv="foreign", q="Q5", arm="MC"),        "{:.2f}"),
]


# ---------------------------------------------------------------------------
SUBINVESTOR_ROW_ORDER = [
    ("pension",                'Pension Funds (yeon-gigeum-deung)'),
    ("insurance",              'Insurance (boheom)'),
    ("investment_trust",       'Investment Trust (tusin)'),
    ("pe_funds",               'Private Equity Funds (samo-peondeu)'),
    ("financial_inv",          'Proprietary (geumyung-tuja)'),
    ("other_financial",        'Other Financial (gita-geumyung)'),
    ("bank",                   'Banks (eunhaeng)'),
    ("government",             'Government (gukga)'),
    ("capacity_bound",         'Capacity-Bound (Pension + Insurance)'),
    ("institutional_aggregate",'KRX Institutional Aggregate (gigwan-gye)'),
]
