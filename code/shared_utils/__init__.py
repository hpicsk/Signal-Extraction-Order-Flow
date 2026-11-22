"""
shared_utils — Local copy of shared utilities for self-contained distribution.
"""

from .config import (
    END_DATE_MAIN,
    END_DATE_PRE,
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    START_DATE_MAIN,
    START_DATE_PRE,
)
from .data import (
    CAPACITY_BOUND_COMPONENTS,
    add_investor_signals,
    apply_pipeline_filters,
    compute_forward_returns,
    compute_normalisations,
    cross_sectional_zscore,
    load_all_investors_panel,
    load_domestic_panel,
    merge_subinvestor_panel,
    set_capacity_bound_canonical,
    winsorize,
    with_log_abs_flow,
)
from .io import print_header, save_csv, save_json, save_parquet
from .plotting import PAPER_RCPARAMS, apply_paper_rcparams
from .regression import (
    CRIT_1PCT,
    CRIT_5PCT,
    CRIT_10PCT,
    daily_stats_pack,
    fama_macbeth_horse_race,
    fama_macbeth_univariate,
    fit_fama_macbeth,
    fit_fm_spec,
    fm_avg_daily_r2,
    fm_investor_signal_triple,
    ols_fit,
    ols_with_cluster_se,
    pack_fm_spec_flat,
    pack_fm_spec_investor,
    portfolio_stats,
    prepare_fm_panel,
    quintile_spread,
    run_fm_triple,
    slope_se,
    t_stars,
    wls_fit,
)

__all__ = [
    # .config
    "END_DATE_MAIN",
    "END_DATE_PRE",
    "MC_SEED",
    "MIN_STOCKS_PER_DAY_DEFAULT",
    "START_DATE_MAIN",
    "START_DATE_PRE",
    # .data
    "CAPACITY_BOUND_COMPONENTS",
    "add_investor_signals",
    "apply_pipeline_filters",
    "compute_forward_returns",
    "compute_normalisations",
    "cross_sectional_zscore",
    "load_all_investors_panel",
    "load_domestic_panel",
    "merge_subinvestor_panel",
    "set_capacity_bound_canonical",
    "winsorize",
    "with_log_abs_flow",
    # .io
    "print_header",
    "save_csv",
    "save_json",
    "save_parquet",
    # .plotting
    "PAPER_RCPARAMS",
    "apply_paper_rcparams",
    # .regression
    "CRIT_1PCT",
    "CRIT_5PCT",
    "CRIT_10PCT",
    "daily_stats_pack",
    "fama_macbeth_horse_race",
    "fama_macbeth_univariate",
    "fit_fama_macbeth",
    "fit_fm_spec",
    "fm_avg_daily_r2",
    "fm_investor_signal_triple",
    "ols_fit",
    "ols_with_cluster_se",
    "pack_fm_spec_flat",
    "pack_fm_spec_investor",
    "portfolio_stats",
    "prepare_fm_panel",
    "quintile_spread",
    "run_fm_triple",
    "slope_se",
    "t_stars",
    "wls_fit",
]
