"""Data loading and signal-construction helpers shared across the QF analysis scripts."""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize as _scipy_winsorize


def compute_normalisations(df, d_col='D', m_col='M', v_col='V',
                           smc_col='S_MC', stv_col='S_TV'):
    """Add S^{MC} = D/M and S^{TV} = D/V columns to ``df`` IN PLACE.

    S^{TV} is masked to NaN where V == 0 (the only divisor that can vanish in
    KRX data; M is always positive on a trading day with quoted close).
    """
    df[smc_col] = df[d_col] / df[m_col]
    df[stv_col] = np.where(df[v_col] > 0, df[d_col] / df[v_col], np.nan)


def add_investor_signals(df, names):
    """Compute ``S_MC_{name}`` / ``S_TV_{name}`` for each name via :func:`compute_normalisations`.

    Used by the multi-investor data pipelines to vectorise the per-investor
    and per-sub-investor signal loops. Missing ``D_{name}`` columns are
    filled with zero before normalisation (matches the
    sub-investor outer-merge convention where a category may be absent for
    early years of the panel).
    """
    for name in names:
        d_col = f'D_{name}'
        if d_col not in df.columns:
            df[d_col] = 0.0
        compute_normalisations(df, d_col=d_col,
                               smc_col=f'S_MC_{name}', stv_col=f'S_TV_{name}')


# Canonical "Capacity-Bound Institutions" = pension funds + insurance, a subset
# of the KRX 기관계 aggregate and the canonical D in main analysis (qf_paper.tex
# subsec:data). code/empirical/robustness/investment_trust_inclusion.py re-estimates the reversal with the
# investment-trust sub-type added (qf_paper.tex
# subsec:appendix_aggregate_alignment).
CAPACITY_BOUND_COMPONENTS = ('pension', 'insurance')


def merge_subinvestor_panel(base_wide, sub_wide, *, top_types, sub_types):
    """Outer-merge the sub-investor wide panel onto ``base_wide`` and zero-fill flow.

    A stock-day with nonzero 기관계 aggregate flow but zero across all 8
    sub-types is rare (it would mean trading reported under the aggregate but
    not classified into a sub-type); the outer merge preserves rows where
    either side has signal, and the ``fillna(0)`` makes a missing category an
    explicit zero flow. Shared by the full (2020+) and pre-period (2015-2019)
    Korean pipelines.
    """
    merged = base_wide.merge(sub_wide, on=['date', 'ticker'], how='outer')
    for col in [f'D_{t}' for t in top_types] + [f'D_{s}' for s in sub_types]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
    return merged


def set_capacity_bound_canonical(df, components=CAPACITY_BOUND_COMPONENTS):
    """Point the canonical ``D`` / ``S_MC`` / ``S_TV`` at the capacity-bound aggregate.

    Sets ``D_capacity_bound = sum(D_{component})`` (default pension +
    insurance), computes its normalisations, preserves the raw KRX 기관계
    aggregate under the explicit ``D_institutional_aggregate`` alias (for the
    subsec:appendix_aggregate_alignment paragraph), and overwrites the canonical
    ``D``/``S_MC``/``S_TV`` to the capacity-bound series. Requires
    ``D_institutional`` and the per-component ``D_{component}`` columns plus
    their signals to already exist (call after :func:`add_investor_signals`).
    Shared by the full and pre-period Korean pipelines.
    """
    df['D_capacity_bound'] = sum(df[f'D_{c}'] for c in components)
    compute_normalisations(df, d_col='D_capacity_bound',
                           smc_col='S_MC_capacity_bound',
                           stv_col='S_TV_capacity_bound')
    df['D_institutional_aggregate']    = df['D_institutional']
    df['S_MC_institutional_aggregate'] = df['S_MC_institutional']
    df['S_TV_institutional_aggregate'] = df['S_TV_institutional']
    df['D']    = df['D_capacity_bound']
    df['S_MC'] = df['S_MC_capacity_bound']
    df['S_TV'] = df['S_TV_capacity_bound']


def with_log_abs_flow(df):
    """Return a copy of ``df`` with ``log_abs_flow = log(|D|)`` attached (qf_paper.tex subsec:return_prediction).

    Callers pass a panel already restricted to nonzero flow (``load_domestic_panel``
    drops ``D == 0``), so ``log(|D|)`` is finite without a shift constant.
    """
    out = df.copy()
    out['log_abs_flow'] = np.log(np.abs(out['D']))
    return out


# === Pipeline filter / forward-return constants ===

SIZE_QUINTILE_LABELS = ('Q1_Small', 'Q2', 'Q3', 'Q4', 'Q5_Large')
"""Cross-sectional market-cap quintile labels (qf_paper.tex sec:appendix_liquidity_stratified, Panel A). The
analysis pipeline assigns ``size_quintile`` via :func:`within_day_quantile`
so downstream subsample code can filter on the human-readable label rather
than an integer."""


def within_day_quantile(df, col, labels, *, date_col='date'):
    """Bin ``col`` into ``len(labels)`` equal-count buckets within each day.

    Breakpoints are formed on each day's own cross-section, which is what the
    quantile-partitioned tables state (qf_paper.tex
    sec:appendix_liquidity_stratified). A pooled quantile cut over the
    whole sample would not be the same partition: the market-cap distribution
    drifts upward across the main panel, so a stock could change bucket
    from calendar drift alone without moving in any day's cross-section.

    Cuts on the within-day percentile rank rather than on the level, so tied
    or degenerate cross-sections do not collapse bin edges, and a stock-day
    whose ``col`` is missing stays unlabelled instead of aborting the day.
    """
    rank = df.groupby(date_col)[col].rank(pct=True)
    return pd.cut(rank, bins=np.linspace(0.0, 1.0, len(labels) + 1),
                  labels=list(labels), include_lowest=True)

CORE_PANEL_COLS = ('M', 'V', 'D', 'S_MC', 'S_TV', 'turnover')
"""Core analysis columns: rows missing any of these are dropped before
quintile binning fires. Captures the existing
``dropna(subset=...)`` lists in both pipelines."""


def apply_pipeline_filters(df, *, core_cols=CORE_PANEL_COLS):
    """Drop missing/inf rows and assign ``size_quintile``.

    Reproduces the cleaning steps shared by the full and pre-period data
    pipelines: dropna on the core analysis columns, replace ±inf with NaN
    and drop again on the signal columns, then bin ``M`` into the five
    :data:`SIZE_QUINTILE_LABELS` buckets via :func:`within_day_quantile`.
    Returns the filtered frame (not in-place — the quintile assignment
    returns a new column, but the rest of the chain re-binds ``df`` to
    filtered copies).

    The written column is a convenience for panel inspection, not the one the
    quintile tables regress on: ``load_domestic_panel`` reassigns
    ``size_quintile`` after the nonzero-flow filter so the breakpoints reflect
    the universe actually being partitioned.
    """
    df = df.dropna(subset=list(core_cols))
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['S_MC', 'S_TV', 'turnover'])
    df = df.copy()
    df['size_quintile'] = within_day_quantile(df, 'M', SIZE_QUINTILE_LABELS)
    return df


def compute_forward_returns(df, horizons=(1, 5, 20), group_col='ticker'):
    """Add ``R_t`` and ``R_t{h}`` (forward) return columns IN PLACE.

    ``R_t`` is the per-stock close-to-close pct_change; ``R_t{h}`` shifts
    the ``h``-day pct_change up by ``h`` rows so each row carries the future
    return realised over that stock's next ``h`` in-sample observations
    (``h`` trading days when the stock trades on every in-sample date).
    Assumes ``df`` is sorted by ``[group_col, 'date']``. Horizons (1, 5, 20) correspond to
    daily / weekly / monthly forward returns in qf_paper.tex subsec:return_prediction.

    ``group_col`` is the per-stock identifier (``'ticker'`` for the Korean
    panels) so the forward return is computed within each stock's own series.

    ``close`` is the total-return adjusted close, not the traded print, and it
    is the vendor's own series rather than one reconstructed here: FnGuide's
    수정주가(현금배당포함) export (``data._fnguide_price``), which removes
    splits, free and paid rights offerings and capital reductions *and*
    reinvests the cash dividend, so a distribution is a reinvestment rather
    than a price drop the return records as a loss. The panel's other price,
    ``close_raw``, is the traded print, for the two call sites that read a
    ratio against a quantity formed at that print rather than a return. That
    is the convention the returns row of subsec:data states, so the caption
    and this column stand or fall together.
    """
    by_stock_close = df.groupby(group_col)['close']
    df['R_t'] = by_stock_close.pct_change()
    for h in horizons:
        df[f'R_t{h}'] = by_stock_close.pct_change(periods=h).shift(-h)


def winsorize(series, lower=0.01, upper=0.99):
    """Winsorize a series at given percentiles via ``scipy.stats.mstats.winsorize``.

    ``lower``/``upper`` are kept as percentile cut points (the convention used
    throughout the QF call sites); scipy expects symmetric ``limits`` as
    fractions to truncate at each tail, so the upper bound is reparameterised
    to ``1 - upper``. Returns a pandas Series when given one (preserving
    index/name) and a plain ndarray otherwise.

    The 1/99 default is the published cut: the outlier arm of
    ``empirical.robustness.robustness_tests_extended`` is the only caller
    and takes it, and the manuscript reports it through the
    ``\\winsorLowerPct`` / ``\\winsorUpperPct`` macros, which the generator
    reads back off this signature (subsec:appendix_j_outlier).

    Non-finite entries take no part in the cut points and are written back in
    their original positions. scipy ranks NaN above every finite value, so
    passing them through would place the upper clip beyond the true quantile
    and leave part of the right tail unwinsorised. A series with no finite
    entry has no cut points and comes back unchanged.
    """
    values = np.asarray(series, dtype=float)
    finite = np.isfinite(values)
    out = values.copy()
    if finite.any():
        out[finite] = np.asarray(
            _scipy_winsorize(values[finite], limits=[lower, 1.0 - upper]))
    if isinstance(series, pd.Series):
        return pd.Series(out, index=series.index, name=series.name)
    return out


def cross_sectional_zscore(s):
    """Z-score a series; return ``None`` when the std is zero (degenerate cross-section)."""
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else None


_DEFAULT_PANEL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'results', 'full_daily_data_all_investors.parquet',
)

# Columns load_domestic_panel needs for its own filter/bin/sort regardless of
# the caller's ``columns=`` request: D (nonzero filter), M and date
# (size_quintile breakpoints), and date/ticker (final sort key).
_DOMESTIC_REQUIRED_COLS = ('date', 'ticker', 'D', 'M')


def load_domestic_panel(input_file=_DEFAULT_PANEL, *, columns=None):
    """Load the unified daily panel filtered to nonzero Capacity-Bound Institutional flow.

    Reads ``results/full_daily_data_all_investors.parquet`` (the multi-investor file
    produced by ``code/data/full_daily_data_pipeline.py``), keeps
    only stock-days where the canonical ``D`` column is nonzero (``D =
    D_capacity_bound = D_pension + D_insurance``), and recomputes
    ``size_quintile`` via :func:`within_day_quantile` on the filtered subset,
    so quintile membership reflects the capacity-bound universe on the day it
    is measured. The KRX 기관계 aggregate is preserved separately as
    ``D_institutional_aggregate`` for the subsec:appendix_aggregate_alignment
    paragraph.

    ``columns``, if given, restricts the Parquet read to those columns (the
    filter/bin/sort keys in :data:`_DOMESTIC_REQUIRED_COLS` are always added);
    leave it ``None`` to load the full panel.
    """
    if input_file is None:
        input_file = _DEFAULT_PANEL
    if columns is not None:
        columns = list(dict.fromkeys(list(columns) + list(_DOMESTIC_REQUIRED_COLS)))
    df = pd.read_parquet(input_file, columns=columns)
    df = df[df['D'] != 0].copy()
    df = df.drop(columns=['size_quintile'], errors='ignore')
    df['size_quintile'] = within_day_quantile(df, 'M', SIZE_QUINTILE_LABELS)
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    return df


def load_all_investors_panel(
    input_file=None,
    *,
    flow_col: str | None = None,
    required_cols: Iterable[str] | None = None,
    columns: Iterable[str] | None = None,
    verbose: bool = True,
):
    """Load the multi-investor daily panel (no capacity-bound flow filter).

    Mirrors :func:`load_domestic_panel` but exposes the unfiltered
    multi-investor panel. Used by foreign / individual / multi-investor
    analyses where the ``D != 0`` capacity-bound restriction would
    discard the relevant observations.

    Parameters
    ----------
    input_file
        Path to the all-investors Parquet; defaults to the same file
        ``load_domestic_panel`` reads.
    flow_col
        If given, restrict to stock-days where ``df[flow_col] != 0`` (e.g.
        ``"D_foreign"`` for foreign-investor analyses).
    required_cols
        If given, raise ``ValueError`` when any of these columns is absent.
        Catches stale panels missing the per-investor signal columns.
    columns
        If given, restrict the Parquet read to those columns (``date`` /
        ``ticker``, ``flow_col``, and ``required_cols`` are always added so
        the filter, summary, and existence check still work); leave ``None``
        to load the full panel.
    verbose
        Print a short summary (rows, unique dates/tickers, date range).
    """
    if input_file is None:
        input_file = _DEFAULT_PANEL
    if columns is not None:
        always = ['date', 'ticker']
        if flow_col is not None:
            always.append(flow_col)
        if required_cols is not None:
            always.extend(required_cols)
        columns = list(dict.fromkeys(list(columns) + always))
    df = pd.read_parquet(input_file, columns=columns)

    if required_cols is not None:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    if flow_col is not None:
        n_before = len(df)
        df = df[df[flow_col] != 0].copy()
        if verbose:
            print(f"After {flow_col} != 0 filter: {len(df):,} observations "
                  f"(dropped {n_before - len(df):,})")

    if verbose:
        print(f"Loaded {len(df):,} observations")
        print(f"Unique dates: {df['date'].nunique():,}")
        print(f"Unique tickers: {df['ticker'].nunique():,}")
        if len(df):
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df
