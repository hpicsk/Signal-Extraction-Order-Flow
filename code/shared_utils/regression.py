"""Single OLS / WLS / FM backend used across the QF analysis scripts.

Also exports the small ``daily_stats_pack`` helper that converts a vector
of per-day cross-sectional statistics (correlations, coefficients, ICs)
into the standard Fama-MacBeth-style mean / std / SE / t quadruple used
by the daily-loop call sites in ``multicollinearity`` and
``threshold_sensitivity``.


All linear regression in the paper routes through this module. Bodies delegate
to ``statsmodels`` (single-equation OLS / WLS) and ``linearmodels``
(panel and Fama-MacBeth), resting on community-vetted implementations rather
than hand-rolled matrix algebra. ``ols_fit`` and ``wls_fit`` return the raw
``statsmodels.regression.linear_model.RegressionResultsWrapper`` so callers
access the standard ``.params``, ``.bse``, ``.fittedvalues``, ``.resid``,
``.rsquared``, ``.normalized_cov_params``, ``.nobs`` interface directly.
``slope_se`` is a thin convenience that drops the intercept column from
the ``.bse`` vector (with an optional HC0 sandwich).

Fama-MacBeth helpers (``fit_fama_macbeth``, ``fama_macbeth_univariate``,
``fama_macbeth_horse_race``) are thin wrappers around
``linearmodels.panel.FamaMacBeth``. The shared pre-processing recipe — drop
NaNs on signal/return columns, drop days with fewer than
``min_stocks_per_day`` stocks, drop days where any regressor has zero
cross-sectional std, day-standardize regressors — lives in
``prepare_fm_panel`` and is the only bespoke step. The regression itself
is delegated.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import FamaMacBeth
from scipy.stats import norm


# Two-sided standard-normal critical values at the 1/5/10% levels. Every sample
# in the paper exceeds 1000 observations, so the normal approximation matches
# the Student-t to two decimals. These mirror
# ``output/_formatters.{CRIT_1PCT,CRIT_5PCT,CRIT_10PCT}`` (the LaTeX layer keeps
# its own copy to stay import-light); significance stars on a bare t-statistic
# route through ``t_stars`` below — the t-stat counterpart of the p-value-based
# ``output/_formatters.stars``.
CRIT_1PCT = float(norm.ppf(0.995))   # 2.576
CRIT_5PCT = float(norm.ppf(0.975))   # 1.960
CRIT_10PCT = float(norm.ppf(0.95))   # 1.645


def t_stars(t) -> str:
    """Return ``***`` / ``**`` / ``*`` / ``''`` for ``|t|`` at the 1/5/10%
    two-sided normal critical values (:data:`CRIT_1PCT` etc.).

    The p-value-based counterpart used by the LaTeX table bodies is
    ``output/_formatters.stars``.
    """
    a = abs(t)
    if a > CRIT_1PCT:
        return "***"
    if a > CRIT_5PCT:
        return "**"
    if a > CRIT_10PCT:
        return "*"
    return ""


def daily_stats_pack(values) -> dict:
    """Pack a per-day statistic vector into ``{mean, std, se, t}``.

    ``std`` uses ``ddof=1`` (sample), ``se = std / sqrt(N)``, ``t = mean /
    se`` (zero when ``se == 0``). All three call sites in the QF empirical
    scripts use this exact recipe with ``ddof=1``; pulling it into one
    helper makes the consistency explicit and prevents accidental ddof
    drift.
    """
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    se = std / float(np.sqrt(len(arr)))
    t = mean / se if se > 0 else 0.0
    return {'mean': mean, 'std': std, 'se': se, 't': t}


def portfolio_stats(returns) -> dict:
    """Mean / std / Sharpe / t-stat / cum-return summary for a daily return series.

    ``std`` uses ``ddof=0`` (numpy default); ``sharpe = (mean / std) *
    sqrt(252)``; ``t_stat = mean / (std / sqrt(N))``; ``mean_bps = mean
    * 10000``; ``cum_ret`` is the compounded total. Used by the
    long-short portfolio tests in ``threshold_sensitivity`` and
    ``robustness_tests_extended``.
    """
    arr = np.asarray(returns, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std())
    sharpe = float((mean / std) * np.sqrt(252)) if std > 0 else 0.0
    t_stat = float(mean / (std / np.sqrt(n))) if std > 0 else 0.0
    cum_ret = float((1.0 + pd.Series(arr)).cumprod().iloc[-1] - 1.0) if n else 0.0
    return {
        'mean_bps': mean * 10000.0,
        'std': std,
        'sharpe': sharpe,
        't_stat': t_stat,
        'cum_ret': cum_ret,
        'n_days': n,
    }


def quintile_spread(frame, signal_col, return_col, *, n_quintiles=5):
    """Top-minus-bottom mean ``return_col`` from an ``n_quintiles`` sort on
    ``signal_col`` within a single cross-section (one trading day).

    Sorts via ``pd.qcut(..., duplicates='drop')`` and returns ``mean(return |
    top bucket) - mean(return | bottom bucket)`` (NaN when either tail bucket
    is empty). The caller owns the per-day validity filter — minimum
    cross-section, NaN drop, and (where ties are expected) the ``<
    n_quintiles`` distinct-value skip.
    """
    q = pd.qcut(frame[signal_col], n_quintiles,
                labels=list(range(1, n_quintiles + 1)), duplicates='drop')
    top = frame.loc[q == n_quintiles, return_col].mean()
    bot = frame.loc[q == 1, return_col].mean()
    return top - bot


def _design(X: np.ndarray, fit_intercept: bool) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if fit_intercept:
        return np.column_stack([np.ones(X.shape[0]), X])
    return X


def ols_fit(y, X, *, fit_intercept: bool = True):
    """OLS of ``y`` on ``X`` via ``statsmodels.OLS``.

    Returns the statsmodels ``RegressionResultsWrapper`` directly so callers
    use ``.params``, ``.bse``, ``.fittedvalues``, ``.resid``, ``.rsquared``,
    ``.normalized_cov_params`` (= ``(X'X)^{-1}`` for OLS), and ``.nobs``.
    With ``fit_intercept=True`` (default) a constant column is prepended;
    then ``result.params[0]`` is the intercept and ``result.params[1:]``
    are the slopes.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    return sm.OLS(y, _design(X, fit_intercept)).fit()


def wls_fit(y, X, weights, *, fit_intercept: bool = True):
    """WLS of ``y`` on ``X`` with diagonal ``weights`` via ``statsmodels.WLS``.

    Returns the statsmodels ``RegressionResultsWrapper``. ``.fittedvalues``
    and ``.resid`` are on the original (unweighted) scale;
    ``.normalized_cov_params`` is ``(X' diag(w) X)^{-1}``.
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    return sm.WLS(y, _design(X, fit_intercept), weights=w).fit()


def ols_with_cluster_se(y, X, dates, *, contrasts=None):
    """OLS with one-way cluster-by-date robust SEs via ``statsmodels``.

    ``X`` already includes the intercept column. statsmodels' ``cov_type='cluster'``
    with ``use_correction=True`` applies the Petersen ``G/(G-1) * (N-1)/(N-k)``
    finite-sample factor used throughout the paper.

    Returns a dict with ``coef``, ``se``, ``t`` lists in the order of ``X``'s
    columns.

    ``contrasts``, if given, maps a label to a weight vector over ``X``'s
    columns; each is tested via ``statsmodels``' ``t_test`` against the same
    clustered covariance and returned under a ``contrasts`` key as
    ``{label: {value, se, t_stat, p_value}}``. The contrast of
    subsec:reversal_test (the difference-in-differences across the two
    normalisations and the two trader classes) is a single such weight vector,
    so its standard error comes from the fitted cluster covariance rather than
    from differencing two separately-estimated standard errors.
    """
    res = sm.OLS(y, X).fit(
        cov_type='cluster',
        cov_kwds={'groups': np.asarray(dates), 'use_correction': True},
    )
    coef = np.asarray(res.params, dtype=float)
    se = np.asarray(res.bse, dtype=float)
    t = np.asarray(res.tvalues, dtype=float)
    out = {
        "coef": coef.tolist(),
        "se": se.tolist(),
        "t": t.tolist(),
    }
    if contrasts:
        out["contrasts"] = {}
        for label, weights in contrasts.items():
            tt = res.t_test(np.asarray(weights, dtype=float))
            out["contrasts"][label] = {
                "value": float(np.squeeze(tt.effect)),
                "se": float(np.squeeze(tt.sd)),
                "t_stat": float(np.squeeze(tt.tvalue)),
                "p_value": float(np.squeeze(tt.pvalue)),
            }
    return out


def slope_se(result, *, hc0: bool = False) -> np.ndarray:
    """Slopes-only SE vector from a statsmodels OLS/WLS result.

    Drops the intercept column if the design matrix's first column is
    constant (i.e., when the caller used the default ``fit_intercept=True``).
    Set ``hc0=True`` for the White HC0 heteroskedasticity-robust sandwich
    (no finite-sample correction), matching the convention used elsewhere
    in the paper.
    """
    if hc0:
        result = result.get_robustcov_results(cov_type="HC0")
    bse = np.asarray(result.bse, dtype=float)
    exog = np.asarray(result.model.exog, dtype=float)
    if exog.ndim == 2 and exog.shape[1] > 0 and np.all(exog[:, 0] == 1.0):
        return bse[1:]
    return bse


def prepare_fm_panel(df, signal_cols, return_col, *,
                     min_stocks_per_day, row_filter=None,
                     entity_col="ticker", date_col="date"):
    """Apply the shared FM pre-processing recipe and return a panel ready
    for ``linearmodels.FamaMacBeth`` (or a matching pooled-OLS spec).

    Steps, in order:
      1. ``row_filter(sub)`` is applied per date if provided.
      2. NaNs are dropped on ``signal_cols + [return_col]``.
      3. Days with fewer than ``min_stocks_per_day`` rows are dropped.
      4. Days where any ``signal_cols`` regressor has zero cross-sectional
         std (degenerate cross-section that would inject NaN slopes) are
         dropped.
      5. Each signal column is z-scored within day.

    Returns a flat DataFrame (NOT MultiIndexed) with ``entity_col``,
    ``date_col``, the day-standardized ``signal_cols``, and ``return_col``.
    Returns ``None`` if no day passes the filter.

    The FM and pooled-OLS paths in the QF pipeline share this filtered
    panel by construction so ``output.assertions.signal_structure.scaling.check_fama_macbeth`` can
    cross-check ``n_obs`` and ``n_dates``.
    """
    parts = []
    for _, sub in df.groupby(date_col):
        if row_filter is not None:
            sub = row_filter(sub)
        sub = sub.dropna(subset=[*signal_cols, return_col])
        if len(sub) < min_stocks_per_day:
            continue
        ok = True
        zs = {}
        for c in signal_cols:
            s = sub[c]
            std = s.std()
            if std == 0 or not np.isfinite(std):
                ok = False
                break
            zs[c] = (s - s.mean()) / std
        if not ok:
            continue
        out = sub[[entity_col, date_col, return_col]].copy()
        for c in signal_cols:
            out[c] = zs[c].values
        parts.append(out)
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def fit_fama_macbeth(df, signal_cols, return_col, *,
                     min_stocks_per_day, row_filter=None,
                     cov_type="unadjusted", kernel=None, bandwidth=None,
                     entity_col="ticker", date_col="date"):
    """Run Fama-MacBeth via ``linearmodels.panel.FamaMacBeth``.

    The panel is pre-processed by ``prepare_fm_panel`` (day-standardized
    regressors, NaN/degenerate-day filter). The actual cross-sectional
    regression and time-series aggregation are delegated to
    ``linearmodels``.

    ``cov_type`` defaults to ``"unadjusted"`` (simple Fama-MacBeth t-stat,
    ``std/sqrt(T)``-style SE). Pass ``cov_type="kernel"`` with a Bartlett
    or Parzen ``kernel`` and ``bandwidth`` (= number of lags) for the
    HAC-adjusted FM SE used by the subsec:appendix_j_outlier robustness check.

    Returns ``(FamaMacBethResults, panel)`` where ``panel`` is the flat
    pre-processed frame returned by ``prepare_fm_panel``. Returns
    ``(None, None)`` if no day survived the filter.
    """
    panel = prepare_fm_panel(
        df, signal_cols, return_col,
        min_stocks_per_day=min_stocks_per_day, row_filter=row_filter,
        entity_col=entity_col, date_col=date_col,
    )
    if panel is None:
        return None, None
    indexed = panel.set_index([entity_col, date_col])
    mod = FamaMacBeth(indexed[return_col], indexed[signal_cols])
    res = mod.fit(cov_type=cov_type, kernel=kernel, bandwidth=bandwidth,
                  debiased=True)
    return res, panel


def _fm_stats(res, col):
    """Pack a single regressor's stats from a ``FamaMacBethResults``."""
    return {
        "coefficient": float(res.params[col]),
        "se": float(res.std_errors[col]),
        "t_stat": float(res.tstats[col]),
        "p_value": float(res.pvalues[col]),
    }


def fm_avg_daily_r2(panel, signal_cols, return_col, *, date_col="date"):
    """Mean across days of per-day OLS R^2 with an intercept.

    ``linearmodels.FamaMacBeth`` reports ``avg_rsquared`` from an
    intercept-free fit; that R^2 uses uncentered ss_tot and differs from
    the centered ss_tot convention used throughout the QF table bodies
    when the dependent variable has nonzero mean (e.g. ``log|D|``). This
    helper restores the intercept-included R^2 the FM-driven body cells
    in ``code/output/generate_table_bodies.py`` are pinned to.
    """
    cols = signal_cols if isinstance(signal_cols, (list, tuple)) else [signal_cols]
    r2s = []
    for _, g in panel.groupby(date_col):
        y = g[return_col].values
        X = g[cols].values
        res = ols_fit(y, X)
        ss_res = float(res.resid @ res.resid)
        y_dev = y - y.mean()
        ss_tot = float(y_dev @ y_dev)
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return float(np.mean(r2s)) if r2s else float("nan")


def fama_macbeth_univariate(df, signal_col, return_col, *,
                            min_stocks_per_day, row_filter=None,
                            entity_col="ticker"):
    """Day-standardized univariate Fama-MacBeth via ``linearmodels``.

    Thin wrapper over ``fit_fama_macbeth`` that preserves the historical
    return-dict shape: ``{coefficient, se, t_stat, p_value, n_days,
    avg_n_stocks}``. Returns ``None`` if no day passes the filter.

    ``entity_col`` defaults to ``"ticker"`` (the Korean panel entity).
    """
    res, panel = fit_fama_macbeth(
        df, [signal_col], return_col,
        min_stocks_per_day=min_stocks_per_day, row_filter=row_filter,
        entity_col=entity_col,
    )
    if res is None:
        return None
    out = _fm_stats(res, signal_col)
    n_per_day = panel.groupby("date").size()
    out["n_days"] = int(len(n_per_day))
    out["avg_n_stocks"] = float(n_per_day.mean())
    return out


def fama_macbeth_horse_race(df, signal_specs, return_col, *,
                            min_stocks_per_day, row_filter=None,
                            entity_col="ticker"):
    """Day-standardized multivariate Fama-MacBeth via ``linearmodels``.

    ``signal_specs`` is a list of ``(column_name, output_label)`` tuples
    in the desired output order. Returns
    ``{label: {coefficient, se, t_stat, p_value}, ..., n_days,
    avg_n_stocks}`` or ``None`` if no day passed the filters.

    ``entity_col`` defaults to ``"ticker"`` (the Korean panel entity).
    """
    cols = [c for c, _ in signal_specs]
    res, panel = fit_fama_macbeth(
        df, cols, return_col,
        min_stocks_per_day=min_stocks_per_day, row_filter=row_filter,
        entity_col=entity_col,
    )
    if res is None:
        return None
    out = {label: _fm_stats(res, c) for c, label in signal_specs}
    n_per_day = panel.groupby("date").size()
    out["n_days"] = int(len(n_per_day))
    out["avg_n_stocks"] = float(n_per_day.mean())
    return out


def pack_fm_spec_flat(spec, cols, *, return_horizon=None):
    """Translate one ``run_fm_triple`` spec into the flat-key shape.

    The per-spec JSON files in ``RESULTS_DIR`` (``signal_structure_*.json``,
    ``results/return_prediction_results.json``, etc.) consumed by
    ``output/_macro_emitters`` and ``output/test_assertions`` are pinned to
    ``coef_<col>``/``se_<col>``/``t_stat_<col>``/``p_value_<col>`` plus
    ``avg_r_squared``/``avg_n``/``total_n``/``n_dates``. Pass
    ``return_horizon`` to also emit the optional ``return_horizon`` key used
    by ``results/return_prediction_results.json``.
    """
    out = {}
    for c in cols:
        out[f'coef_{c}'] = spec[c]['coef']
        out[f'se_{c}'] = spec[c]['se']
        out[f't_stat_{c}'] = spec[c]['t_stat']
        out[f'p_value_{c}'] = spec[c]['p_value']
    out['avg_r_squared'] = spec['r_squared']
    out['avg_n'] = spec['avg_n']
    out['total_n'] = spec['total_n']
    out['n_dates'] = spec['n_dates']
    if return_horizon is not None:
        out['return_horizon'] = return_horizon
    return out


def pack_fm_spec_investor(spec, col, *, include_r_squared=True):
    """Translate one column of a ``run_fm_triple`` spec into the per-investor shape.

    Used by ``multi_investor_robustness`` and related per-investor
    scripts whose output is consumed by
    ``output/_macro_emitters._emit_robustness_investor_macros``. Returns
    ``{mean, t_stat, p_value, n_days}`` plus ``mean_R2`` (in percent) when
    ``include_r_squared=True``. ``None``-spec inputs are mapped to a
    NaN-filled dict of the same shape.
    """
    if spec is None:
        out = {'mean': float('nan'), 't_stat': float('nan'),
               'p_value': float('nan'), 'n_days': 0}
        if include_r_squared:
            out['mean_R2'] = float('nan')
        return out
    out = {
        'mean': spec[col]['coef'],
        't_stat': spec[col]['t_stat'],
        'p_value': spec[col]['p_value'],
        'n_days': spec['n_dates'],
    }
    if include_r_squared:
        out['mean_R2'] = spec['r_squared'] * 100.0
    return out


def fm_investor_signal_triple(df, investor_type, *, return_col,
                              min_stocks_per_day, entity_col="ticker"):
    """Per-investor FM triple (univariate MC / TV / horse race) on the
    canonical ``S_MC_{type}`` / ``S_TV_{type}`` / ``D_{type}`` schema.

    ``entity_col`` keys the panel entity dimension (default ``"ticker"``).

    Constructs the three column names from ``investor_type``, applies the
    nonzero-flow row filter on ``D_{investor_type}``, and routes through
    ``run_fm_triple``. Returns the per-investor JSON shape consumed by
    ``output/_macro_emitters._emit_robustness_investor_macros`` —
    ``{investor_type, n_days, univariate_MC, univariate_TV, horse_race_MC,
    horse_race_TV, horse_race_R2}`` — with R² in **percent**.
    """
    smc = f'S_MC_{investor_type}'
    stv = f'S_TV_{investor_type}'
    d_col = f'D_{investor_type}'
    triple = run_fm_triple(
        df, return_col=return_col, signal_mc=smc, signal_tv=stv,
        min_stocks_per_day=min_stocks_per_day,
        row_filter=lambda sub: sub[sub[d_col] != 0],
        entity_col=entity_col,
    )
    mc = pack_fm_spec_investor(triple['mc'], smc)
    return {
        'investor_type': investor_type,
        'n_days': mc['n_days'],
        'univariate_MC': mc,
        'univariate_TV': pack_fm_spec_investor(triple['tv'], stv),
        'horse_race_MC': pack_fm_spec_investor(triple['hr'], smc, include_r_squared=False),
        'horse_race_TV': pack_fm_spec_investor(triple['hr'], stv, include_r_squared=False),
        'horse_race_R2': (triple['hr']['r_squared'] * 100.0
                          if triple['hr'] is not None else float('nan')),
    }


def fit_fm_spec(df, cols, return_col, *, min_stocks_per_day,
                row_filter=None, entity_col="ticker"):
    """Fit one Fama-MacBeth spec and pack the canonical per-spec result dict.

    Runs ``fit_fama_macbeth`` on ``cols`` and returns

        {<col>: {'coef', 'se', 't_stat', 'p_value'},  # one per regressor
         'r_squared': float,   # intercept-included, via fm_avg_daily_r2
         'avg_n': float, 'total_n': int, 'n_dates': int}

    or ``None`` if no day survived ``prepare_fm_panel``. The per-spec building
    block ``run_fm_triple`` is built on.
    """
    res, panel = fit_fama_macbeth(
        df, cols, return_col,
        min_stocks_per_day=min_stocks_per_day, row_filter=row_filter,
        entity_col=entity_col,
    )
    if res is None:
        return None
    out = {c: {
        'coef': float(res.params[c]),
        'se': float(res.std_errors[c]),
        't_stat': float(res.tstats[c]),
        'p_value': float(res.pvalues[c]),
    } for c in cols}
    n_per_day = panel.groupby('date').size()
    out['r_squared'] = fm_avg_daily_r2(panel, cols, return_col)
    out['avg_n'] = float(n_per_day.mean())
    out['total_n'] = int(len(panel))
    out['n_dates'] = int(len(n_per_day))
    return out


def run_fm_triple(df, *, return_col, signal_mc, signal_tv,
                  min_stocks_per_day, row_filter=None, entity_col="ticker"):
    """Run the three Fama-MacBeth specs used throughout the QF paper.

    Fits ``return_col ~ signal_mc`` (univariate MC), ``~ signal_tv``
    (univariate TV), and ``~ signal_mc + signal_tv`` (horse race) on the
    same panel via ``fit_fama_macbeth``. Each spec is pre-processed
    independently by ``prepare_fm_panel`` (zero-mean within day,
    degenerate-day drop) so the three N counts can differ when the
    horse-race regressors are jointly degenerate on dates the univariate
    runs accept.

    ``entity_col`` keys the panel entity dimension (default ``"ticker"`` for
    the Korean panel).

    Returns ``{'mc': spec_dict, 'tv': spec_dict, 'hr': spec_dict}`` where
    each ``spec_dict`` is

        {
            <col>: {'coef', 'se', 't_stat', 'p_value'},  # one per regressor
            'r_squared': float,   # via fm_avg_daily_r2 (intercept-included)
            'avg_n': float, 'total_n': int, 'n_dates': int,
        }

    A spec is ``None`` when its panel was empty after pre-processing.
    """
    def _run(cols):
        return fit_fm_spec(df, cols, return_col,
                           min_stocks_per_day=min_stocks_per_day,
                           row_filter=row_filter, entity_col=entity_col)

    return {
        'mc': _run([signal_mc]),
        'tv': _run([signal_tv]),
        'hr': _run([signal_mc, signal_tv]),
    }
