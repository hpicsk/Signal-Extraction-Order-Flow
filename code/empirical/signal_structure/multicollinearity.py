#!/usr/bin/env python3
"""QF Paper - Collinearity of the two filters (capacity-bound).

Computes the daily cross-sectional correlation and variance inflation factor
of ``S_MC`` and ``S_TV`` on the Capacity-Bound
Institutional panel (pension + insurance subset; the canonical ``D`` in
the panel loaded via ``load_domestic_panel`` is ``D_capacity_bound``).
The ``--regime domestic`` CLI flag selects this panel; it is *not* the
KRX 기관계 aggregate. Output JSON (``results/multicollinearity_test_results.json``)
is consumed by ``code/output/generate_table_bodies.py``.

Degrees of freedom
------------------
MEASURED  the two diagnostics themselves, on the daily cross-sections that
          the horse race of tab:investor_heterogeneity runs on. Nothing here
          is fitted.
CHOSEN    ``MIN_STOCKS_DIAGNOSTIC = 10``, looser than the Fama-MacBeth floor
          ``MIN_STOCKS_PER_DAY_DEFAULT`` because these two statistics are
          descriptive and the floor that gates an inferential estimate is
          stricter than the one that gates a reported correlation.
BASELINE: each diagnostic's mean over the daily cross-sections | obvious
          alternative: a referee asks for the worst day rather than the
          mean | discharge: run. The output carries each diagnostic's daily
          maximum, and the largest daily variance inflation factor is 2.43.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from shared_utils import (
    MC_SEED,
    cross_sectional_zscore,
    load_all_investors_panel,
    load_domestic_panel,
    print_header,
    save_json,
)
from paths import RESULTS_DIR

# Minimum cross-section for a day to enter the descriptive daily correlation
# and VIF diagnostics (looser than the Fama-MacBeth floor
# MIN_STOCKS_PER_DAY_DEFAULT, which gates the horse race).
MIN_STOCKS_DIAGNOSTIC = 10


@dataclass
class Config:
    name: str
    primary: str          # first signal column
    secondary: str        # second signal column
    dataset: str          # "domestic" -> load_domestic_panel; "all_investors" -> raw load
    output_json: str


def _load(cfg: Config) -> pd.DataFrame:
    print_header(f"LOADING DATA ({cfg.name.upper()})")
    if cfg.dataset == "domestic":
        df = load_domestic_panel()
    elif cfg.dataset == "all_investors":
        df = load_all_investors_panel(
            required_cols=[cfg.primary, cfg.secondary, "R_t1"],
        )
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    return df


def test_correlation(df, cfg: Config):
    """Daily cross-sectional Pearson correlation between primary and secondary signals."""
    print_header(f"TEST 1: CORRELATION ({cfg.primary}, {cfg.secondary})")
    daily = []
    for _, sub in df.groupby("date"):
        sub = sub.dropna(subset=[cfg.primary, cfg.secondary])
        if len(sub) > MIN_STOCKS_DIAGNOSTIC:
            corr = sub[cfg.primary].corr(sub[cfg.secondary])
            if not np.isnan(corr):
                daily.append(corr)
    arr = np.asarray(daily)
    out = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "n_days": int(len(arr)),
    }
    print(f"Mean: {out['mean']:.4f}  Std: {out['std']:.4f}  "
          f"[{out['min']:.4f}, {out['max']:.4f}]")
    return out


def test_vif(df, cfg: Config):
    """Daily VIF for the two-regressor design via ``statsmodels.variance_inflation_factor``."""
    print_header("TEST 2: VARIANCE INFLATION FACTOR")
    daily = []
    for _, sub in df.groupby("date"):
        sub = sub.dropna(subset=[cfg.primary, cfg.secondary])
        if len(sub) <= MIN_STOCKS_DIAGNOSTIC:
            continue
        z_a = cross_sectional_zscore(sub[cfg.primary])
        z_b = cross_sectional_zscore(sub[cfg.secondary])
        if z_a is None or z_b is None:
            continue
        # statsmodels expects a design matrix whose first column is a constant
        # for centred-R^2 semantics; the VIF for either standardised signal is
        # symmetric in the two-regressor case.
        X = np.column_stack([np.ones(len(z_a)), z_a.values, z_b.values])
        try:
            vif = float(variance_inflation_factor(X, 1))
        except (np.linalg.LinAlgError, ValueError):
            continue
        if np.isfinite(vif):
            daily.append(vif)
    arr = np.asarray(daily)
    out = {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "n_days": int(len(arr)),
    }
    print(f"Mean VIF: {out['mean']:.4f}")
    return out


CONFIGS = {
    "domestic": Config(
        name="domestic",
        primary="S_MC",
        secondary="S_TV",
        dataset="domestic",
        output_json="multicollinearity_test_results.json",
    ),
}


def run(cfg: Config):
    """Run the two collinearity diagnostics for one investor config; write JSON."""
    print_header(f"QF PAPER - MULTICOLLINEARITY TESTS ({cfg.name.upper()})", level='banner')
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    df = _load(cfg)
    results = {
        "name": cfg.name,
        "timestamp": datetime.now().isoformat(),
        "correlation": test_correlation(df, cfg),
        "vif": test_vif(df, cfg),
    }

    save_json(results, RESULTS_DIR / cfg.output_json)

    return results


def main():
    """CLI entry: run the domestic multicollinearity tests."""
    np.random.seed(MC_SEED)
    run(CONFIGS["domestic"])


if __name__ == "__main__":
    main()
