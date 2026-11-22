#!/usr/bin/env python3
"""Round-trip-execution versus net-accumulation decomposition of investor flow.

Backs the regime assignment in the manuscript (main tab:directionality_kr in
subsec:regime_assignment): the daily ``log|D|`` scaling regression measures the
carrier of *gross-flow magnitude*, which coincides with the matched filter of
a capacity-regime class only when that class's round-trip execution shares the
signal's market-cap carrier, as Korean capacity-bound execution does. Where a
class instead executes through volume (Korean foreign), the daily scaling
points at the
volume carrier even though the *directional* component of the flow is
cap-scaled. This module measures that decoupling without consulting returns, in
Korea.

For each investor class we split per-stock daily net flow into two magnitudes
and ask which carrier each scales with cross-sectionally:

    gross flow  = mean_t |D_t|          (execution magnitude)
    net accum   = |mean_t D_t|          (directional drift / persistence)

    log(gross flow)  = a + beta_M log M_bar + beta_V log V_bar
    log(net accum)   = a + beta_M log M_bar + beta_V log V_bar

with one observation per stock (stock-mean carriers M_bar, V_bar) and
heteroskedasticity-robust HC1 standard errors. The directionality ratio
``net accum / gross flow`` (median across stocks) is the scale-free, return-
free measure of how persistent a class's accumulation is: round-trip execution
dominates daily magnitude for every class, and capacity-bound mandates net out
a larger share of gross flow than the round-tripping foreign classes.

Both quantities are read in currency units (tab:directionality_kr). The directionality ratio is a ratio of two
time-aggregations of D and never touches M or V. What fixes the space is that
eq:general_dgp defines D as a currency flow: the ratio is a statistic of that
sequence, and dividing the traded print out first would measure the
directionality of D_t/p_t instead. The matched filters are invariant to the
choice, since S^MC = D/M and S^TV = D/V divide D by a carrier formed at the
same day's price and close_raw cancels in both, so this convention moves the
ratio and nothing else in the paper. The share-space ratio ships beside the
published one as ``directionality_median_share_space``.

The return-free discriminator is the directionality ratio (a ratio of two flow
magnitudes in the same units). Capacity-bound flow is ~2--3x more directional
than foreign, and on the non-overlapping pre-period holdout as well; this is
the return-free signature corroborating the a-priori capacity-bound
assignment.

The ratio's numerator averages over a stock's nonzero-flow days and its
denominator does not, so the ratio falls with the day count on its own and the
two classes do not trade equally often. code/empirical/signal_structure/directionality_frequency_null.py
measures the ordering that day count alone delivers (subsec:regime_assignment,
subsec:appendix_directionality_frequency); it imports ``_per_stock``,
``_aggregate`` and ``MIN_DAYS`` from here, so its null is built on this module's
sample.

Degrees of freedom
------------------
MIN_DAYS        CHOSEN — 250 nonzero-flow days, approximately one trading
                year, so each of a stock's gross and net magnitudes is averaged
                over enough days for the class's purchases and sales to offset.
SE_TYPE         CHOSEN — HC1. The regression carries one observation per stock,
                so its errors are cross-sectionally heteroskedastic in stock
                size and homoskedastic standard errors would understate them.
carriers        CHOSEN — ``_joint`` regresses on stock-mean log M and log V
                together, the same pair and the same functional form as the
                published daily scaling regression, so the two regressions
                differ in the dependent variable alone.

BASELINE: the directionality ratio as the return-free discriminator | obvious
alternative: a referee asks whether the ratio just ranks classes by how often
they trade, since its numerator averages over nonzero-flow days and its
denominator does not | discharge: code/empirical/signal_structure/directionality_frequency_null.py measures
the ordering day count alone delivers, on this module's own sample.

Output: ../results/flow_persistence_decomposition.json
"""

import numpy as np
import pandas as pd

from shared_utils import ols_fit, save_json
from paths import (
    DAILY_DATA_ALL_INVESTORS,
    DAILY_DATA_PRE2020,
    DAILY_DATA_TAIWAN,
    RESULTS_DIR,
)

OUTPUT_PATH = RESULTS_DIR / "flow_persistence_decomposition.json"

# A stock's purchases and sales need enough days to offset before its net
# accumulation can be told from its gross flow; 250 nonzero-flow days is about
# one trading year (subsec:regime_assignment).
MIN_DAYS = 250
SE_TYPE = "HC1"  # one obs per stock -> heteroskedastic-robust cross-section


def _joint(y, cap, vol):
    """HC1 joint log-log regression log(y) ~ log(cap) + log(vol).

    Returns the cap (M) and volume (V) elasticities with their t-statistics
    and the dominant carrier (larger |beta|). ``ols_fit`` prepends the
    intercept; HC1 robust SEs are appropriate for a one-obs-per-stock
    cross-section (no date dimension to cluster on).
    """
    res = ols_fit(np.log(y), np.column_stack([np.log(cap), np.log(vol)]))
    rob = res.get_robustcov_results(cov_type=SE_TYPE)
    b_m, b_v = float(rob.params[1]), float(rob.params[2])
    t_m, t_v = float(rob.tvalues[1]), float(rob.tvalues[2])
    return {
        "beta_M": b_m, "t_M": t_m,
        "beta_V": b_v, "t_V": t_v,
        "dominant": "M" if abs(b_m) > abs(b_v) else "V",
    }


def _decompose(agg, label, elasticities=None):
    """Gross-flow + net-accum blocks and directionality from a per-stock
    aggregate frame (columns: gross_flow, net_accum, cap_bar, vol_bar).

    ``elasticities`` supplies a second frame for the two carrier regressions,
    for the market where the ratio and the regressions are read in different
    measurement spaces (see the module docstring). It defaults to
    ``agg``, which is the Korean case.
    """
    e = agg if elasticities is None else elasticities
    directionality = float((agg["net_accum"] / agg["gross_flow"]).median())
    block = {
        "label": label,
        "n_stocks": int(len(agg)),
        "directionality_median": directionality,
        "gross_flow": _joint(e["gross_flow"], e["cap_bar"], e["vol_bar"]),
        "net_accum": _joint(e["net_accum"], e["cap_bar"], e["vol_bar"]),
    }
    print(f"  {label}: {len(agg):,} stocks, directionality={directionality:.3f}; "
          f"gross->{block['gross_flow']['dominant']}, net->{block['net_accum']['dominant']}")
    return block


def _per_stock(df, d_col, cap_col, vol_col):
    """Per-stock gross-flow/net-accum magnitudes + mean carriers, restricted to
    strictly positive magnitudes/carriers (no day-count filter applied yet)."""
    g = df.groupby("ticker")
    agg = pd.DataFrame({
        "n_days":      g[d_col].size(),
        "gross_flow":  g[d_col].apply(lambda v: v.abs().mean()),
        "net_accum":   g[d_col].apply(lambda v: abs(v.mean())),
        "cap_bar":     g[cap_col].mean(),
        "vol_bar":     g[vol_col].mean(),
    })
    return agg[(agg["gross_flow"] > 0) & (agg["net_accum"] > 0) &
              (agg["cap_bar"] > 0) & (agg["vol_bar"] > 0)]


def _aggregate(per_stock, min_days=MIN_DAYS):
    """``_per_stock`` output restricted to stocks with >= ``min_days``
    nonzero-flow days."""
    return per_stock[per_stock["n_days"] >= min_days]


# ---------------------------------------------------------------------------
# Korea — currency units (KRW). D is independently disclosed; no close artifact.
# ---------------------------------------------------------------------------

def _korea(path, classes):
    cols = ["ticker", "close_raw", "volume", "M", "V"] + [c for _, c, _ in classes]
    df = pd.read_parquet(path, columns=cols)
    out = {"space": "value"}
    for key, d_col, label in classes:
        sub = df[(df[d_col].abs() > 0) & (df["M"] > 0) & (df["V"] > 0) &
                 (df["close_raw"] > 0) & (df["volume"] > 0)][
            ["ticker", d_col, "close_raw", "volume", "M", "V"]].copy()
        out[key] = _decompose(_aggregate(_per_stock(sub, d_col, "M", "V")), label)
    return out



# ---------------------------------------------------------------------------
# Taiwan -- the rows the Taiwanese ratio and its null are read off, in both
# measurement spaces. This module does not decompose Taiwan itself;
# ``directionality_frequency_null`` imports these two functions, so the
# Taiwanese cells are built by the same row rule as the Korean ones.
# ---------------------------------------------------------------------------

def taiwan_panel(classes):
    """The Taiwan columns both readings need, in one load."""
    return pd.read_parquet(
        DAILY_DATA_TAIWAN,
        columns=["stock_id", "close_raw", "volume", "M", "V"]
        + [c for _, c, _ in classes],
    ).rename(columns={"stock_id": "ticker"})


def taiwan_rows(df, d_col):
    """One class's ``taiwan_panel`` rows, carrying both measurement spaces.

    The rows are that class's nonzero-flow days on which every carrier either
    reading uses is defined, so the value-space and the share-space ratio are
    read off one set of days and differ only in whether the traded print scales
    each day's magnitude. The per-stock universes can still differ by a name:
    ``_per_stock`` keeps only stocks with strictly positive net accumulation,
    and a stock whose share flow cancels exactly over the window need not have
    its value flow cancel at a price that moves.

    close_raw, not close: both share-space terms below invert a level that was
    formed at the traded price (D = net shares x close_raw, M = close_raw x
    shares), so the back-adjusted close would not cancel and would return a
    share count rescaled by the stock's event history.
    """
    sub = df[(df[d_col].abs() > 0) & (df["M"] > 0) & (df["V"] > 0) &
             (df["close_raw"] > 0) & (df["volume"] > 0)].copy()
    sub["net_shares"] = sub[d_col] / sub["close_raw"]  # D / close_raw
    sub["shares_out"] = sub["M"] / sub["close_raw"]    # M / close_raw (cap carrier)
    return sub

def main():
    print("Korea (currency units), main window:")
    korea = _korea(DAILY_DATA_ALL_INVESTORS, [
        ("capacity_bound", "D_capacity_bound", "Capacity-Bound Institutions"),
        ("foreign",        "D_foreign",        "Foreign"),
    ])
    print("Korea (currency units), 2015-2019 pre-period:")
    korea_pre = _korea(DAILY_DATA_PRE2020, [
        ("capacity_bound", "D_capacity_bound", "Capacity-Bound Institutions"),
        ("foreign",        "D_foreign",        "Foreign"),
    ])

    save_json(
        {
            "config": {"min_days": MIN_DAYS, "se_type": SE_TYPE},
            "korea": korea,
            "korea_preperiod": {"window": "2015-2019", **korea_pre},
        },
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
