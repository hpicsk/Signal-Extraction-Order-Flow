#!/usr/bin/env python3
"""
KOSPI 200 falsification for foreign-flow $S^{TV}$ predictability.

Splits foreign-investor stock-days by KOSPI 200 membership and re-runs
Fama-MacBeth return prediction (univariate $S^{TV}$ and the horse race, the
fits tab:foreign_kospi200_split prints) on four panels:

    Full       — all foreign-flow stock-days (baseline; matches tab:investor_heterogeneity, Panel B)
    KOSPI 200  — stock-days where the issuer is a KOSPI 200 constituent
    Non-K200   — stock-days where the issuer is outside KOSPI 200
    Non-K200,  — the same subset restricted to the largest size quintile.
    largest      Membership tracks firm size, so without this cell neither
                 side of the split can separate an index effect from a size
                 effect.

KOSPI 200 ≈ MSCI Korea Standard universe by overlap; non-KOSPI 200 stocks
receive ≈ 0 passive foreign rebalancing flow. If foreign $S^{TV}$
predictability survives (or strengthens) in the non-KOSPI 200 subset, the
Informed-Executor reading cannot be explained by passive index tracking.

Inputs
------
    results/full_daily_data_all_investors.parquet
    paths.KOSPI200_PANEL  (data/registry/kospi200_index_panel_daily.parquet)

Output
------
    results/foreign_kospi200_split_results.json

Degrees of freedom
------------------
CHOSEN    KOSPI 200 membership as the partition. It is the passive-flow axis
          the objection names, and membership is point-in-time from the KRX
          panel rather than reconstructed.
CHOSEN    the ``Non-K200, largest`` cell. Membership tracks firm size, so
          without it neither side of the split separates an index effect from
          a size effect.
CHOSEN    ``RETURN_HORIZON`` and ``MIN_STOCKS_PER_DAY``, inherited from
          subsec:return_prediction and ``shared_utils.config``.
MEASURED  every coefficient and $t$-statistic, on each subset's own
          nonzero-flow stock-days.
FITTED    nothing.

BASELINE: KOSPI 200 membership as the proxy for passively tracked names |
          obvious alternative: a referee says foreign passive money tracks
          MSCI Korea rather than KOSPI 200, so the split does not isolate
          passive flow | discharge: subsec:appendix_kospi200, where the
          KOSPI 200 is the broader passively held set, so splitting by KOSPI 200
          membership dilutes a passive-tracking effect inside the in-index subset rather than
          amplifying it, and the largest-quintile cell separates the index
          axis from size.

"""


import numpy as np
import pandas as pd
from shared_utils import (
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    fama_macbeth_horse_race as _fm_horse_race,
    fama_macbeth_univariate as _fm_univariate,
    print_header,
    save_json,
)

from paths import DAILY_DATA_ALL_INVESTORS, KOSPI200_PANEL as _KOSPI200_PANEL, RESULTS_DIR

INPUT_FILE = str(DAILY_DATA_ALL_INVESTORS)
KOSPI200_PANEL = str(_KOSPI200_PANEL)
RESULTS_PATH = str(RESULTS_DIR / "foreign_kospi200_split_results.json")

MIN_STOCKS_PER_DAY = MIN_STOCKS_PER_DAY_DEFAULT
RETURN_HORIZON = "R_t1"
SIGNAL_TV = "S_TV_foreign"
SIGNAL_MC = "S_MC_foreign"
FLOW_COL = "D_foreign"

# The size cell the liquidity-stratified robustness already reports foreign flow
# in (sec:appendix_liquidity_stratified), reused here so the size control is a cell the
# paper already defines rather than a new stratification.
LARGEST_QUINTILE = "Q5_Large"

np.random.seed(MC_SEED)


def load_panel():
    """Load the all-investors panel restricted to nonzero foreign flow; add 6-digit ticker."""
    df = pd.read_parquet(INPUT_FILE)
    df = df[df[FLOW_COL] != 0].copy()
    # Strip the FnGuide 'A' prefix to align with KRX 6-digit ticker codes.
    df["ticker_6"] = df["ticker"].str.lstrip("A")
    return df


def load_kospi200_membership():
    """Load the daily KOSPI 200 constituent panel as (date, ticker_6, is_kospi200=True)."""
    panel = pd.read_parquet(KOSPI200_PANEL)
    panel = panel[panel["index"] == "코스피 200"][["date", "ticker"]].copy()
    panel["is_kospi200"] = True
    panel = panel.rename(columns={"ticker": "ticker_6"})
    return panel


def attach_membership(df, k200):
    """Left-join membership flag onto the panel; non-KOSPI 200 rows get is_kospi200=False."""
    out = df.merge(k200, on=["date", "ticker_6"], how="left")
    out["is_kospi200"] = out["is_kospi200"].fillna(False).astype(bool)
    return out


def fama_macbeth_univariate(df, signal_col):
    """Convenience wrapper: shared FM univariate bound to RETURN_HORIZON / MIN_STOCKS_PER_DAY."""
    return _fm_univariate(df, signal_col, RETURN_HORIZON,
                          min_stocks_per_day=MIN_STOCKS_PER_DAY)


def fama_macbeth_horse_race(df, signal_a, signal_b):
    """Convenience wrapper: shared FM horse race with labels signal_TV / signal_MC."""
    return _fm_horse_race(
        df,
        [(signal_a, "signal_TV"), (signal_b, "signal_MC")],
        RETURN_HORIZON,
        min_stocks_per_day=MIN_STOCKS_PER_DAY,
    )


def run_panel(df, label):
    """Run univariate-TV + horse-race FM on a sub-panel; return result dict + print summary."""
    print(f"\n[{label}]  rows = {len(df):,}  unique stocks = {df['ticker_6'].nunique():,}  "
          f"unique dates = {df['date'].nunique():,}")
    res = {
        "panel_size": int(len(df)),
        "n_unique_stocks": int(df["ticker_6"].nunique()),
        "n_unique_dates": int(df["date"].nunique()),
        "univariate_TV": fama_macbeth_univariate(df, SIGNAL_TV),
        "horse_race": fama_macbeth_horse_race(df, SIGNAL_TV, SIGNAL_MC),
    }
    if res["univariate_TV"]:
        print(f"  univariate S_TV : t = {res['univariate_TV']['t_stat']:+6.2f}   "
              f"coef = {res['univariate_TV']['coefficient']:+.5f}")
    if res["horse_race"]:
        hr = res["horse_race"]
        print(f"  horse-race S_TV : t = {hr['signal_TV']['t_stat']:+6.2f}   "
              f"coef = {hr['signal_TV']['coefficient']:+.5f}")
        print(f"  horse-race S_MC : t = {hr['signal_MC']['t_stat']:+6.2f}   "
              f"coef = {hr['signal_MC']['coefficient']:+.5f}")
    return res


def main():
    """Entry point: load + split panel by KOSPI 200 membership and run FM on each sub-panel."""
    print_header("KOSPI 200 falsification for foreign-flow $S^{TV}$ predictability",
                 width=72)

    df = load_panel()
    print(f"Loaded foreign-flow panel: {len(df):,} stock-days, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    k200 = load_kospi200_membership()
    print(f"KOSPI 200 daily membership: {len(k200):,} stock-day flags")

    df = attach_membership(df, k200)

    daily_k200_count = (
        df[df["is_kospi200"]].groupby("date")["ticker_6"].nunique().describe()
    )
    print("\nDaily KOSPI 200 stock count (within foreign-flow panel):")
    print(daily_k200_count.to_string())

    full = df
    in_k200 = df[df["is_kospi200"]].copy()
    out_k200 = df[~df["is_kospi200"]].copy()

    results = {
        "config": {
            "return_horizon": RETURN_HORIZON,
            "min_stocks_per_day": MIN_STOCKS_PER_DAY,
            "kospi200_proxy_for": "MSCI Korea Standard passively-held universe",
            "ticker_normalisation": "FnGuide 'A'-prefix stripped to KRX 6-digit",
        },
        "full": run_panel(full, "Full foreign panel"),
        "kospi200": run_panel(in_k200, "KOSPI 200 subset"),
        "non_kospi200": run_panel(out_k200, "Non-KOSPI 200 subset"),
        "non_kospi200_largest": run_panel(
            out_k200[out_k200["size_quintile"] == LARGEST_QUINTILE].copy(),
            f"Non-KOSPI 200 subset, {LARGEST_QUINTILE}"),
    }

    save_json(results, RESULTS_PATH)

    full_t = results["full"]["univariate_TV"]["t_stat"]
    in_t = results["kospi200"]["univariate_TV"]["t_stat"]
    out_t = results["non_kospi200"]["univariate_TV"]["t_stat"]
    print("\nFalsification summary (univariate S_TV t-stats):")
    print(f"  Full        t = {full_t:+6.2f}")
    print(f"  KOSPI 200   t = {in_t:+6.2f}")
    print(f"  Non-K200    t = {out_t:+6.2f}")
    if abs(out_t) >= abs(in_t):
        print("  -> Non-K200 t ≥ K200 t in magnitude: passive-rebalancing channel is "
              "insufficient to explain foreign predictability.")
    else:
        print("  -> Non-K200 t < K200 t: passive-rebalancing channel non-negligible; "
              "discuss in subsubsec:kospi200_falsification.")


if __name__ == "__main__":
    main()
