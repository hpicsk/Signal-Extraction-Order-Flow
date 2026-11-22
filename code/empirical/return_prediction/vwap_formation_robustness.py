#!/usr/bin/env python3
r"""VWAP formation-price robustness (stale-closing-print test).

Backs subsec:vwap_formation, the formation-price row of
tab:robustness_summary, and the closing-print clause of qf_paper.tex
sec:robustness. The headline matched filter S^MC = D/M is
formed on day t with M = close_t x shares, so the single last-trade close C_t
enters the signal denominator. On thinly traded names that last print can be
stale, so the next-day predictability could be read as a stale-close artifact
rather than information.

This test re-forms the market cap on the day's volume-weighted average price
VWAP_t = V_t / volume_t (traded value over share volume) in place of the
closing price. It rebuilds the cap as M^vwap = M x (VWAP_t / close^raw_t),
forms S^MC,vwap = D / M^vwap, and re-runs the horse race against the SAME
standard close-to-close next-day return R_{t+1}. The identity
M^vwap = M x (VWAP/close^raw) is exact because ``market_cap`` is the closing
print times shares, and VWAP and close^raw are two prices of the same session:
the ratio swaps one traded price for another and changes nothing else. It is
``close_raw`` and not the panel's total-return ``close`` for exactly that
reason — dividing by the adjusted close would scale each stock by its own
accumulated distribution history instead.

Keeping the return close-to-close is deliberate. Forming BOTH the denominator
and the return base on VWAP_t would reintroduce a shared-price term (VWAP_t
appearing in the signal and the return base), the very mechanical channel the
test must avoid. Here VWAP_t enters only the signal and close_t only the return
base, so the idiosyncratic price noise of each is separated. This test removes
the closing price from the signal denominator; the skip-a-day test
(code/empirical/return_prediction/skip_day_return_prediction.py) removes day-t price from the return base.
Together they strip C_t from both places it appears. The traded-value signal
S^TV = D/V is unchanged because V is realised traded value and contains no
closing price.

If the capacity-bound S^MC dominance and the across-class reversal
survive on VWAP formation prices, the result is not a closing-price artifact.

Output: ../results/vwap_formation_robustness_results.json

Degrees of freedom
------------------
CHOSEN    VWAP as the alternative formation price. It is the traded price the
          objection names, and $V_t/\mathrm{volume}_t$ is the only one the
          panel carries besides the close.
CHOSEN    ``close_raw`` and not the adjusted close in the rebuild ratio,
          argued above: the adjusted close would scale each stock by its own
          distribution history.
CHOSEN    the close-to-close ``RETURN_COL`` for both arms, argued above:
          moving the return base to VWAP reintroduces the shared-price term
          the test exists to remove.
MEASURED  every cell of ``CELLS``, on the same stock-days as the published
          horse race.
FITTED    nothing.

BASELINE: the closing price as the headline formation price | obvious
          alternative: a referee says a stale last trade on a thin name puts
          the next-day predictability in the denominator rather than in the
          flow | discharge: run. This script is that run, and
          subsec:vwap_formation reports where the re-formed arms land.

"""

import numpy as np
import pandas as pd

from shared_utils import (
    MC_SEED,
    MIN_STOCKS_PER_DAY_DEFAULT,
    run_fm_triple,
    save_json,
)
from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR

OUTPUT_PATH = RESULTS_DIR / "vwap_formation_robustness_results.json"

# The two Korean matched-filter cells of Table investor_heterogeneity:
# Capacity-Bound Inst. matched on S^MC, foreign flow matched on S^TV.
CELLS = (
    ("capacity_bound", "MC", "Korea Capacity-Bound Inst."),
    ("foreign", "TV", "Korea Foreign"),
)
RETURN_COL = "R_t1"   # standard close-to-close [t, t+1] for both arms

np.random.seed(MC_SEED)


def _add_vwap_cap(df):
    """Attach VWAP and the VWAP-implied market cap ``M_vwap``.

    ``VWAP_t = V_t / volume_t`` is the volume-weighted average price. Both legs
    are marcap's published totals for the session, so the VWAP is the price
    actually paid on average, in the units of ``close_raw`` — the closing price,
    not the panel's total-return ``close``. The ratio below therefore takes
    ``close_raw``: a same-session price ratio has to be read on one basis, and
    dividing by the total-return close would scale each stock by its own
    accumulated distribution history instead. ``M_vwap = M * (VWAP/close_raw)``
    is the market cap implied by the average price rather than the closing
    print, formed on the print ``market_cap`` itself is formed on.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["VWAP"] = np.where(df["volume"] > 0, df["V"] / df["volume"], np.nan)
    df["M_vwap"] = df["M"] * (df["VWAP"] / df["close_raw"])
    return df


def _matched_mismatched(triple, matched_arm, smc_col, stv_col):
    """Pull matched/mismatched t-statistics from a ``run_fm_triple`` result.

    ``matched_arm`` is 'MC' (matched = S^MC, mismatched = S^TV) or 'TV'. The
    univariate t is read from the matched arm's own univariate spec; the
    horse-race t's are read from the joint spec for each arm.
    """
    matched_col, mismatched_col = (
        (smc_col, stv_col) if matched_arm == "MC" else (stv_col, smc_col)
    )
    uni_spec = triple["mc"] if matched_arm == "MC" else triple["tv"]
    hr = triple["hr"]
    return {
        "uni_t": uni_spec[matched_col]["t_stat"],
        "hr_matched_t": hr[matched_col]["t_stat"],
        "hr_mismatched_t": hr[mismatched_col]["t_stat"],
        "n_days": hr["n_dates"],
    }


def _cell(df, investor_type, matched_arm, label):
    """Close-formation vs VWAP-formation matched-filter t-statistics for one cell.

    Both arms run through the identical ``run_fm_triple`` path on the same
    nonzero-flow panel and the same close-to-close return; the only difference
    is the market-cap denominator (``S_MC_{type}`` on close vs
    ``S_MC_vwap_{type}`` on ``M_vwap``). The matched ``S^TV`` column is shared
    by both arms because traded value carries no closing price.
    """
    smc = f"S_MC_{investor_type}"
    stv = f"S_TV_{investor_type}"
    smc_vwap = f"S_MC_vwap_{investor_type}"
    d_col = f"D_{investor_type}"

    df = df.copy()
    df[smc_vwap] = (df[d_col] / df["M_vwap"]).replace([np.inf, -np.inf], np.nan)
    row_filter = lambda sub: sub[sub[d_col] != 0]  # noqa: E731

    close_arm = run_fm_triple(
        df, return_col=RETURN_COL, signal_mc=smc, signal_tv=stv,
        min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT,
        row_filter=row_filter, entity_col="ticker",
    )
    vwap_arm = run_fm_triple(
        df, return_col=RETURN_COL, signal_mc=smc_vwap, signal_tv=stv,
        min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT,
        row_filter=row_filter, entity_col="ticker",
    )

    out = {
        "label": label,
        "matched_filter": f"S^{matched_arm}",
        "matched_arm": matched_arm,
        "close": _matched_mismatched(close_arm, matched_arm, smc, stv),
        "vwap": _matched_mismatched(vwap_arm, matched_arm, smc_vwap, stv),
    }
    print(f"  {label:<28s} matched=S^{matched_arm}  "
          f"matched HR t: {out['close']['hr_matched_t']:+6.2f} (close) -> "
          f"{out['vwap']['hr_matched_t']:+6.2f} (VWAP)   "
          f"mismatched HR t: {out['close']['hr_mismatched_t']:+6.2f} -> "
          f"{out['vwap']['hr_mismatched_t']:+6.2f}")
    return out


def main():
    print("VWAP formation-price robustness, Korea (main window):")
    kr = pd.read_parquet(
        DAILY_DATA_ALL_INVESTORS,
        columns=["date", "ticker", "close_raw", "volume", "V", "M", RETURN_COL,
                 "D_capacity_bound", "S_MC_capacity_bound", "S_TV_capacity_bound",
                 "D_foreign", "S_MC_foreign", "S_TV_foreign"],
    )
    kr = _add_vwap_cap(kr)

    korea = {inv: _cell(kr, inv, arm, label) for inv, arm, label in CELLS}

    save_json(
        {"config": {"return_col": RETURN_COL,
                    "min_stocks_per_day": MIN_STOCKS_PER_DAY_DEFAULT},
         "korea": korea},
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
