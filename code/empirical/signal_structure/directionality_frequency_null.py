#!/usr/bin/env python3
"""Trading frequency as a mechanical source of the directionality ratio
(subsec:regime_assignment, sec:taiwan,
subsec:appendix_directionality_frequency).

**The gap this closes.** The regime assignment of subsec:regime_assignment is
made before any return is examined, and the statistic that corroborates it is
the directionality ratio, net accumulation over gross flow,

    D_net / D_gross = |mean_t D_t| / mean_t |D_t|,

taken as a median over the stocks a class trades on at least ``MIN_DAYS``
days. The ratio is read as persistence: a class that builds a directional
position keeps a larger share of its own gross flow as net accumulation than
a class that makes round trips. But the numerator is a mean over the class's own trading days and
the denominator is not, so the ratio falls with the day count on its own. For
flow with no directional component at all, independent across days, the
numerator shrinks as the square root of the day count while the denominator
does not, and the ratio approaches ``1/sqrt(n)`` rather than zero. The classes
being ordered do not trade equally often, and in both markets the capacity
class is the rarer one: capacity-bound flow is disclosed on 34.3% of the Korean
panel's stock-days against 99.2% for foreign flow
(subsec:appendix_panel_filters), and SITC flow
on 15.3% of the Taiwan panel's against 87.1%. The class the assignment places
in the capacity regime is therefore also the class with the smaller per-stock
day count, and part of its higher ratio is arithmetic.

The published ordering is confounded with trading frequency in both markets,
and the confound runs in the direction that would manufacture it. It is the
sharper threat in Taiwan, where sec:taiwan rests the corroboration on the
flow's direction, and where the disclosure gap between the two classes is the
wider of the two markets. This module measures how much of each market's ordering
frequency alone delivers.

**The null.** For each stock we keep the observed daily magnitudes and randomise
only the direction, drawing independent signs

    D_t  ->  eps_t |D_t|,     eps_t = +/- 1 with equal probability,

and recomputing the class's median ratio. The draw preserves each stock's day
count exactly, its gross flow exactly (the denominator is a function of |D_t|
alone), and the shape of its magnitude distribution exactly, including the
heavy right tail that a Gaussian null would understate. The only quantity it
destroys is the sign structure that a persistent directional position creates.
The median ratio across draws is therefore what the class's trading frequency
and execution magnitudes produce with no accumulation, and the observed ratio
divided by it is a frequency-free directionality statistic.

The null is built in currency units in both markets, the units the published ratio
is measured in (tab:directionality_kr), because
eq:general_dgp defines D as a currency flow and the ratio is a statistic of
that sequence. The choice is not innocuous for the null: the traded print moves
day to day, so in Taiwan the magnitude sequence the signs are flipped over
differs between value and share space and so does the null it produces. Taiwan
is therefore run twice, once in each space on the same rows, and the share-space
cells are the disclosed robustness reading. The construction is imported from
``flow_persistence_decomposition`` rather than rebuilt here, so each null is
read off the stocks and days of the ratio it is compared against.

Degrees of freedom:

    MEASURED  the observed ratios, by ``flow_persistence_decomposition``,
              whose ``_per_stock``, ``_aggregate``, ``MIN_DAYS`` and Taiwan row
              construction are imported here, so the published ratio and its
              null are read off the same stocks and the same days.
    CHOSEN    the sign-flip null rather than a parametric draw. It conditions on
              the observed magnitudes, so no distributional family is asserted
              and the null carries the tails as measured.
    CHOSEN    ``N_DRAWS``, which fixes the resolution of the randomisation
              p-value at 2/(N_DRAWS + 1) and no other quantity. It is set by
              the one cell that does not land on that floor rather than by the
              Korean cells, which do.
    BASELINE: the sign-flip null | obvious alternative: a referee asks why not
              the closed-form Gaussian null 1/sqrt(n), which needs no simulation
              | discharge: ``null_gaussian_median`` below reports it beside the
              sign-flip null in every cell, and the appendix states the gap.

Inputs
------
    results/full_daily_data_all_investors.parquet
    results/full_daily_data_all_investors_pre2020.parquet
    results/full_daily_data_taiwan_all_investors.parquet

Output
------
    results/directionality_frequency_null.json
"""

import numpy as np
import pandas as pd

from shared_utils import (
    END_DATE_MAIN,
    END_DATE_PRE,
    MC_SEED,
    START_DATE_MAIN,
    START_DATE_PRE,
    print_header,
    save_json,
)
from paths import DAILY_DATA_ALL_INVESTORS, DAILY_DATA_PRE2020, RESULTS_DIR
from empirical.signal_structure.flow_persistence_decomposition import (
    MIN_DAYS,
    _aggregate,
    _per_stock,
    taiwan_panel,
    taiwan_rows,
)

OUTPUT_PATH = RESULTS_DIR / "directionality_frequency_null.json"

# Sign-flip draws per cell. The two-sided randomisation p-value resolves to
# 2/(N_DRAWS + 1), which is the only quantity this constant fixes: the excess
# ratios are governed by the null mean, whose own error is smaller by a further
# factor of sqrt(N_DRAWS) and is already negligible at a tenth of this count.
# The count is set by the one cell whose p-value is not at the resolution floor
# (Taiwan SITC, subsec:appendix_directionality_frequency). A tail probability
# near 0.03 carries a Monte Carlo standard error of sqrt(0.03 * 0.97 / N_DRAWS)
# on its own count, which at 1,000 draws is a third of the distance from that
# cell's p-value to the 5% level and leaves the printed second decimal unstable
# across draws; at 10,000 it is a tenth of it. ``p_two_sided_mc_se`` below
# reports the figure per cell rather than leaving it to be recomputed. Every
# Korean cell lands on the floor and would be served by a tenth of this count.
N_DRAWS = 10_000

# The two classes of the headline reversal in each market
# (subsec:return_prediction, sec:taiwan), in the order the directionality
# tables report them. The pair being ordered is the capacity class against
# foreign flow in both markets.
CLASSES_KR = (
    ("capacity_bound", "D_capacity_bound", "Capacity-Bound Institutions"),
    ("foreign", "D_foreign", "Foreign"),
)
CLASSES_TW = (
    ("institutional", "D_institutional", "Investment Trust (SITC)"),
    ("foreign", "D_foreign", "Foreign"),
)


def _sign_flip_null(magnitudes, rng, n_draws=N_DRAWS):
    """Median directionality ratio under ``n_draws`` independent sign flips.

    ``magnitudes`` is one array of |D_t| per stock. Each draw replaces every
    day's sign with an independent Rademacher variable and returns the median
    across stocks of |mean_t eps_t |D_t|| / mean_t |D_t|, so the returned array
    is the null distribution of the published statistic, not of a per-stock one.
    """
    ratios = np.empty((len(magnitudes), n_draws))
    for i, a in enumerate(magnitudes):
        signs = rng.integers(0, 2, size=(a.size, n_draws)).astype(np.float64) * 2.0 - 1.0
        # a @ signs sums eps_t |D_t| over days; dividing both sides by the day
        # count turns the sums into means and leaves the ratio unchanged.
        ratios[i] = np.abs(a @ signs) / (a.size * a.mean())
    return np.median(ratios, axis=0)


def _cell(sub, d_col, cap_col, vol_col, label, rng):
    """Observed ratio, sign-flip null, and Gaussian reference for one class.

    ``sub`` is that class's nonzero-flow rows as the published decomposition
    filters them, in that market's measurement space; ``d_col`` is the flow
    magnitude column of that space.
    """
    agg = _aggregate(_per_stock(sub, d_col, cap_col, vol_col))
    observed = float((agg["net_accum"] / agg["gross_flow"]).median())

    # Restrict once, so the null and the published ratio are read off the same
    # stocks and the same days.
    kept = set(agg.index)
    sub = sub[sub["ticker"].isin(kept)]
    magnitudes = [v.abs().to_numpy() for _, v in sub.groupby("ticker")[d_col]]
    null = _sign_flip_null(magnitudes, rng)
    null_mean, null_sd = float(null.mean()), float(null.std(ddof=1))

    # Randomisation p-value in the direction the class's observed ratio falls,
    # doubled for a two-sided statement. The +1 counts the observed value
    # itself, so the p-value is never zero.
    tail = (null >= observed).sum() if observed >= null_mean else (null <= observed).sum()
    p_two_sided = min(1.0, 2.0 * (tail + 1) / (N_DRAWS + 1))
    # The p-value is itself a Monte Carlo estimate: the tail count is binomial
    # over the draws, so this is the standard error of the number reported.
    q = tail / N_DRAWS
    p_mc_se = 2.0 * float(np.sqrt(q * (1.0 - q) / N_DRAWS))

    block = {
        "label": label,
        "n_stocks": int(len(agg)),
        "median_days": float(agg["n_days"].median()),
        "observed": observed,
        "null_mean": null_mean,
        "null_sd": null_sd,
        # The Gaussian null at the same day counts, median of 1/sqrt(n). It
        # needs no simulation and is the obvious alternative to the sign flip.
        "null_gaussian_median": float(np.median(1.0 / np.sqrt(agg["n_days"]))),
        "excess": observed / null_mean,
        "p_two_sided": p_two_sided,
        "p_two_sided_mc_se": p_mc_se,
    }
    print(f"  {label:<28} n={block['n_stocks']:>5,} stocks, median {block['median_days']:.0f} days; "
          f"observed={observed:.4f} null={null_mean:.4f} (sd {null_sd:.4f}, "
          f"Gaussian {block['null_gaussian_median']:.4f}) excess={block['excess']:.2f} "
          f"p={p_two_sided:.4f} (mc se {p_mc_se:.4f})")
    return block


def _ordering(out, cap_key, for_key):
    """The two observed ratios divided, beside the same ratio under the null.

    The published ordering is the ratio of the two observed ratios. The same
    ratio under the null is what trading frequency and magnitude shape deliver
    on their own, so the two together price the confound. ``excess`` divides
    them, which measures each class against its own day count and is the
    ordering subsec:regime_assignment rests the corroboration on.
    """
    cap, for_ = out[cap_key], out[for_key]
    block = {
        "observed": cap["observed"] / for_["observed"],
        "null": cap["null_mean"] / for_["null_mean"],
        "excess": cap["excess"] / for_["excess"],
    }
    print(f"  ordering: observed {block['observed']:.2f}x, "
          f"null {block['null']:.2f}x, excess {block['excess']:.2f}x")
    return block


def run_korea(path, window, rng):
    """Every Korean class in one estimation window, plus the ordering ratios.

    Currency units (KRW): the KRX net-buy value D is independently disclosed, so
    no price term is shared with the carriers and none has to be divided out.
    """
    cols = ["ticker", "M", "V"] + [c for _, c, _ in CLASSES_KR]
    df = pd.read_parquet(path, columns=cols)
    out = {"window": window, "space": "value"}
    for key, d_col, label in CLASSES_KR:
        sub = df[(df[d_col].abs() > 0) & (df["M"] > 0) & (df["V"] > 0)][
            ["ticker", d_col, "M", "V"]].copy()
        out[key] = _cell(sub, d_col, "M", "V", label, rng)
    out["ordering"] = _ordering(out, "capacity_bound", "foreign")
    return out


def run_taiwan(window, rng, space):
    """Every Taiwan class in one measurement space, plus the ordering ratios.

    ``space`` selects which magnitude column the ratio and its null are built
    from. Currency units are the published reading, the currency flow
    eq:general_dgp defines; share space divides the traded print out of the
    flow first and is the disclosed robustness reading. ``taiwan_rows`` is
    imported rather than repeated, so both spaces run on the rows the published
    ratio was measured on and differ in that column alone.
    """
    df = taiwan_panel(CLASSES_TW)
    out = {"window": window, "space": space}
    for key, d_col, label in CLASSES_TW:
        sub = taiwan_rows(df, d_col)
        cols = ((d_col, "M", "V") if space == "value"
                else ("net_shares", "shares_out", "volume"))
        out[key] = _cell(sub, *cols, label, rng)
    out["ordering"] = _ordering(out, "institutional", "foreign")
    return out


def main():
    rng = np.random.default_rng(MC_SEED)
    print_header("Directionality ratio against a trading-frequency null")
    print(f"{N_DRAWS} sign-flip draws, MIN_DAYS = {MIN_DAYS}")
    # The window a result is stamped with is the one its panel was built on,
    # read from shared_utils.config rather than typed here. The Taiwan panel
    # takes the same window (sec:taiwan), and its pipeline reads the same two
    # constants, so it is stamped from them too rather than from a literal that
    # goes stale the day either panel moves.
    main_window = f"{START_DATE_MAIN[:4]}-{END_DATE_MAIN[:4]}"
    pre_window = f"{START_DATE_PRE[:4]}-{END_DATE_PRE[:4]}"
    print(f"Korea (currency units), {main_window}:")
    korea = run_korea(DAILY_DATA_ALL_INVESTORS, main_window, rng)
    print("Korea pre-period (currency units):")
    korea_pre = run_korea(DAILY_DATA_PRE2020, pre_window, rng)
    print(f"Taiwan (currency units), {main_window}:")
    taiwan = run_taiwan(main_window, rng, "value")
    print(f"Taiwan (share space, the disclosed robustness reading), {main_window}:")
    taiwan_share = run_taiwan(main_window, rng, "share")

    save_json(
        {
            "config": {"n_draws": N_DRAWS, "min_days": MIN_DAYS, "seed": MC_SEED},
            "korea": korea,
            "korea_preperiod": korea_pre,
            "taiwan": taiwan,
            "taiwan_share_space": taiwan_share,
        },
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
