#!/usr/bin/env python3
"""Size the foreign-investor leg the panel's definition excludes.

subsec:data builds foreign flow from 등록외국인 (registered foreigners), the KRX
category that carries the institutions holding a foreign investment registration.
The exchange also discloses 기타외국인 (other foreigners), the unregistered
residual, and the two together are the 외국인계 total a reader who has only seen
KRX headline foreign flow will assume was used. The restriction is therefore a
researcher degree of freedom on one of the two classes whose contrast
subsec:reversal_test estimates, and it needs a magnitude rather than an
assertion that the excluded leg is small.

This script measures that magnitude on the published panel. It reloads both legs
from the DataGuide export, restricts them to the stock-days the panel actually
estimates on, and reports what share of foreign trading the definition drops and
how often adding it back would change the sign of daily net foreign flow. The
sign share is the one that speaks to the signal: $S^{MC}$ and $S^{TV}$ are both
$D$ over a positive denominator, so a stock-day whose foreign net buy keeps its
sign under either definition keeps its position on the same side of the signal
cross-section.

The reloaded legs also settle a premise the sample restriction rests on. The
nonzero-flow restriction of subsec:appendix_panel_filters reads an exact zero
as a record that the class did not trade, rather than as buys and sells that
happened to balance.
That is a claim about the data and the two legs falsify it directly, since a
balanced day would show a zero net against a positive gross. The same test runs
on the capacity-bound class, whose zeros the restriction removes far more of:
its flow sums the pension and insurance sub-types, and their buy and sell legs
are in the export too.

Degrees of freedom
------------------
MEASURED  both foreign legs, read from the same DataGuide sheets and the same
          window the pipeline uses, and the pension and insurance legs,
          read by the pipeline's own sub-investor loader. Nothing here is
          fitted or chosen, and the script reports shares rather than
          estimating anything.
CHOSEN    the published panel as the population, rather than the raw export.
          The claim in sec:appendix_data is about the sample the paper
          estimates on, so the shares are taken over the stock-days that
          survive the subsec:data cuts.

BASELINE: 등록외국인 as the foreign class | obvious alternative: 외국인계, the
          registered plus other-foreigner total, which the same loader builds
          from two more columns of the same sheet | discharge: subsec:data,
          which states the restriction, and sec:appendix_data, which quotes
          the two shares this script measures.

Inputs
------
    results/full_daily_data_all_investors.parquet
    data/fnguide/raw/fnguide_investor_foreign-pe_20260219.xlsx
        (four 등록/기타외국인 sheets)
    the pension and insurance buy and sell sheets of
        data._fnguide.SUBINVESTOR_SOURCES, through the pipeline's cache

Output
------
    results/foreign_definition_share.json
"""

import pandas as pd
from shared_utils import END_DATE_MAIN, START_DATE_MAIN, print_header, save_json

from paths import DAILY_DATA_ALL_INVESTORS, RESULTS_DIR

RESULTS_PATH = str(RESULTS_DIR / "foreign_definition_share.json")

# Panel window, read from START_DATE_MAIN / END_DATE_MAIN. The export runs well
# past both ends, so the slice is applied on load rather than after the merge.
SAMPLE_START, SAMPLE_END = START_DATE_MAIN, END_DATE_MAIN

# The four sheets of the DataGuide export that carry the two foreign legs, in
# the same file data._fnguide.load_investor_flow reads. ``reg`` reproduces the panel's own
# D_foreign; ``oth`` is the leg the definition excludes.
FOREIGN_SHEETS = {
    "buy_reg":  "매수대금(등록외국인)",
    "sell_reg": "매도대금(등록외국인)",
    "buy_oth":  "매수대금(기타외국인)",
    "sell_oth": "매도대금(기타외국인)",
}

# A ticker-format mismatch or a renamed sheet would leave the merged 기타외국인
# columns empty, and an empty leg reports as a 0% share — the finding this
# script exists to produce, arrived at by failing to load the data. The
# registered leg is reloaded alongside it precisely so that failure is loud: it
# must reproduce the panel's own D_foreign to the won on every row.
MAX_REGISTERED_MISMATCH_KRW = 1.0

# The same guard on the capacity-bound legs: pension plus insurance must
# reproduce the panel's own D_capacity_bound to the won on every row.
MAX_CAPACITY_MISMATCH_KRW = 1.0


def load_foreign_legs():
    """Return a long ``(date, ticker, buy/sell x reg/oth)`` frame of both legs."""
    from data._fnguide import load_fnguide_sheet, melt_fnguide_wide  # noqa: PLC0415

    from paths import FNGUIDE_RAW_DIR  # noqa: PLC0415

    path = str(FNGUIDE_RAW_DIR / "fnguide_investor_foreign-pe_20260219.xlsx")
    legs = None
    for col, sheet in FOREIGN_SHEETS.items():
        print(f"  loading {sheet}", flush=True)
        wide = load_fnguide_sheet(path, sheet)
        part = melt_fnguide_wide(wide.loc[SAMPLE_START:SAMPLE_END], col)
        legs = part if legs is None else legs.merge(part, on=["date", "ticker"],
                                                    how="outer")
    # A stock-day absent from a sheet is a day that leg did not trade, which is
    # a zero rather than a missing value.
    return legs.fillna(0.0)


def measure(panel, legs):
    """Shares of foreign trading carried by the excluded 기타외국인 leg."""
    df = panel.merge(legs, on=["date", "ticker"], how="left")
    df[list(FOREIGN_SHEETS)] = df[list(FOREIGN_SHEETS)].fillna(0.0)

    d_reg = df["buy_reg"] - df["sell_reg"]
    d_oth = df["buy_oth"] - df["sell_oth"]
    d_tot = d_reg + d_oth

    mismatch = (df["D_foreign"] - d_reg).abs().max()
    if mismatch > MAX_REGISTERED_MISMATCH_KRW:
        raise ValueError(
            f"Reloaded 등록외국인 net buy disagrees with the panel's D_foreign by "
            f"up to {mismatch:,.0f} KRW. The panel is built from the same sheets, "
            f"so the two must be identical — a nonzero gap means the merge keyed "
            f"wrongly or a sheet was renamed, and the 기타외국인 share below would "
            f"then measure a failed load rather than a small leg."
        )

    gross_reg = float((df["buy_reg"] + df["sell_reg"]).sum())
    gross_oth = float((df["buy_oth"] + df["sell_oth"]).sum())

    # The premise behind the nonzero-flow restriction. A stock-day the class
    # traded on both sides to an exactly balancing won amount would carry a zero
    # net against a positive gross, and the reading in
    # subsec:appendix_panel_filters of an exact zero as non-participation
    # would then be false on those rows.
    zero_net = d_reg == 0
    balanced = zero_net & ((df["buy_reg"] + df["sell_reg"]) > 0)

    return {
        "n_stock_days": int(len(df)),
        "n_tickers": int(df["ticker"].nunique()),
        "n_days": int(df["date"].nunique()),
        "gross_registered_krw": gross_reg,
        "gross_other_krw": gross_oth,
        "gross_other_share_pct": 100.0 * gross_oth / (gross_reg + gross_oth),
        "net_other_share_pct": float(
            100.0 * d_oth.abs().sum() / d_tot.abs().sum()),
        "sign_flip_share_pct": float(100.0 * (d_reg * d_tot < 0).mean()),
        "days_traded_share_pct": float(
            100.0 * ((df["buy_oth"] + df["sell_oth"]) > 0).mean()),
        "max_registered_mismatch_krw": float(mismatch),
        "n_zero_net": int(zero_net.sum()),
        "n_zero_net_with_gross": int(balanced.sum()),
    }


def load_capacity_bound_legs():
    """Return ``(date, ticker, net_buy_value, gross)`` summed over the
    capacity-bound sub-types."""
    from data._fnguide import load_subinvestor_flow  # noqa: PLC0415
    from shared_utils import CAPACITY_BOUND_COMPONENTS  # noqa: PLC0415

    from paths import CACHE_DIR, FNGUIDE_RAW_DIR  # noqa: PLC0415

    # The pipeline's own call, all eight sub-types over the same window, so the
    # load reads the cache the panel was built from.
    flow = load_subinvestor_flow(SAMPLE_START, SAMPLE_END,
                                 raw_dir=FNGUIDE_RAW_DIR, cache_dir=CACHE_DIR)
    cap = flow[flow["subinvestor_type"].isin(CAPACITY_BOUND_COMPONENTS)]
    cap = cap.assign(gross=cap["buy_value"] + cap["sell_value"])
    return (cap.groupby(["date", "ticker"], as_index=False)
               [["net_buy_value", "gross"]].sum())


def measure_capacity_bound_zeros(panel, legs):
    """Count the capacity-bound zero-net stock-days, and those with trading."""
    if panel["D_capacity_bound"].isna().any():
        raise ValueError("D_capacity_bound has missing rows, which the zero "
                         "test below would read as neither zero nor traded.")
    df = panel.merge(legs, on=["date", "ticker"], how="left")
    # A stock-day absent from the sub-investor sheets is one that neither
    # sub-type traded, as in load_foreign_legs.
    df[["net_buy_value", "gross"]] = df[["net_buy_value", "gross"]].fillna(0.0)

    mismatch = (df["D_capacity_bound"] - df["net_buy_value"]).abs().max()
    if mismatch > MAX_CAPACITY_MISMATCH_KRW:
        raise ValueError(
            f"Reloaded pension plus insurance net buy disagrees with the panel's "
            f"D_capacity_bound by up to {mismatch:,.0f} KRW. The panel sums the "
            f"same legs, so a nonzero gap means the merge keyed wrongly, and the "
            f"counts below would then measure a failed load."
        )

    # A zero net against a positive gross is a day on which the two sub-types
    # traded, whether each balanced its own buys and sells or the two offset.
    zero_net = df["D_capacity_bound"] == 0
    return {
        "max_net_mismatch_krw": float(mismatch),
        "n_zero_net": int(zero_net.sum()),
        "n_zero_net_with_gross": int((zero_net & (df["gross"] > 0)).sum()),
    }


def main():
    """Entry point: size the excluded foreign leg on the published panel."""
    print_header("Foreign definition — the excluded 기타외국인 leg", level="banner")
    panel = pd.read_parquet(DAILY_DATA_ALL_INVESTORS,
                            columns=["date", "ticker", "D_foreign", "D_capacity_bound"])
    print(f"Panel: {len(panel):,} stock-days", flush=True)

    out = measure(panel[["date", "ticker", "D_foreign"]], load_foreign_legs())
    out["capacity_bound"] = measure_capacity_bound_zeros(
        panel[["date", "ticker", "D_capacity_bound"]], load_capacity_bound_legs())
    print(f"\n기타외국인 share of foreign gross value traded: "
          f"{out['gross_other_share_pct']:.3f}%")
    print(f"  of aggregate |net flow|:                     "
          f"{out['net_other_share_pct']:.3f}%")
    print(f"  stock-days whose foreign net-buy sign flips: "
          f"{out['sign_flip_share_pct']:.3f}%")
    print(f"\nZero-net registered stock-days: {out['n_zero_net']:,}, of which "
          f"{out['n_zero_net_with_gross']:,} report positive gross value traded")
    cap = out["capacity_bound"]
    print(f"Zero-net capacity-bound stock-days: {cap['n_zero_net']:,}, of which "
          f"{cap['n_zero_net_with_gross']:,} report positive gross value traded")

    save_json({"config": {"window": [SAMPLE_START, SAMPLE_END],
                          "sheets": FOREIGN_SHEETS},
               "panel": out}, RESULTS_PATH)
    return out


if __name__ == "__main__":
    main()
