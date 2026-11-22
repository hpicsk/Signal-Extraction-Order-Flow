#!/usr/bin/env python3
r"""
Taiwan daily panel for matched-filter replication.

Assembles, for each stock-day in START_DATE_MAIN -- END_DATE_MAIN:
    D_foreign, D_institutional      [TWD net buy]
    M, V, turnover, close, close_raw, R_t1, R_t5, R_t20
    is_ex_date, is_cap_red          [disclosed corporate action on this session]

Investor mapping (FinMind / TWSE):
    SITC (Investment_Trust)            -> D_institutional   (capacity regime)
    Foreign_Investor                    -> D_foreign         (volume regime)
    Dealer_self + Dealer_Hedging
        + Foreign_Dealer_Self           -> D_dealer          (auxiliary)
    individual = -(disclosed sum)       -> D_individual      (derived)

SITC (Securities Investment Trust Companies) is the structurally
capacity-bound Taiwanese mandate — benchmark-tracking domestic funds
whose rebalancing pattern aligns with the capacity regime, analogous to
Korea's pension + insurance subset. It is therefore the canonical
``D`` for the Taiwan replication (``panel["D"] = panel["D_institutional"]``
below), deliberately distinguished from Korea's retail-distributed 투신
(see the Korean Investment Trust vs. Taiwan SITC labelling-caveat
subsection of sec:appendix_subinvestor_decomposition).

``close`` is the vendor's total-return adjusted close, ``TaiwanStockPriceAdj``
(the 還原股價 series): structural corporate actions (splits, 無償配股, 現增, 減資)
are removed *and* the cash dividend is reinvested, so a distribution leaves the
return series flat rather than recording a loss the holder did not take. That is
the same convention the Korean panel carries in its own ``close`` — FnGuide's
수정주가(현금배당포함) — so one forward-return definition spans both markets
(sec:taiwan).

Total return is the only convention available on this side. TWSE/TPEx publish a
single *fused* reference price per 除權息 covering cash, 無償配股 and 現增
together, so a price-return series would need a split of that fused step with
nothing to validate the split against, and the vendor serves no price-return
variant at any tier. The Korean panel could carry either and carries this one,
which is what makes the two markets comparable rather than merely parallel.

``close_raw`` keeps the unadjusted traded print, because three quantities are
levels rather than returns and must use the price that actually traded:
D_TWD = (buy - sell) * close_raw, M = close_raw * NumberOfSharesIssued (joined
from shares parquet on the session's own date), and the share-space terms
in flow_persistence_decomposition. V = Trading_money (direct in TWD).

Sessions the stock did not trade carry ``close == 0`` in the raw endpoint and
the last traded price in the adjusted one. They leave the panel on the raw
print, through the ``M > 0`` filter below, so no row enters at a price nobody
could have transacted at. A session the adjusted endpoint does not serve at all
leaves the panel too, counted rather than filled.

``is_ex_date`` / ``is_cap_red`` flag the sessions carrying a disclosed 除權息 or
減資, read from the vendor's two filing endpoints (``div_result/`` and the
market-wide capital-reduction reference prices), so the corporate-action
robustness check in twn_analysis can drop the affected forward windows by event
rather than by return magnitude.

The universe is the vendor's live listing registry. A company the exchange has
delisted can fall out of it, so the panel carries a survivorship exposure the
Korean panel does not; ``results/twn_universe_coverage.json`` measures it against the
vendor's own delisting record and sec:taiwan states it.

The price endpoints also serve the sessions a stock traded on the emerging board
(興櫃) before it listed, and those are not TWSE or TPEx sessions. The shares
endpoint carries a stock while it is listed, usually from the session before its
first listed session, so a session enters the panel only if the shares endpoint
carries it and the stock's previous session (sec:taiwan).
``results/twn_universe_coverage.json`` counts the sessions this removes.

Output: ../results/full_daily_data_taiwan_all_investors.parquet

Degrees of freedom
------------------
CHOSEN    the window ``START_DATE_MAIN``--``END_DATE_MAIN``, inherited from
          ``shared_utils.config`` so the Taiwan panel spans the same years as
          the Korean one (sec:taiwan).
CHOSEN    SITC as the Taiwanese capacity mandate and ``panel["D"]``. The
          mapping is read off the mandate, as sec:taiwan argues, and no
          return is consulted to make it.
CHOSEN    the universe: four-digit common-stock codes on TWSE and TPEx, as the
          vendor's live registry lists them, on each session that the shares
          endpoint carries when it also carries the stock's previous session
          (above). The exclusions are the security types ``data._finmind``
          names; what they and the registry's own coverage remove is measured
          into ``results/twn_universe_coverage.json`` rather than asserted.
CHOSEN    the dealer grouping ``DEALER_TYPES``, which follows the disclosure
          rather than a judgement: the foreign broker-dealer line is
          proprietary trading and is grouped with the two domestic dealer
          lines.
CHOSEN    the total-return adjusted close, the only convention Taiwan admits,
          because the exchanges publish one fused reference price per
          distribution (above).
MEASURED  every panel quantity, and every count sec:taiwan quotes,
          on the sessions the vendor's own records cover.
FITTED    nothing.

BASELINE: FinMind as the disclosure source | obvious alternative: a referee
          says a third-party API is not the exchange, so a replication built
          on it tests the vendor's retention rather than the market |
          discharge: sec:taiwan, which sizes what the live registry drops
          (``twnSurvivorStocks`` delisted names it no longer lists, together
          ``twnSurvivorSessions`` stock-days) instead of asserting
          completeness, and whose
          corporate-action sensitivity re-estimates every return cell on the
          windows the vendor's own event disclosure flags.

"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data._finmind import delisted_in_window, emerging_exits, load_prices, load_universe
from shared_utils import (
    END_DATE_MAIN,
    START_DATE_MAIN,
    add_investor_signals,
    compute_forward_returns,
)
from paths import (
    DAILY_DATA_TAIWAN,
    FINMIND_DIV_RESULT_DIR,
    FINMIND_INSTFLOW_DIR,
    FINMIND_OHLCV_DIR,
    FINMIND_PRICE_ADJ_DIR,
    FINMIND_SHARES_DIR,
    RESULTS_DIR,
    TWN_CAP_RED,
    TWN_DELISTING,
    TWN_STOCK_INFO,
)

INSTFLOW_DIR = Path(FINMIND_INSTFLOW_DIR)
OHLCV_DIR = Path(FINMIND_OHLCV_DIR)
PRICE_ADJ_DIR = Path(FINMIND_PRICE_ADJ_DIR)
SHARES_DIR = Path(FINMIND_SHARES_DIR)
DIV_RESULT_DIR = Path(FINMIND_DIV_RESULT_DIR)

OUTPUT_PATH = DAILY_DATA_TAIWAN
COVERAGE_PATH = RESULTS_DIR / "twn_universe_coverage.json"
# Taiwan sample window mirrors the Korean panel (shared_utils.config) for the
# "symmetry with the Korean panel" claim in sec:taiwan.
START_DATE = START_DATE_MAIN
END_DATE = END_DATE_MAIN

INVESTOR_MAP = {
    "Investment_Trust": "D_institutional",
    "Foreign_Investor": "D_foreign",
}
DEALER_TYPES = {"Dealer_self", "Dealer_Hedging", "Foreign_Dealer_Self"}
# The single aggregated proprietary line the exchange publishes in place of the
# three above; see the guard in process_stock for why its presence excludes a
# stock rather than raising.
AGGREGATE_DEALER = "Dealer"
_AGGREGATE_DEALER_STOCKS = []
# Sessions the stock did not trade, accumulated across stocks so main() can
# report what the panel refused. The raw endpoint prints them at zero and the
# adjusted one fills them with the last traded price, so the count is a
# property of the traded print and is taken before the ``M > 0`` filter that
# removes them. sec:taiwan states the rule: a panel that kept them would carry
# a price nobody could have transacted at.
_NO_TRADE_ROWS = 0
# In-window OHLCV sessions the share-count rule in process_stock removes,
# accumulated for the coverage record.
_UNLISTED_ROWS = 0

_CAP_RED_DATES = None
_EMERGING_EXITS = None


def _cap_red_dates():
    """Every stock's 減資 dates, read once. Keyed by stock_id.

    The capital-reduction reference prices are one market-wide table, so
    reading it inside the per-stock loop would re-read the whole market once
    per stock.
    """
    global _CAP_RED_DATES
    if _CAP_RED_DATES is None:
        c = pd.read_parquet(TWN_CAP_RED, columns=["stock_id", "date"])
        c["date"] = pd.to_datetime(c["date"])
        _CAP_RED_DATES = {str(k): set(v) for k, v in
                          c.groupby(c["stock_id"].astype(str))["date"]}
    return _CAP_RED_DATES


def _emerging_exits():
    """Every stock's emerging-board retirement date, read once. Keyed by stock_id."""
    global _EMERGING_EXITS
    if _EMERGING_EXITS is None:
        _EMERGING_EXITS = emerging_exits(TWN_STOCK_INFO)
    return _EMERGING_EXITS


def _place_on_session(filed, sessions):
    """Mark, for each session, whether a filing lands on it.

    A filing is realised on the first session on or after its filed date. Most
    fall on a session and mark themselves; the rest were filed for a date the
    exchange did not open — a typhoon closure or a suspension — and the step is
    taken when trading resumes. Matching on the exact date instead leaves those
    unmarked, and an unmarked step is one the corporate-action sensitivity check
    below cannot drop. How many fall that way is a property of the exchange's
    calendar in the window, so ``twn_analysis`` counts the flagged rows rather
    than this docstring.
    """
    out = np.zeros(len(sessions), dtype=bool)
    if not len(filed):
        return out
    idx = np.searchsorted(sessions, np.sort(np.asarray(filed, dtype="datetime64[ns]")),
                          side="left")
    out[idx[idx < len(sessions)]] = True
    return out


def _event_flags(stock_id, dates):
    """``(is_ex_date, is_cap_red)`` boolean arrays over ``dates``.

    Both flags are read from the vendor's filing endpoints — ``div_result/``
    for 除權息 and the capital-reduction reference prices for 減資. Every filed
    date counts: this flag answers whether the session could carry a step, and
    a filing too broken to price is still a filing.
    """
    sessions = dates.to_numpy().astype("datetime64[ns]")
    div_path = DIV_RESULT_DIR / f"{stock_id}.parquet"
    ex = []
    if div_path.exists():
        d = pd.read_parquet(div_path)
        if len(d):
            ex = pd.to_datetime(d["date"]).to_numpy()
    red = sorted(_cap_red_dates().get(str(stock_id), set()))
    return (_place_on_session(ex, sessions),
            _place_on_session(np.array(red, dtype="datetime64[ns]"), sessions))


def process_stock(stock_id):
    """Build a per-stock daily frame for the requested window. Returns
    DataFrame or None if data is insufficient."""
    inst_path = INSTFLOW_DIR / f"{stock_id}.parquet"
    shar_path = SHARES_DIR / f"{stock_id}.parquet"
    if not (inst_path.exists() and shar_path.exists()):
        return None

    ohlc = load_prices(stock_id, OHLCV_DIR, PRICE_ADJ_DIR, START_DATE, END_DATE)
    if ohlc is None or ohlc.empty:
        return None
    ohlc["is_ex_date"], ohlc["is_cap_red"] = _event_flags(stock_id, ohlc["date"])

    inst = pd.read_parquet(inst_path)
    shar = pd.read_parquet(shar_path)
    if inst.empty or shar.empty:
        return None

    inst["date"] = pd.to_datetime(inst["date"])
    shar["date"] = pd.to_datetime(shar["date"])
    inst = inst[(inst["date"] >= START_DATE) & (inst["date"] <= END_DATE)]
    if inst.empty:
        return None

    # A renamed or withdrawn FinMind category would silently vanish from the
    # aggregations below and be swept into the derived retail residual, which
    # is defined as minus the disclosed sum. Fail loudly instead, mirroring
    # the guard in code/data/pre_period_pipeline.py.
    #
    # One shortfall is a disclosure format rather than a missing category. On
    # the emerging board and in a listing's first months the exchange publishes
    # proprietary trading as a single ``Dealer`` line instead of the three the
    # panel's D_dealer is summed from. A stock whose in-window disclosure never
    # uses the three lines is excluded, and this is the one place that decision
    # is made. The exclusion is safe only for a stock that the registry keeps
    # on the emerging board past the window, which never listed inside it, so
    # any other such stock raises. ``results/twn_universe_coverage.json`` lists the
    # stocks excluded here.
    # Anything else missing is the failure this guard was written for and
    # still raises.
    seen = set(inst["name"].unique())
    expected = set(INVESTOR_MAP) | DEALER_TYPES
    if not expected <= seen:
        if DEALER_TYPES <= (expected - seen) and AGGREGATE_DEALER in seen:
            exit_date = _emerging_exits().get(str(stock_id))
            if exit_date is None or exit_date <= pd.Timestamp(END_DATE):
                raise ValueError(
                    f"{stock_id}: in-window proprietary flow is disclosed on the "
                    f"one-line form only, but the registry does not keep the "
                    f"stock on the emerging board past {END_DATE} (retired "
                    f"{exit_date}), so excluding it would drop a TWSE or TPEx "
                    f"listing")
            _AGGREGATE_DEALER_STOCKS.append(str(stock_id))
            return None
        raise ValueError(
            f"{stock_id}: FinMind investor categories in-window are {sorted(seen)}, "
            f"missing {sorted(expected - seen)}. The derived individual "
            f"residual assumes the disclosed categories are exhaustive.")

    # Pivot instflow to wide investor-type columns (signed shares: buy - sell)
    inst["net_shares"] = inst["buy"] - inst["sell"]

    def aggregate_type(df, names):
        sub = df[df["name"].isin(names)].groupby("date")["net_shares"].sum()
        return sub

    pieces = {
        col: aggregate_type(inst, [name])
        for name, col in INVESTOR_MAP.items()
    }
    pieces["D_dealer"] = aggregate_type(inst, list(DEALER_TYPES))

    flow_wide = pd.concat(pieces, axis=1).fillna(0).reset_index()
    flow_wide.columns.name = None

    # Merge with OHLCV. The adjusted series takes the name ``close`` so the
    # shared forward-return helper reads the same column in both markets; the
    # traded print survives as ``close_raw`` for the level uses below.
    ohlc = ohlc[["date", "close", "adj_close_tr", "Trading_Volume",
                 "Trading_money", "is_ex_date", "is_cap_red"]].rename(
        columns={"close": "close_raw", "adj_close_tr": "close",
                 "Trading_Volume": "volume", "Trading_money": "V"}
    )
    df = ohlc.merge(flow_wide, on="date", how="left").fillna({
        "D_institutional": 0, "D_foreign": 0, "D_dealer": 0,
    })

    # Convert net shares to TWD at the price that actually traded, not the
    # back-adjusted one: D is a realised value, so it carries no factor.
    for col in ["D_institutional", "D_foreign", "D_dealer"]:
        df[col] = df[col].astype(float) * df["close_raw"].astype(float)

    # Implied retail flow: zero-sum across categories (this assumes the disclosed
    # flows are exhaustive; in TWSE the disclosed five categories sum to 0 by
    # construction, so D_individual = -(sum of disclosed five)).
    df["D_individual"] = -(df["D_institutional"] + df["D_foreign"] + df["D_dealer"])

    # The panel holds TWSE and TPEx sessions only (sec:taiwan). The shares
    # endpoint carries a stock while it is listed, usually from the session
    # before its first listed one, so a session is kept only if the endpoint
    # carries it and the stock's previous session. The previous session is read
    # off the stock's whole price history, so the window's first session is
    # judged against the one before the window. The event flags above were
    # placed on the full session list, so a filing dated inside the emerging
    # spell does not land on the first listed session.
    sessions = pd.to_datetime(pd.read_parquet(
        OHLCV_DIR / f"{stock_id}.parquet", columns=["date"])["date"])
    sessions = sessions.sort_values().reset_index(drop=True)
    carried = sessions.isin(shar["date"])
    listed = df["date"].isin(sessions[carried & carried.shift(fill_value=True)])
    global _UNLISTED_ROWS
    _UNLISTED_ROWS += int((~listed).sum())
    df = df[listed].merge(shar[["date", "NumberOfSharesIssued"]], on="date",
                          how="left", validate="many_to_one")
    # Shares outstanding is a contemporaneous count, so the cap it forms takes
    # the contemporaneous price. Pairing it with the back-adjusted close would
    # rescale the cap by the stock's whole event history. The filter also
    # removes the sessions the stock did not trade, whose raw print is zero.
    df["M"] = df["close_raw"].astype(float) * df["NumberOfSharesIssued"].astype(float)
    global _NO_TRADE_ROWS
    _NO_TRADE_ROWS += int((df["close_raw"] <= 0).sum())
    df = df[df["M"] > 0]

    if df.empty:
        return None

    df["turnover"] = df["V"] / df["M"]
    df["stock_id"] = stock_id

    # Forward returns are computed once on the assembled panel in main() via
    # the shared compute_forward_returns helper (group-aware over stock_id).
    keep = ["stock_id", "date", "close", "close_raw", "volume", "M", "V",
            "turnover", "is_ex_date", "is_cap_red",
            "D_institutional", "D_foreign", "D_dealer", "D_individual"]
    return df[keep]


def in_window_traded_value(stock_id):
    """In-window traded value of a stock the panel does not keep, or None if
    it has no in-window OHLCV either. Feeds the coverage record of what the
    construction drops."""
    path = OHLCV_DIR / f"{stock_id}.parquet"
    if not path.exists():
        return None
    o = pd.read_parquet(path)
    if o.empty:
        return None
    o["date"] = pd.to_datetime(o["date"])
    o = o[(o["date"] >= START_DATE) & (o["date"] <= END_DATE)]
    if o.empty:
        return None
    return float(o["Trading_money"].sum())


def main():
    universe = load_universe(TWN_STOCK_INFO)
    print(f"Universe size (four-digit, live registry): {len(universe):,}")

    rows = []
    failed = 0
    dropped_with_ohlcv, dropped_value = 0, 0.0
    for i, sid in enumerate(universe):
        df = process_stock(sid)
        if df is None or df.empty:
            failed += 1
            # A stock with in-window OHLCV that process_stock declined traded
            # inside the window without entering the panel. Count it and its
            # traded value into the coverage record. Most of this count is the
            # one-line Dealer disclosure rule in process_stock, which excludes
            # only a stock on the emerging board throughout the window.
            value = in_window_traded_value(sid)
            if value is not None:
                dropped_with_ohlcv += 1
                dropped_value += value
            continue
        rows.append(df)
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1:,}/{len(universe):,}  "
                  f"rows so far = {sum(len(r) for r in rows):,}")

    print(f"Successful: {len(rows):,}   Empty/missing: {failed:,}")
    if not rows:
        print("No data assembled.")
        sys.exit(1)

    panel = pd.concat(rows, ignore_index=True)

    # A session the adjusted endpoint does not serve carries no adjusted close.
    # No fallback is substituted: the affected rows leave the panel.
    no_px = panel["close"].isna()
    if no_px.any():
        print(f"\nUncovered sessions: dropped {int(no_px.sum()):,} of "
              f"{len(panel):,} stock-days ({100 * no_px.mean():.3f}%) across "
              f"{panel.loc[no_px, 'stock_id'].nunique():,} stocks with no "
              f"total-return adjusted close")
        panel = panel[~no_px].reset_index(drop=True)

    # Forward returns: sort by stock so the shared group-aware helper computes
    # per-stock close-to-close returns, matching the Korean panels' definition.
    # R_t (contemporaneous) is unused in the Taiwan replication, so drop it.
    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)
    compute_forward_returns(panel, group_col="stock_id")
    panel = panel.drop(columns="R_t")

    # Matched/mismatched signals S_MC = D/M, S_TV = D/V for each investor type
    add_investor_signals(panel, ["institutional", "foreign", "dealer", "individual"])

    # Aliases: institutional flow is the primary "domestic" capacity flow
    panel["D"] = panel["D_institutional"]
    panel["S_MC"] = panel["S_MC_institutional"]
    panel["S_TV"] = panel["S_TV_institutional"]

    # Drop rows with non-finite signals (zero V, etc.).
    finite_mask = np.isfinite(panel[["S_MC", "S_TV", "M", "V"]]).all(axis=1)
    panel = panel[finite_mask].reset_index(drop=True)
    panel = panel.sort_values(["date", "stock_id"]).reset_index(drop=True)

    print(f"\nFinal panel: {len(panel):,} rows, "
          f"{panel['stock_id'].nunique():,} stocks, "
          f"{panel['date'].nunique():,} dates "
          f"({panel['date'].min().date()} -> {panel['date'].max().date()})")
    print("Investor coverage (nonzero days):")
    for inv in ["institutional", "foreign", "dealer", "individual"]:
        nz = (panel[f"D_{inv}"] != 0).sum()
        print(f"  D_{inv:<14} {nz:>10,}  ({nz / len(panel):.1%})")

    if _AGGREGATE_DEALER_STOCKS:
        print(f"\nAggregated dealer disclosure: excluded "
              f"{len(_AGGREGATE_DEALER_STOCKS)} stocks whose in-window "
              f"proprietary flow is published on one line rather than three "
              f"({', '.join(sorted(_AGGREGATE_DEALER_STOCKS))})")

    # What the live registry costs: the delisted names it no longer lists,
    # measured against the vendor's own delisting record.
    survivor_ids, survivor_sessions = delisted_in_window(
        TWN_DELISTING, TWN_STOCK_INFO, OHLCV_DIR, START_DATE, END_DATE)
    print(f"\nNo-trade sessions refused: {_NO_TRADE_ROWS:,}")
    print(f"Sessions without a share count on them or on the session before, "
          f"refused: {_UNLISTED_ROWS:,}")
    print(f"\nSurvivorship: {len(survivor_ids)} names delisted in-window are "
          f"absent from the live registry and traded {survivor_sessions:,} "
          f"in-window sessions ({', '.join(survivor_ids)})")

    kept_value = float(panel["V"].sum())
    coverage = {
        "universe_four_digit": len(universe),
        "kept_stocks": int(panel["stock_id"].nunique()),
        "dropped_no_adjusted_close": int(no_px.sum()),
        "dropped_no_trade_rows": _NO_TRADE_ROWS,
        "dropped_unlisted_rows": _UNLISTED_ROWS,
        "dropped_aggregate_dealer_stocks": sorted(_AGGREGATE_DEALER_STOCKS),
        "dropped_with_in_window_ohlcv": dropped_with_ohlcv,
        "dropped_traded_value_twd": dropped_value,
        "kept_traded_value_twd": kept_value,
        "dropped_traded_value_share":
            dropped_value / (dropped_value + kept_value) if kept_value else None,
        "survivorship_missing_stocks": survivor_ids,
        "survivorship_missing_sessions": survivor_sessions,
    }
    print(f"\nUniverse coverage: {dropped_with_ohlcv:,} four-digit names have "
          f"in-window OHLCV but no usable instflow or shares record "
          f"({100 * coverage['dropped_traded_value_share']:.3f}% of in-window "
          f"traded value)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nWrote {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1024 ** 2:.1f} MB)")
    COVERAGE_PATH.write_text(json.dumps(coverage, indent=2))
    print(f"Wrote {COVERAGE_PATH}")


if __name__ == "__main__":
    main()
