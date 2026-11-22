"""Taiwan vendor series: FinMind's listing registry and its two price series.

Everything the Taiwan panel reads is a FinMind endpoint as the vendor serves
it. Nothing here reconstructs a price, patches an event, or re-derives a
listing: what the endpoints answer is what the panel carries, and what they do
not answer is what sec:taiwan discloses.

Two price series come back for one stock. ``TaiwanStockPrice`` (``ohlcv/``) is
the traded print, and three quantities are levels rather than returns and must
use it: the net-buy value a flow is converted at, the market capitalization
formed against the share count, and the share-space terms of the persistence
decomposition. ``TaiwanStockPriceAdj`` (``price_adj/``) is the 還原股價 series,
which is total return — cash dividends come out along with 無償配股, 現增 and
減資 — and the forward returns are formed on it alone. There is no price-return
variant of the endpoint at any tier, and TWSE and TPEx publish a single fused
reference price per 除權息 covering the cash and the stock components together,
so a price-return series would need a split of that fused step with nothing to
validate the split against.

The universe is the vendor's live listing registry, ``TaiwanStockInfo``. It
lists a name the exchange currently quotes, so a company the exchange has
delisted can fall out of it, which is the survivorship exposure sec:taiwan
states and ``delisted_in_window`` measures against the vendor's own delisting
record.
"""

from __future__ import annotations

import re

import pandas as pd
import pyarrow.parquet as pq

# Instrument types the registry names that are not common equity. A stock is
# dropped if any of its rows carries one: the type says what an instrument is,
# and none of these is a status a name holds for part of a window and not the
# rest. 創新版股票 / 創新板股票 is the TWSE Innovation Board, a relaxed-disclosure
# tier rather than a main-board listing.
EXCLUDED_TYPES = frozenset({
    "ETF", "ETN", "受益證券", "存託憑證", "臺灣存託憑證",
    "創新版股票", "創新板股票",
})
# The two boards the panel covers. 興櫃 (emerging) is the third the registry
# carries and is not a listing.
LISTED_MARKETS = ("twse", "tpex")
# A common stock's code is four digits. ETFs are numbered 00xx, warrants carry
# six alphanumeric characters, and preferred lines end in a letter.
COMMON_CODE = re.compile(r"\d{4}")


def _registry(stock_info_path) -> pd.DataFrame:
    raw = pd.read_parquet(stock_info_path)
    raw["stock_id"] = raw["stock_id"].astype(str)
    return raw


def load_universe(stock_info_path) -> list[str]:
    """The four-digit TWSE and TPEx common stocks the live registry lists.

    The registry serves one row per (market, industry) a stock has been
    classified under rather than one row per stock, and a classification the
    exchange has retired keeps the date it was retired on while one still in
    force carries the date the registry was pulled. So the exclusions drop the
    *stock* on the evidence of any of its rows: dropping rows and deduplicating
    afterwards would keep a name alive on whichever classification the response
    happened to list first, and every Innovation Board name also carries an
    ordinary industry row.

    A name still on the emerging board when the registry was pulled is dropped
    with them. Its twse or tpex row is the promotion that follows 興櫃, so it
    marks a listing that begins after the panel rather than one inside it.
    """
    raw = _registry(stock_info_path)
    excluded = set(raw.loc[raw["industry_category"].isin(EXCLUDED_TYPES), "stock_id"])
    listed = set(raw.loc[raw["type"].isin(LISTED_MARKETS), "stock_id"])

    emerging = raw.loc[raw["type"] == "emerging", ["stock_id", "date"]].copy()
    emerging["date"] = pd.to_datetime(emerging["date"], errors="coerce")
    unparsed = emerging.loc[emerging["date"].isna(), "stock_id"].tolist()
    if unparsed:
        raise ValueError(
            f"{len(unparsed)} emerging rows carry no readable date "
            f"({unparsed[:5]}); each would read as a name that was never on the "
            f"emerging board and admit it to the whole window")
    pulled = pd.to_datetime(raw["date"], errors="coerce").max()
    still_emerging = emerging.groupby("stock_id")["date"].max()
    listed -= set(still_emerging.index[still_emerging >= pulled])

    return sorted(s for s in (listed - excluded) if COMMON_CODE.fullmatch(s))


def emerging_exits(stock_info_path) -> dict[str, pd.Timestamp]:
    """The date on which the registry retired each stock's emerging-board row.

    A stock that listed from the emerging board keeps that row, dated when the
    classification was retired (``load_universe``). The price endpoints serve
    the stock's emerging-board sessions too, so the date marks where its listed
    sessions begin, give or take the session on which the record lands.
    """
    raw = _registry(stock_info_path)
    emerging = raw[raw["type"] == "emerging"]
    return pd.to_datetime(emerging["date"]).groupby(emerging["stock_id"]).max().to_dict()


def delisted_in_window(delisting_path, stock_info_path, ohlcv_dir, start, end):
    """The delisted names the live registry no longer lists, and their sessions.

    ``TaiwanStockDelisting`` records that a name left a board and on what date.
    A four-digit code it names, absent from the live registry and still printing
    inside the window, is a stock the panel cannot reach: the universe is drawn
    from the registry, so the name is gone from it while its sessions are not
    gone from the tape. Counting them is what puts a magnitude on the
    survivorship exposure sec:taiwan states.

    Returns ``(stock_ids, sessions)`` — the codes and the number of in-window
    OHLCV sessions they carry between them.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    delisted = pd.read_parquet(delisting_path)
    delisted["stock_id"] = delisted["stock_id"].astype(str)
    delisted["date"] = pd.to_datetime(delisted["date"])
    exits = delisted[delisted["stock_id"].str.fullmatch(COMMON_CODE)
                     & (delisted["date"] >= start) & (delisted["date"] <= end)]

    live = set(_registry(stock_info_path)["stock_id"])
    missing = sorted(set(exits["stock_id"]) - live)

    sessions = 0
    reached = []
    for sid in missing:
        n = _session_count(ohlcv_dir, sid, start, end)
        if n:
            reached.append(sid)
            sessions += n
    return reached, sessions


def _session_count(ohlcv_dir, stock_id, start, end) -> int:
    path = ohlcv_dir / f"{stock_id}.parquet"
    if not path.exists() or pq.read_metadata(path).num_rows == 0:
        return 0
    dates = pd.to_datetime(pd.read_parquet(path, columns=["date"])["date"])
    return int(((dates >= start) & (dates <= end)).sum())


def load_prices(stock_id, ohlcv_dir, price_adj_dir, start, end):
    """The traded print and the total-return adjusted close, on one frame.

    Returns ``date, close, adj_close_tr, Trading_Volume, Trading_money`` over
    the requested window, or ``None`` where the vendor serves no traded session
    in it. ``close`` and the two turnover columns are the raw endpoint's;
    ``adj_close_tr`` is the adjusted endpoint's close, joined on the date.

    The two endpoints do not agree on which sessions exist. The raw series
    carries a session the stock did not trade as ``close == 0``, and the
    adjusted series fills the same session with the last traded price, so a
    caller filtering the adjusted close alone would keep a price nobody could
    transact at. The join is therefore left onto the raw series and the panel's
    own ``M > 0`` filter removes those rows, on the traded print rather than on
    the filled one. A session the adjusted endpoint does not serve at all
    arrives with ``adj_close_tr`` missing and leaves the panel with the rows
    that carry no adjusted close.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    ohlcv_path = ohlcv_dir / f"{stock_id}.parquet"
    adj_path = price_adj_dir / f"{stock_id}.parquet"
    if not ohlcv_path.exists() or not adj_path.exists():
        return None
    # The download writes a zero-row file for every stock the endpoint returned
    # nothing for, and such a file carries no schema at all, so the column
    # projection below cannot be pushed down on it: the row count settles from
    # the metadata first.
    if pq.read_metadata(ohlcv_path).num_rows == 0:
        return None

    raw = pd.read_parquet(ohlcv_path,
                          columns=["date", "close", "Trading_Volume", "Trading_money"])
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[(raw["date"] >= start) & (raw["date"] <= end)]
    if raw.empty:
        return None

    if pq.read_metadata(adj_path).num_rows == 0:
        adj = pd.DataFrame({"date": pd.to_datetime([]), "adj_close_tr": []})
    else:
        adj = pd.read_parquet(adj_path, columns=["date", "close"])
        adj["date"] = pd.to_datetime(adj["date"])
        adj = adj.rename(columns={"close": "adj_close_tr"})

    out = raw.merge(adj, on="date", how="left").sort_values("date")
    return out.reset_index(drop=True)
