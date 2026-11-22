#!/usr/bin/env python3
"""Point-in-time Korean common-stock universe, built from the marcap dataset.

subsec:data defines the panel's universe as the common stocks of KOSPI and
KOSDAQ, read point in time: a code enters on the session it first trades as a
common stock and leaves on the session it stops, so no later reclassification
reaches back over a stock's history. The marcap dataset is a daily snapshot of
every code KRX quoted that session and keeps a code after its delisting, which
is what makes the universe survivorship-free without a delisting calendar
beside it.

A security type is not a column marcap carries, and the listed name does not
settle it: 맥쿼리인프라, an infrastructure investment company, carries no type
word, and 메리츠금융지주 carries 리츠 inside an operating company's name. The
type is read instead off FnGuide's all-codes export, the 수정주가 series the
returns are formed on. The export prices no preferred line, REIT or listed
investment company, and it prices every other KOSPI and KOSDAQ issue on each
session it is listed, so a session the export does not price is filed as a
``fund``. On the 2026-08-13 export every such session belongs to a REIT or an
investment company (ship, infrastructure, resource, mutual).

The export also prices SPACs, so a SPAC is read off its name: 스팩 or SPAC, or
the shell number before 호 that two shells' listed names keep in place of the
word (케이프이에스제4호). The name test runs after the export test, so a
numbered ship fund, which the export does not price, is filed as a fund. KRX
gives a preferred line a terminal character other than ``0``, and KONEX is a
market. The classification is applied per session and the sessions are cut
into maximal runs of one kind, so a shell that becomes an operating company on
its merger carries one row per kind rather than one row asserting its latest
kind back over the shell's own prices.

The universe spans the sessions both sources carry, from the export's first
session: before it the export cannot tell a fund from a stock.

Output: ``results/kr_universe_panel.parquet`` with one row per
``(code, kind)`` run — ``code, name, market, kind, first_date, last_date,
n_days``. ``code/data/_market.py`` reads the ``common`` rows of it.

Degrees of freedom
------------------
MEASURED  the security type, read off the sessions FnGuide's all-codes export
          prices.
CHOSEN    the rules the export leaves to this script: the SPAC name test, the
          terminal-character test for preferred lines (the exchange's own
          numbering), and the KONEX market. Each reads a KRX naming or
          numbering convention rather than a threshold.
CHOSEN    MIN_DELISTED_COMMON and DELISTING_GRACE, a floor on the delisted
          common stocks the universe keeps. A safety guardrail rather than a
          tuned cutpoint; the constants say where it sits.
MEASURED  every span and count in the output, over the sessions both sources
          hold.
FITTED    nothing.

BASELINE: FnGuide's all-codes export as the security-type source | obvious
          alternative: a referee says the exchange's own listing register, not
          a vendor's coverage, is what defines a universe | discharge:
          sec:appendix_data, which states the convention each exclusion
          follows, and ``check_prose_universe_composition``, which asserts
          that every session the export prices is filed as common stock, SPAC
          or KONEX and that every common-stock session is one the export
          prices.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

KINDS = ("common", "preferred", "spac", "fund", "konex", "other")

# A SPAC shell's listed name carries 스팩 or SPAC, or only the shell number
# before 호.
_SPAC_RE = re.compile(r"스팩|SPAC|\d호$", re.IGNORECASE)
# KOSDAQ GLOBAL is a disclosure tier of KOSDAQ rather than a separate market,
# so its names are KOSDAQ commons.
_COMMON_MARKETS = ("KOSPI", "KOSDAQ", "KOSDAQ GLOBAL")

# A pull of the export without the 상폐-included filter prices only the issues
# still listed when it was pulled, so every delisted stock would be filed as a
# fund, and the price join in code/data/_market.py would not see it: that join only
# receives sessions the export prices. The universe must therefore keep common
# stocks whose last session falls well before its own. On the 2026-08-13
# export it keeps 740, and a pull without the filter would keep almost none.
# The floor sits far below the first and above the second, so it catches the
# filter rather than drift in the delisting count.
MIN_DELISTED_COMMON = 100
# A code whose last common session falls within this of the universe's last
# session may only have stopped trading in the final weeks, so it is not
# counted as delisted. check_prose_panel_survivorship allows the same month.
DELISTING_GRACE = pd.Timedelta(days=30)


def classify_session(code: str, name: str, market: str, priced: bool) -> str:
    """The security kind of one Korean listing on one session, first match winning.

    1. the KONEX market                                  -> ``konex``
    2. a terminal character other than ``0``             -> ``preferred``
    3. a session FnGuide's all-codes export leaves unpriced -> ``fund``
    4. 스팩 / SPAC, or a shell number before 호            -> ``spac``
    5. KOSPI / KOSDAQ / KOSDAQ GLOBAL                    -> ``common``
    6. anything else                                     -> ``other``

    KRX issues six-character alphanumeric codes as well as the familiar
    six-digit ones (``00088K`` 한화3우B, ``0001A0`` 덕양에너젠). The terminal
    character catches the K/L/M-suffixed preferred lines among them, and an
    alphanumeric common ends in ``0`` and falls through to its market.
    """
    code = str(code).strip()
    name = str(name).strip()
    if market == "KONEX":
        return "konex"
    if code[-1] != "0":
        return "preferred"
    if not priced:
        return "fund"
    if _SPAC_RE.search(name):
        return "spac"
    if market in _COMMON_MARKETS:
        return "common"
    return "other"


def _classify_frame(df: pd.DataFrame) -> pd.Series:
    """``classify_session`` over a frame, evaluated once per distinct state.

    A session-by-session call would classify the same (code, name, market,
    priced) state on every one of its sessions. The distinct states number in
    the thousands against millions of rows, so the classification is taken on
    them and joined back.
    """
    keys = ["Code", "Name", "Market", "priced"]
    states = df[keys].drop_duplicates()
    states["kind"] = [classify_session(c, n, m, p)
                      for c, n, m, p in states.itertuples(index=False)]
    return df.merge(states, on=keys, how="left")["kind"]


def build_universe_panel(marcap_dir: Path, out_path: Path) -> pd.DataFrame:
    """Classify every marcap session the export spans and cut each code into
    single-kind runs.

    The name and market a run carries are the most recent ones observed for
    that code under that kind, so a row displays the listing as it last stood
    rather than as it was first quoted.
    """
    from data._fnguide_price import load_price_panel

    px = load_price_panel()[["date", "ticker"]]
    frames = []
    for year_file in sorted(Path(marcap_dir).glob("marcap-*.parquet")):
        df = pd.read_parquet(year_file, columns=["Code", "Name", "Market", "Date"])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no marcap-YYYY.parquet files under {marcap_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["Code", "Date"])
    df["Code"] = df["Code"].astype(str).str.zfill(6)
    df["Name"] = df["Name"].where(df["Name"].map(lambda v: isinstance(v, str)), "")
    df["Market"] = df["Market"].where(df["Market"].map(lambda v: isinstance(v, str)), "")
    df["Date"] = pd.to_datetime(df["Date"])
    first = px["date"].min()
    last = min(df["Date"].max(), px["date"].max())
    df = df[(df["Date"] >= first) & (df["Date"] <= last)].reset_index(drop=True)
    served = pd.MultiIndex.from_frame(px[["ticker", "date"]])
    df["priced"] = pd.MultiIndex.from_arrays([df["Code"], df["Date"]]).isin(served)
    df["kind"] = _classify_frame(df).to_numpy()

    df = df.sort_values(["Code", "Date"], kind="mergesort").reset_index(drop=True)
    new_run = (df["Code"] != df["Code"].shift()) | (df["kind"] != df["kind"].shift())
    df["run"] = new_run.cumsum()

    runs = df.groupby("run", sort=True).agg(
        code=("Code", "first"),
        kind=("kind", "first"),
        first_date=("Date", "min"),
        last_date=("Date", "max"),
        n_days=("Date", "size"),
    )
    # The displayed name and market are per (code, kind) rather than per run:
    # a code whose kind alternates shows the same listing on each of its runs
    # of that kind.
    meta = (df.sort_values("Date", kind="mergesort")
              .groupby(["Code", "kind"], sort=False)
              .agg(name=("Name", "last"), market=("Market", "last"))
              .reset_index()
              .rename(columns={"Code": "code"}))
    panel = runs.merge(meta, on=["code", "kind"], how="left")
    panel = panel[["code", "name", "market", "kind",
                   "first_date", "last_date", "n_days"]]
    panel = panel.sort_values(["kind", "code", "first_date"]).reset_index(drop=True)

    ends = panel[panel["kind"] == "common"].groupby("code")["last_date"].max()
    delisted = int((ends < last - DELISTING_GRACE).sum())
    if delisted < MIN_DELISTED_COMMON:
        raise RuntimeError(
            f"the universe keeps {delisted} common stocks whose last session "
            f"precedes {last.date()} by more than {DELISTING_GRACE.days} days, "
            f"below the floor of {MIN_DELISTED_COMMON}. An export pulled "
            f"without the 상폐-included filter would look like this; check the "
            f"export's vintage before proceeding.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    return panel


def main() -> None:
    from paths import KR_UNIVERSE_PANEL, MARCAP_DIR

    panel = build_universe_panel(Path(MARCAP_DIR), Path(KR_UNIVERSE_PANEL))
    print(f"Universe panel: {len(panel):,} rows "
          f"({panel['code'].nunique():,} codes), "
          f"{panel['first_date'].min():%Y-%m-%d} to {panel['last_date'].max():%Y-%m-%d}")
    print(panel["kind"].value_counts().to_string())
    print(f"Wrote {KR_UNIVERSE_PANEL}")


if __name__ == "__main__":
    main()
