# `data/registry/` — the small tables the pipeline reads

Everything here is committed, so a clone carries it. Each file is a registry
rather than a measurement: it says which securities existed, which of them the
exchange stopped quoting, and on which sessions a corporate action took effect.
The large price, flow and quantity trees are not here — `data/marcap/`,
`data/fnguide/` and `data/finmind/` are supplied by the replicator, as the
repository README describes.

| File | What it is | Rows | Read by |
|---|---|---|---|
| `twn_stock_info.parquet` | FinMind `TaiwanStockInfo`, the live Taiwanese listing registry, pulled 2026-09-16. One row per (stock, market, industry) classification, not one per stock | 4,323 | `code/data/_finmind.py`, which draws the Taiwan universe from it |
| `twn_delisting.parquet` | FinMind `TaiwanStockDelisting`, the vendor's record of a name leaving a board, covering 1995-09-23 to 2026-09-03 | 725 | `code/data/_finmind.py`, to size what the live registry omits |
| `twn_capital_reduction.parquet` | FinMind `TaiwanStockCapitalReductionReferencePrice`, the market-wide 減資 reference prices, covering 2011-01-25 to 2026-09-09. The start is the publisher's: TWSE's 股票減資恢復買賣參考價格 table refuses any query beginning before it | 674 | `code/data/twn_full_daily_data_pipeline.py`, for the `is_cap_red` flag |
| `kospi200_index_panel_daily.parquet` | KOSPI 200 daily membership, one row per (date, index, ticker), reconstructed from KRX month-end snapshots and constituent-change events, with a synthetic event standing in wherever a snapshot proves a change the event log never recorded | 1,673,235 | The KOSPI 200 falsification in `code/empirical/` |
| `kr_delisting_calendar.csv` | KRX delisting records: ticker, name, market, delisting date, reason, and whether the delisting ended the entity | 1,397 | `code/data/_fnguide_price.py`, which uses a genuine delisting inside a price gap to separate a reissued code's two occupants |
