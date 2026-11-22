"""Centralized paths for the replication pipeline.

All scripts in ``code/`` should import data and output paths from this
module rather than hard-coding ``/home/st/...`` or computing relative paths
via ``os.path.dirname(__file__)``.

Every external dataset the pipeline reads sits under one root inside this
repository, ``data/``, in the fixed layout below. Two kinds of thing live
there. The registry files are small and are committed, so a clone carries
them. The three data trees are large and are not: each is a directory the
replicator supplies, by cloning the public archive in the case of marcap and
by placing (or symlinking) the vendor download in the case of the other two.
``QF_DATA_ROOT`` relocates the whole root for a replicator who keeps the
trees elsewhere.

Expected layout under ``data/``:

    marcap/data/     marcap daily-data directory containing
                     ``marcap-YYYY.parquet`` files, read for the quantities
                     KRX publishes and the closing price they are formed at
                     (volume, traded value, market cap, close), and for the
                     point-in-time universe ``code/data/kr_universe.py`` builds from
                     the same snapshots
    fnguide/raw/     FnGuide DataGuide export files: the investor-flow sheets,
                     the 수정주가 export the Korean returns are formed on and
                     the universe reads its security types off, and the annual
                     financials
    finmind/         one directory per FinMind Taiwan dataset, each holding one
                     parquet per stock: ``ohlcv/`` (TaiwanStockPrice),
                     ``price_adj/`` (TaiwanStockPriceAdj, the 還原股價 series),
                     ``instflow/`` (TaiwanStockInstitutionalInvestorsBuySell),
                     ``shares/`` (TaiwanStockShareholding) and ``div_result/``
                     (TaiwanStockDividendResult)
    registry/        the small committed tables: the Taiwanese listing and
                     delisting registries, the market-wide capital-reduction
                     reference prices, the KOSPI 200 membership panel, and the
                     KRX delisting calendar that separates a reissued code's
                     two occupants

External paths are resolved lazily on attribute access, so a checker importing
this module for one pure function does not need the trees. A tree that is not
there raises ``RuntimeError`` naming it and what supplies it.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Repo-internal paths (no env vars; layout is fixed within the repo)
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "_cache"
TABLES_DIR = REPO_ROOT / "tables"

# --------------------------------------------------------------------------
# Convenience: pre-built result files that downstream scripts read
# --------------------------------------------------------------------------

DAILY_DATA_ALL_INVESTORS = RESULTS_DIR / "full_daily_data_all_investors.parquet"
DAILY_DATA_PRE2020 = RESULTS_DIR / "full_daily_data_all_investors_pre2020.parquet"
DAILY_DATA_TAIWAN = RESULTS_DIR / "full_daily_data_taiwan_all_investors.parquet"
HML_FACTOR_FILE = RESULTS_DIR / "hml_factor_daily.csv"
# Point-in-time Korean security-type universe, built from marcap and the
# FnGuide 수정주가 export by ``code/data/kr_universe.py``; ``code/data/_market.py`` reads
# its ``common`` rows.
KR_UNIVERSE_PANEL = RESULTS_DIR / "kr_universe_panel.parquet"
# The FnGuide 수정주가 export parsed once into long form; built on first use by
# ``code/data/_fnguide_price.py``.
FNGUIDE_PRICE_CACHE = CACHE_DIR / "fnguide_price.parquet"


# --------------------------------------------------------------------------
# External paths — resolved on attribute access
# --------------------------------------------------------------------------

_DATA_ROOT_DEFAULT = REPO_ROOT / "data"

# Logical name -> subpath under the data root.
_EXTERNAL_ROOTS = {
    "MARCAP_DIR": "marcap/data",
    "FNGUIDE_DIR": "fnguide",
    "FINMIND_DIR": "finmind",
    "REGISTRY_DIR": "registry",
}

# What supplies each root, quoted back in the error a missing one raises.
_ROOT_SOURCE = {
    "MARCAP_DIR": "clone https://github.com/FinanceData/marcap into data/marcap",
    "FNGUIDE_DIR": "place the FnGuide DataGuide export tree at data/fnguide "
                   "(subscription required; see README, Data requirements)",
    "FINMIND_DIR": "download the five FinMind Taiwan datasets into "
                   "data/finmind (README, Data requirements)",
    "REGISTRY_DIR": "committed with the repository; restore it from git",
}

_DERIVED = {
    # FnGuide-derived
    "FNGUIDE_RAW_DIR": ("FNGUIDE_DIR", "raw"),
    "FINANCIALS_FILE": ("FNGUIDE_RAW_DIR", "fnguide_financials_annual_20260219.xlsx"),
    # The two adjusted-price conventions, on four sheets; parsed by
    # ``code/data/_fnguide_price.py``.
    "FNGUIDE_ADJCLOSE_FILE": ("FNGUIDE_RAW_DIR", "fnguide_price_adjclose_20260813.xlsx"),
    # FinMind-derived (Taiwan): one directory per dataset, one parquet per stock.
    "FINMIND_INSTFLOW_DIR": ("FINMIND_DIR", "instflow"),
    "FINMIND_OHLCV_DIR": ("FINMIND_DIR", "ohlcv"),
    "FINMIND_PRICE_ADJ_DIR": ("FINMIND_DIR", "price_adj"),
    "FINMIND_SHARES_DIR": ("FINMIND_DIR", "shares"),
    "FINMIND_DIV_RESULT_DIR": ("FINMIND_DIR", "div_result"),
    # Committed registries.
    "TWN_STOCK_INFO": ("REGISTRY_DIR", "twn_stock_info.parquet"),
    "TWN_DELISTING": ("REGISTRY_DIR", "twn_delisting.parquet"),
    "TWN_CAP_RED": ("REGISTRY_DIR", "twn_capital_reduction.parquet"),
    "KOSPI200_PANEL": ("REGISTRY_DIR", "kospi200_index_panel_daily.parquet"),
    "KR_DELISTING_CALENDAR": ("REGISTRY_DIR", "kr_delisting_calendar.csv"),
}

_resolved: dict[str, Path] = {}


def _data_root() -> Path:
    root = Path(os.environ.get("QF_DATA_ROOT", _DATA_ROOT_DEFAULT)).expanduser()
    if not root.is_dir():
        raise RuntimeError(
            f"Data root {root} does not exist. The repository carries it as "
            f"data/; set QF_DATA_ROOT to your own root if the trees live "
            f"elsewhere. See paths.py docstring for the expected layout.")
    return root


def _resolve_root(name: str) -> Path:
    p = _data_root() / _EXTERNAL_ROOTS[name]
    if not p.is_dir():
        raise RuntimeError(
            f"{p} does not exist. To supply it: {_ROOT_SOURCE[name]}.")
    return p


def __getattr__(name: str) -> Path:
    if name in _resolved:
        return _resolved[name]
    if name in _EXTERNAL_ROOTS:
        p = _resolve_root(name)
        _resolved[name] = p
        return p
    if name in _DERIVED:
        parent_name, suffix = _DERIVED[name]
        parent = __getattr__(parent_name)
        p = parent / suffix
        _resolved[name] = p
        return p
    raise AttributeError(f"module 'paths' has no attribute {name!r}")
