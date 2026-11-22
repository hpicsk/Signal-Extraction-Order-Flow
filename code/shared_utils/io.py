"""JSON output helper with numpy-aware serialization.

``save_json`` is the single result-writing entry point across the analysis
scripts, pairing numpy-type coercion with ``json.dump(..., indent=2)``.

Coercion is done via ``json.dump(default=...)`` so plain-Python dicts pass
through unchanged (byte-identical to a vanilla ``json.dump`` call).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def _coerce(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def save_json(obj, path, *, indent: int = 2, announce: bool = True,
              ensure_ascii: bool = True) -> str:
    """Write ``obj`` to ``path`` as JSON with numpy-type coercion.

    Parameters
    ----------
    obj : any JSON-compatible structure (dicts, lists, scalars, numpy scalars/arrays).
    path : str | os.PathLike — output file. Parent directory is created if missing.
    indent : passed through to ``json.dump`` (default 2).
    announce : if True, print ``Results saved to: <path>`` on success.
    ensure_ascii : passed through to ``json.dump`` (default True). Set False
        for output dicts containing non-ASCII labels (e.g. KRX 기관계).

    Returns the path written, as a string, for chaining.
    """
    path = os.fspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_coerce, ensure_ascii=ensure_ascii)
    if announce:
        print(f"Results saved to: {path}")
    return path


def save_csv(df, path, *, index: bool = False, announce: bool = True) -> str:
    """Write ``df`` to ``path`` as CSV with parent-dir mkdir + announce line.

    Mirrors ``save_json``'s contract for the ``DataFrame.to_csv`` path used
    by the empirical drivers. The CSV
    contents themselves are byte-identical to a vanilla ``df.to_csv(path,
    index=index)`` call; this helper only adds the directory creation and
    the standardised ``Results saved to: <path>`` line.
    """
    path = os.fspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    if announce:
        print(f"Results saved to: {path}")
    return path


def save_parquet(df, path, *, index: bool = False, announce: bool = True) -> str:
    """Write ``df`` to ``path`` as Parquet with parent-dir mkdir + announce line.

    The Parquet counterpart of :func:`save_csv`, used for the large daily
    panels (``full_daily_data_all_investors[_pre2020]``). Parquet is ~20x
    faster to read and ~3x smaller on disk than CSV for these multi-million-row
    frames, and round-trips native dtypes (datetime, float64) so downstream
    readers need no ``parse_dates`` re-inference. Same contract as
    ``save_csv``: only adds the directory creation and the standardised
    ``Results saved to: <path>`` line on top of ``DataFrame.to_parquet``.
    """
    path = os.fspath(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=index)
    if announce:
        print(f"Results saved to: {path}")
    return path


_HEADER_CHARS = {"banner": "#", "section": "=", "subsection": "-"}


def print_header(title: str, *, level: str = "section", width: int = 70) -> None:
    """Print a bordered console header used by the QF analysis scripts.

    ``level`` selects the border character: ``banner`` (``#``) for top-level
    phase boundaries, ``section`` (``=``) for sections, ``subsection``
    (``-``) for nested blocks.
    """
    bar = _HEADER_CHARS.get(level, "=") * width
    print(f"\n{bar}\n{title}\n{bar}")
