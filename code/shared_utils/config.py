"""Project-wide configuration constants.

Centralizes values that are otherwise duplicated across analysis scripts so the
calibration or coverage window can be updated in one place. Each constant lists
its callers in the docstring; check those when changing a value.

This is where most of the project's degrees of freedom live, so each constant
opens with MEASURED, CHOSEN or FITTED, and a script that imports one inherits
that tag rather than restating it. Only ``SIGMA_XI_CALIBRATED`` is FITTED, to
the single target named in its docstring. The rest are structural, and each
docstring names the sweep or sensitivity arm that backs the value where one
exists.
"""

# === Sample windows ===

START_DATE_MAIN = "2020-01-02"
END_DATE_MAIN = "2025-12-31"
"""CHOSEN — main sample window (Korean panel, 2020--2025). The panel was
first scored on 2020--2024, and the sixth year was added after the criteria in
``empirical/robustness/window_2025_criteria.py`` were committed. The window
closes at a calendar-year boundary because 2026 is partial. Used by every
empirical analysis script and by the data pipelines under ``code/data/``."""

START_DATE_PRE = "2015-01-01"
END_DATE_PRE = "2019-12-31"
"""CHOSEN — pre-period out-of-sample window, the five years immediately
before the main one and disjoint from it. Used by ``data/pre_period_pipeline.py``."""

# === Cross-section thresholds ===

MIN_STOCKS_PER_DAY_DEFAULT = 30
"""CHOSEN — default minimum cross-section per trading day for Fama-MacBeth aggregation
(qf_paper.tex Methodology, subsec:return_prediction). Every Fama-MacBeth
specification in the paper uses this value."""

# === Reproducibility ===

MC_SEED = 42
"""CHOSEN — the seed that the empirical scripts pass to their random draws,
committed before the runs."""

