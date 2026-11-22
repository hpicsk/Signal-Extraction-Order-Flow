"""LaTeX formatting + IO + nav helpers shared by ``code/output/generate_table_bodies.py``.

Pulled out of the main generator so the macro-emission layer can be read as
declarative inventory (see ``code/output/_macro_specs.py``) and the body-generator layer
as bespoke LaTeX. Nothing in this module touches the manuscript-specific
emit logic — it is pure formatting, file IO, dotted-path navigation, and
``scipy``-delegated statistics helpers.
"""

import json
import math

from scipy import stats

from paths import TABLES_DIR


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

# Significance star thresholds (standard *,**,*** convention; the table
# notes in sec:empirical).
SIG_THRESHOLD_1PCT  = 0.01
SIG_THRESHOLD_5PCT  = 0.05
SIG_THRESHOLD_10PCT = 0.10

# Two-sided critical values from the standard normal, derived from the
# significance thresholds above. Every sample in this manuscript exceeds 1000
# observations, so the normal approximation is exact to 2 decimal places.
# Used by the prose/year-list macro gates that quote 1%/5% significance
# (subsec:appendix_j_oos) and the prose-qualifier lint.
CRIT_1PCT  = float(stats.norm.ppf(1 - SIG_THRESHOLD_1PCT / 2))   # 2.576
CRIT_5PCT  = float(stats.norm.ppf(1 - SIG_THRESHOLD_5PCT / 2))   # 1.960
CRIT_10PCT = float(stats.norm.ppf(1 - SIG_THRESHOLD_10PCT / 2))  # 1.645

# Default decimal precision by quantity type (matches manuscript table
# conventions: t-stats to 2dp, coefficients to 4dp).
DECIMALS_TSTAT = 2
DECIMALS_COEF  = 4

# Unit conversions for the long-short return table (sec:appendix_robustness_battery)
# and the bps macros: a decimal return r maps to basis points as r * BPS_PER_RETURN,
# and a daily-bps figure annualises as bps * TRADING_DAYS_PER_YEAR (the manuscript's
# 252 trading-day year, subsec:appendix_j_longshort).
BPS_PER_RETURN = 10_000
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Format helpers
# ---------------------------------------------------------------------------

def stars(p):
    """Return significance stars for a p-value."""
    if p < SIG_THRESHOLD_1PCT:
        return "***"
    elif p < SIG_THRESHOLD_5PCT:
        return "**"
    elif p < SIG_THRESHOLD_10PCT:
        return "*"
    return ""


def fmt_t(t, p=None, decimals=DECIMALS_TSTAT):
    """Format a t-statistic with significance stars, wrapped for LaTeX.

    Negative values are wrapped in $...$ for proper math minus signs.
    Returns e.g. '7.21' or '$-3.12$***'.
    """
    raw = f"{t:.{decimals}f}"
    sig = stars(p) if p is not None else ""
    if t < 0:
        return f"${raw}${sig}"
    return f"{raw}{sig}"


def fmt_t_paren(t, p=None, decimals=DECIMALS_TSTAT):
    """Format a parenthesised t-statistic for table rows.

    Returns e.g. '(7.21)' or '($-3.12$)'.
    """
    raw = f"{t:.{decimals}f}"
    if t < 0:
        return f"(${raw}$)"
    return f"({raw})"


def fmt_coef(c, decimals=DECIMALS_COEF):
    """Format a coefficient with LaTeX-safe negative sign.

    Returns e.g. '0.0004' or '$-0.0001$'.
    """
    raw = f"{c:.{decimals}f}"
    if c < 0:
        return f"${raw}$"
    return raw


def fmt_coef_star(c, p, decimals=DECIMALS_COEF):
    """Format a coefficient with significance stars.

    Returns e.g. '0.0004***' or '$-0.0001$***'.
    """
    raw = f"{c:.{decimals}f}"
    sig = stars(p)
    if c < 0:
        return f"${raw}${sig}"
    return f"{raw}{sig}"


def fmt_comma(n):
    """Format integer with LaTeX-safe thousand separators: 200000 → '200{,}000'."""
    return f"{int(n):,}".replace(",", "{,}")


def fmt_sci(val, sig_figs=2):
    r"""Format a number in LaTeX scientific notation: 3.5e-5 → '3.5 \times 10^{-5}'."""
    if val == 0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / (10 ** exp)
    if abs(mantissa - round(mantissa)) < 1e-10:
        return f"{int(round(mantissa))} \\times 10^{{{exp}}}"
    return f"{mantissa:.{sig_figs - 1}f} \\times 10^{{{exp}}}"


def fmt_r2_from_pct(v, decimals=2):
    """Format R² that is already in percentage units (0.29 → '0.29')."""
    return f"{v:.{decimals}f}"


def _bold_max(a_str, b_str, a_val, b_val, key=abs):
    """Wrap whichever of ``(a_str, b_str)`` has the larger ``key(value)`` in
    ``\\textbf{...}``. Ties go to ``b_str`` being bolded, matching the
    ``if key(a) > key(b): bold a; else: bold b`` idiom used throughout the
    body generators (tab:directionality_kr,
    subsec:subtype_dose_response).
    """
    if key(a_val) > key(b_val):
        return f"\\textbf{{{a_str}}}", b_str
    return a_str, f"\\textbf{{{b_str}}}"


def _wrap_negative_in_math(x):
    """Wrap a pre-formatted cell containing ``-`` in ``$...$`` so the minus
    renders as a math minus. If ``x`` is already ``\\textbf{...}``, the math
    delimiters are inserted inside the brace pair so the bold survives.
    """
    if "-" not in x:
        return x
    if "\\textbf" not in x:
        return f"${x}$"
    return x.replace("\\textbf{", "\\textbf{$").replace("}", "$}", 1)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


WRITTEN_BODIES: set[str] = set()


def write_body(filename, content):
    """Write a table body .tex file, recording it for ``sweep_stale_bodies``."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    with open(path, "w") as f:
        f.write(content)
    WRITTEN_BODIES.add(filename)
    print(f"  wrote {path}")


def sweep_stale_bodies():
    r"""Delete body files in TABLES_DIR that this run did not write.

    A renamed or retired table leaves its old body behind: the generator stops
    writing it, nothing \input{}s it, and it holds the run it was last written
    from. The file is gitignored, so no committed-artifact check sees it, and a
    future \input under the old name would read a stale table with the suite
    green. Removing what this run did not produce keeps TABLES_DIR exactly the
    generator's output.
    """
    for path in sorted(TABLES_DIR.glob("*_body.tex")):
        if path.name not in WRITTEN_BODIES:
            path.unlink()
            print(f"  removed stale {path}")


# ---------------------------------------------------------------------------
# Nav + spec-driven emission helpers
# ---------------------------------------------------------------------------

def _with_botrule(lines):
    """Replace the trailing ``\\\\`` of the last row with ``\\\\\\botrule``.

    Mutates and returns ``lines`` so the call sites can write either
    ``_with_botrule(lines)`` standalone or ``"\\n".join(_with_botrule(lines))``.
    """
    if lines:
        lines[-1] = lines[-1].replace(" \\\\", " \\\\\\botrule")
    return lines


def _nav(source, path):
    """Walk a dotted path into a nested dict/list. ``"a.b.c"`` → ``source["a"]["b"]["c"]``.

    Returns ``None`` if any intermediate key is absent — matches the existing
    ``if ... in ...:`` guards scattered through the original ``_emit_*`` helpers.
    """
    cur = source
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _emit_from_spec(cmd, source, spec):
    """Emit one ``\\newcommand`` per row in ``spec`` (a list of
    ``(macro_name, dotted_path, fmt_string)`` tuples). A row whose path
    resolves to ``None`` raises, naming the macro and the path, so a results
    key that a regeneration dropped stops the generator instead of removing
    the macro from ``tables/numbers.tex``.
    """
    for name, path, fmt in spec:
        value = _nav(source, path)
        if value is None:
            raise KeyError(f"\\{name}: results path {path!r} resolves to nothing")
        cmd(name, fmt.format(value))


def _emit_yearly_subperiod_extrema(cmd, year_dict, prefix, axes=("MC", "TV"),
                                   n_years=True):
    """Emit signed min/max macros over a yearly-subperiod horse-race result.

    ``year_dict`` is e.g. ``ext["yearly_subperiods"]`` — a dict keyed by year
    string whose values carry ``horse_race.MC.t_stat`` and ``horse_race.TV.t_stat``.

    ``axes`` selects which filters get extrema. The stability claim each caller
    makes is about that class's matched filter, so the capacity-side callers
    pass ``MC`` and the foreign one ``TV``; the other axis is described by year
    counts in prose rather than by its extremes. ``n_years`` adds the count
    alongside the span, which only the table note that reports both needs.
    """
    items = sorted(year_dict.items())
    series_by_axis = {
        axis: [(int(y), d["horse_race"][axis]["t_stat"]) for y, d in items]
        for axis in axes
    }
    for axis, series in series_by_axis.items():
        lo = min(series, key=lambda x: x[1])
        hi = max(series, key=lambda x: x[1])
        cmd(f"{prefix}MinHR{axis}t",    f"{lo[1]:.2f}")
        cmd(f"{prefix}MaxHR{axis}t",    f"{hi[1]:.2f}")
        cmd(f"{prefix}MinHR{axis}Year", f"{lo[0]}")
        cmd(f"{prefix}MaxHR{axis}Year", f"{hi[0]}")
    cmd(f"{prefix}YearsCovered", f"{items[0][0]}--{items[-1][0]}")
    if n_years:
        cmd(f"{prefix}NYears", f"{len(items)}")


# --------------------------------------------------------------------------
# Statistics helpers (delegated to numpy / scipy.stats)
# ---------------------------------------------------------------------------

def _p_from_t(t, n):
    """Two-sided p-value from a Student-$t_{n-1}$ statistic via
    ``scipy.stats.t.sf`` (the survival function, ``1 - cdf``, but computed
    accurately in the upper tail). ``n`` is the sample size that produced
    ``t``; the degrees of freedom are ``n - 1``.
    """
    return float(2.0 * stats.t.sf(abs(t), df=n - 1))
