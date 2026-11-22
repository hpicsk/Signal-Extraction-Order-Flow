#!/usr/bin/env python3
"""Prose-adjacency lint for manuscript qualitative claims.

Closes the loop between manuscript prose and the numeric macros it references.
`code/output/test_assertions.py` already verifies that each macro value matches the source
JSON/CSV. This module verifies that the *qualifier words* around each macro
(`insignificant`, `marginally significant`, `much stronger than`, `$p<0.05$`,
`predicts returns negatively`, etc.) agree with the macro's numeric value.

Detection model
---------------
A macro reference `\name` is treated as a t-statistic usage when it appears
inside a math block `$...$` whose body contains `t = ...` (possibly with
sub/superscripts). Qualifier words are then scoped to the enclosing sentence
and paired with the nearest t-stat usage by character distance. The same
pairing is applied to `$p < X$` inequalities and to two-macro comparatives
(`much stronger than`, `order of magnitude`).

Thresholds
----------
The default significance band is 5% (two-sided), matching the manuscript's
working convention (see qf_paper.tex subsec:return_prediction and the horizon-table tabnote).
The 1% / 5% / 10% constants are sourced from `_formatters` so the linter,
`stars()`, and the author's vocabulary share one source of truth.

Severities
----------
- `error`   — qualifier word contradicts the adjacent macro value; fails the
              build via SystemExit(1) when invoked from test_assertions.main().
- `warning` — `$p<X$` claim is technically true but conservative by an order
              of magnitude (e.g., $p<0.05$ when the t-stat implies $p<0.001$).
- `info`    — comparative claim ("much stronger ... than ...") whose two
              macros cannot be unambiguously located in one sentence.

Register
--------
A second, macro-free pass (`_check_register`) scans the same files for prose
that has adopted the repo's internal vocabulary
(arm codenames, ALL-CAPS status tokens,
ordinary words used in the repo's sense) or a term of art whose evidential
force comes from an institution the reader cannot check (including timing
claims that restate an unauditable chronology). Both are
errors, suppressible per line with `% noqa: prose-lint(repo-register)` or
`% noqa: prose-lint(term-of-art)`.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

from paths import REPO_ROOT
from output._formatters import (
    CRIT_1PCT,
    CRIT_5PCT,
    CRIT_10PCT,
)
from output.assertions.common import load_macros, parse_num


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MANUSCRIPT_FILES = [
    REPO_ROOT / "qf_paper.tex",
]

# Two-sided normal critical values (CRIT_1PCT/5PCT/10PCT) are imported from
# _formatters, the single source shared with the macro emitters.
#
# Directional ("predicts positively/negatively") claims use 5% two-sided,
# matching the manuscript's signed-claim convention.
PREDICTS_THRESHOLD = CRIT_5PCT

# Comparative magnitude ratios.
MUCH_STRONGER_RATIO = 2.0
ORDER_OF_MAGNITUDE_RATIO = 7.0

# A $p<X$ inequality is downgraded to a warning when the t-stat implies a
# p-value tighter than the claim by this factor. Default = one order of
# magnitude (e.g., $p<0.05$ written when actual $p<0.005$).
LOOSE_P_CLAIM_FACTOR = 10.0

# Negation/scope bigrams masked before qualifier matching, so they cannot
# trigger a "significant" / "insignificant" rule.
NEGATION_BIGRAMS = [
    "not significant",
    "not insignificant",
    "economically significant",
    "statistically insignificant",
    "in significant",  # catches "in significant ways" etc.
]

# Per-line suppression: `% noqa: prose-lint(rule-name)` at end of line.
NOQA_RE = re.compile(r"%\s*noqa:\s*prose-lint\(([\w\s,-]+)\)")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Violation:
    severity: str       # 'error', 'warning', 'info'
    file: Path
    line: int
    rule: str
    message: str

    def __str__(self) -> str:
        try:
            rel = self.file.relative_to(REPO_ROOT)
        except ValueError:
            rel = self.file
        return f"{rel}:{self.line} [{self.severity} {self.rule}] {self.message}"


@dataclass
class TStatUsage:
    """A `\\macro` reference appearing in a t-stat context within a sentence."""
    macro: str
    offset: int    # character offset within the sentence
    value: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def t_to_p(t: float) -> float:
    """Two-sided p-value from a t-statistic (normal approximation)."""
    return 2.0 * (1.0 - float(stats.norm.cdf(abs(t))))


# Math block `$...$` (non-greedy).
_MATH_RE = re.compile(r"\$([^$]+)\$")

# Inside a math block, a `t = ...` or `t_{...}^{...} = ...` lhs.
_T_LHS_RE = re.compile(
    r"\bt"                              # t
    r"(?:_\{?(?:[^{}\s=]|\{[^{}\s=]*\})*\}?)?"   # optional subscript
    r"(?:\^\{?(?:[^{}\s=]|\{[^{}\s=]*\})*\}?)?"  # optional superscript
    r"\s*="
)

# Macro reference inside any LaTeX text.
_MACRO_RE = re.compile(r"\\([A-Za-z][A-Za-z0-9]*)")

# $p < X$ inequality, also covering "p<X" without dollar signs.
_PVAL_INEQ_RE = re.compile(
    r"\$?\s*p\s*<\s*0?\.(\d{1,4})\s*\$?"
)

# Comparative magnitude phrases.
_COMP_RE = re.compile(
    r"(much\s+(?:stronger|weaker|larger|smaller)|order\s+of\s+magnitude\s+"
    r"(?:stronger|weaker|larger|smaller)?)",
    re.IGNORECASE,
)


def split_sentences(text: str) -> list[tuple[str, int]]:
    """Split LaTeX prose into rough sentences.

    Returns list of (sentence_text, start_offset_in_text) tuples. Splits on
    `[.!?]` followed by whitespace and an uppercase letter or backslash,
    or on a paragraph break (blank line). Good enough for the manuscript's
    prose conventions — sentence ends are explicit.
    """
    boundaries: list[int] = [0]
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z\\])|\n\s*\n", text):
        boundaries.append(m.end())
    boundaries.append(len(text))

    out: list[tuple[str, int]] = []
    for i in range(len(boundaries) - 1):
        chunk = text[boundaries[i]:boundaries[i + 1]]
        stripped = chunk.strip()
        if not stripped:
            continue
        # Preserve the start offset relative to the original text, accounting
        # for any leading whitespace we stripped.
        leading_ws = len(chunk) - len(chunk.lstrip())
        out.append((stripped, boundaries[i] + leading_ws))
    return out


def offset_to_line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def mask_negations(s: str) -> str:
    """Replace negation bigrams with a neutral token so qualifier patterns
    don't fire on `not significant`, `economically significant`, etc."""
    out = s
    for bigram in NEGATION_BIGRAMS:
        out = re.sub(re.escape(bigram), "__NEG__", out, flags=re.IGNORECASE)
    return out


def parse_suppressions(line: str) -> set[str]:
    """Extract `% noqa: prose-lint(rule-a,rule-b)` rule names from a line."""
    m = NOQA_RE.search(line)
    if not m:
        return set()
    return {r.strip() for r in m.group(1).split(",") if r.strip()}


_BARE_MACRO_LIST_RE = re.compile(
    r"\s*\\[A-Za-z][A-Za-z0-9]*(?:\s*,\s*\\[A-Za-z][A-Za-z0-9]*)*\s*"
)


def find_t_stat_usages(sentence: str, macros: dict[str, str]) -> list[TStatUsage]:
    """Locate macro references in t-stat context.

    Two patterns are recognised:

    1. A math block whose body contains a `t = ` LHS, e.g. ``$t = \\foo$`` or
       ``$t_{\\mathrm{MC}}^{\\mathrm{HR}} = \\foo$``. Every ``\\macro`` inside the block is a
       t-stat usage.
    2. A *continuation* math block immediately following such a block, where
       the body is just a bare ``\\macro`` (or comma-separated list) and at
       most one word separates the two blocks. This handles the manuscript
       idioms ``($t = \\macroA$, $\\macroB$)`` and ``$t = \\macroA$ against
       $\\macroB$``; a clause between the blocks ends the context, so
       ``($t = \\foo$), and is significant on the union, at $\\bar$`` does not
       read ``\\bar`` as a t-statistic.
    """
    usages: list[TStatUsage] = []
    in_tstat_context = False
    prev_end = 0
    for math in _MATH_RE.finditer(sentence):
        body = math.group(1)
        body_start = math.start(1)
        has_t_lhs = bool(_T_LHS_RE.search(body))
        is_bare_list = bool(_BARE_MACRO_LIST_RE.fullmatch(body))
        if len(re.findall(r"[A-Za-z]+", sentence[prev_end:math.start()])) > 1:
            in_tstat_context = False
        prev_end = math.end()

        if has_t_lhs:
            in_tstat_context = True
            include = True
        elif is_bare_list and in_tstat_context:
            include = True
        else:
            in_tstat_context = False
            include = False

        if include:
            for mm in _MACRO_RE.finditer(body):
                name = mm.group(1)
                if name not in macros:
                    continue
                try:
                    val = parse_num(macros[name])
                except ValueError:
                    continue
                usages.append(TStatUsage(
                    macro=name,
                    offset=body_start + mm.start(),
                    value=val,
                ))
    return usages


def pair_qualifier(
    usages: list[TStatUsage],
    qual_start: int,
    qual_end: int,
    *,
    look_after: int = 80,
    look_before: int = 30,
) -> TStatUsage | None:
    """Pair a qualifier word with the t-stat macro it modifies.

    Manuscript convention places the macro in a parenthetical immediately
    *after* the qualifier (``insignificant ($t = \\foo$)``), so the linter
    prefers macros within ``look_after`` chars after the qualifier's match
    end. Falls back to ``look_before`` chars before the qualifier's match
    start. Returns ``None`` when neither window contains a t-stat — the
    qualifier is then treated as narrative and the rule does not fire.
    """
    if not usages:
        return None
    after = [u for u in usages if 0 <= u.offset - qual_end <= look_after]
    if after:
        return min(after, key=lambda u: u.offset - qual_end)
    before = [u for u in usages if 0 <= qual_start - u.offset <= look_before]
    if before:
        return max(before, key=lambda u: u.offset)
    return None


def pair_pvalue(
    usages: list[TStatUsage],
    p_start: int,
    *,
    look_before: int = 60,
    look_after: int = 30,
) -> TStatUsage | None:
    """Pair a ``$p<X$`` inequality with the t-stat macro it qualifies.

    Convention places the p-value *after* its t-stat (``$t = \\foo$,
    $p<0.001$``), so the linter prefers macros within ``look_before`` chars
    before the inequality. Returns ``None`` when no nearby macro exists.
    """
    if not usages:
        return None
    before = [u for u in usages if 0 <= p_start - u.offset <= look_before]
    if before:
        return max(before, key=lambda u: u.offset)
    after = [u for u in usages if 0 <= u.offset - p_start <= look_after]
    if after:
        return min(after, key=lambda u: u.offset - p_start)
    return None


# ---------------------------------------------------------------------------
# Predicate rules
# ---------------------------------------------------------------------------

def _check_insignificant(t: float) -> bool:
    """`insignificant` fails when |t| crosses the 5% critical value."""
    return abs(t) >= CRIT_5PCT


def _check_marginally_significant(t: float) -> bool:
    """`marginally significant` fails when result is either not significant at
    any conventional threshold (|t| < 10% critical) or highly significant at
    the 1% level (calling p<0.01 "marginal" misrepresents the result)."""
    a = abs(t)
    return (a < CRIT_10PCT) or (a >= CRIT_1PCT)


def _check_highly_significant(t: float) -> bool:
    """`highly significant` fails when |t| does not clear the 1% threshold."""
    return abs(t) < CRIT_1PCT


def _check_significant(t: float) -> bool:
    """Bare `significant` fails when |t| does not clear the 5% threshold."""
    return abs(t) < CRIT_5PCT


def _check_predicts_negative(t: float) -> bool:
    return t > -PREDICTS_THRESHOLD


def _check_predicts_positive(t: float) -> bool:
    return t < PREDICTS_THRESHOLD


def _check_strongly_positive(t: float) -> bool:
    return t < CRIT_5PCT


def _check_strongly_negative(t: float) -> bool:
    return t > -CRIT_5PCT


def _check_flips_positive(t: float) -> bool:
    """`flips positive` is a sign claim with 5% support — fails when t < 1.96."""
    return t < PREDICTS_THRESHOLD


def _check_flips_negative(t: float) -> bool:
    """`flips negative` is a sign claim with 5% support — fails when t > -1.96."""
    return t > -PREDICTS_THRESHOLD


# (qualifier regex pattern, predicate, rule name, message template)
QUALIFIER_RULES = [
    (
        re.compile(r"\b(?:highly\s+significant)\b", re.IGNORECASE),
        _check_highly_significant,
        "highly-significant",
        "qualifier 'highly significant' near \\{macro}={value:+.2f} but |t| < {crit:.2f}",
    ),
    (
        re.compile(r"\b(?:marginally\s+significant|marginal\s+significance)\b",
                   re.IGNORECASE),
        _check_marginally_significant,
        "marginally-significant",
        "qualifier 'marginally significant' near \\{macro}={value:+.2f} (p≈{p:.3f}); "
        "the result is either not significant at 10% or stronger than 1%",
    ),
    (
        re.compile(r"\b(?:insignificant|insignificance)\b", re.IGNORECASE),
        _check_insignificant,
        "insignificant",
        "qualifier 'insignificant' near \\{macro}={value:+.2f} but |t| ≥ {crit:.2f} "
        "(p≈{p:.4f}); result is significant at 5%",
    ),
    (
        re.compile(r"\bsignificantly\b|\bsignificant\b", re.IGNORECASE),
        _check_significant,
        "significant",
        "qualifier 'significant' near \\{macro}={value:+.2f} but |t| < {crit:.2f} "
        "(p≈{p:.3f}); result is not significant at 5%",
    ),
    (
        re.compile(r"\bpredicts?\b[^.]{0,80}?\bnegatively\b", re.IGNORECASE),
        _check_predicts_negative,
        "predicts-negative",
        "claim 'predicts ... negatively' near \\{macro}={value:+.2f}; "
        "requires t ≤ {neg_crit:.2f} for a 5% two-sided directional claim",
    ),
    (
        re.compile(r"\bpredicts?\b[^.]{0,80}?\bpositively\b", re.IGNORECASE),
        _check_predicts_positive,
        "predicts-positive",
        "claim 'predicts ... positively' near \\{macro}={value:+.2f}; "
        "requires t ≥ {pos_crit:.2f} for a 5% two-sided directional claim",
    ),
    (
        re.compile(r"\bstrongly\s+positively?\b", re.IGNORECASE),
        _check_strongly_positive,
        "strongly-positive",
        "qualifier 'strongly positive' near \\{macro}={value:+.2f}; "
        "requires t ≥ {pos_crit:.2f}",
    ),
    (
        re.compile(r"\bstrongly\s+negatively?\b", re.IGNORECASE),
        _check_strongly_negative,
        "strongly-negative",
        "qualifier 'strongly negative' near \\{macro}={value:+.2f}; "
        "requires t ≤ {neg_crit:.2f}",
    ),
    (
        re.compile(r"\bflip(?:s|ping|ped)?\s+positive\b", re.IGNORECASE),
        _check_flips_positive,
        "flips-positive",
        "qualifier 'flips positive' near \\{macro}={value:+.2f}; "
        "requires t ≥ {pos_crit:.2f}",
    ),
    (
        re.compile(r"\bflip(?:s|ping|ped)?\s+negative\b", re.IGNORECASE),
        _check_flips_negative,
        "flips-negative",
        "qualifier 'flips negative' near \\{macro}={value:+.2f}; "
        "requires t ≤ {neg_crit:.2f}",
    ),
]


# ---------------------------------------------------------------------------
# Sentence-level scan
# ---------------------------------------------------------------------------

def _format_message(template: str, *, macro: str, value: float) -> str:
    return template.format(
        macro=macro, value=value, p=t_to_p(value),
        crit=CRIT_5PCT, neg_crit=-PREDICTS_THRESHOLD, pos_crit=PREDICTS_THRESHOLD,
    )


@dataclass
class _SentenceCtx:
    """Per-sentence scan state shared by the three rule checkers.

    Bundles the sentence, its offset into the file, and the precomputed
    t-stat usages / negation mask so each `_check_*` helper takes one
    argument instead of six. `file_lines` is cached once to keep the
    per-offset `line_text_for` lookups O(1).
    """

    sentence: str
    sentence_offset: int
    text: str
    file: Path
    usages: list[TStatUsage]
    masked: str

    def __post_init__(self) -> None:
        self.file_lines = self.text.splitlines()

    def lineno(self, off_in_sentence: int) -> int:
        return offset_to_line(self.text, self.sentence_offset + off_in_sentence)

    def line_text_for(self, off_in_sentence: int) -> str:
        idx = self.lineno(off_in_sentence) - 1
        return self.file_lines[idx] if 0 <= idx < len(self.file_lines) else ""


def _check_qualifier_rules(ctx: _SentenceCtx) -> list[Violation]:
    """Qualifier words (`insignificant`, `highly significant`, ...) vs. value."""
    out: list[Violation] = []
    if not ctx.usages:
        return out
    for pattern, predicate, rule, template in QUALIFIER_RULES:
        for qm in pattern.finditer(ctx.masked):
            u = pair_qualifier(ctx.usages, qm.start(), qm.end())
            if u is None:
                continue
            if not predicate(u.value):
                continue
            if rule in parse_suppressions(ctx.line_text_for(u.offset)):
                continue
            out.append(Violation(
                severity="error",
                file=ctx.file,
                line=ctx.lineno(qm.start()),
                rule=rule,
                message=_format_message(template, macro=u.macro, value=u.value),
            ))
    return out


def _check_pvalue_inequalities(ctx: _SentenceCtx) -> list[Violation]:
    """`$p < X$` claims vs. the p-value implied by the paired t-stat."""
    out: list[Violation] = []
    for pm in _PVAL_INEQ_RE.finditer(ctx.sentence):
        digits = pm.group(1)
        try:
            claim_p = float("0." + digits)
        except ValueError:
            continue
        u = pair_pvalue(ctx.usages, pm.start())
        if u is None:
            continue
        actual_p = t_to_p(u.value)
        line_text = ctx.line_text_for(u.offset)
        if actual_p >= claim_p:
            if "p-value-violated" in parse_suppressions(line_text):
                continue
            out.append(Violation(
                severity="error",
                file=ctx.file,
                line=ctx.lineno(pm.start()),
                rule="p-value-violated",
                message=(
                    f"claim 'p<{claim_p:g}' near \\{u.macro}={u.value:+.2f} "
                    f"but actual p≈{actual_p:.4f}"
                ),
            ))
            continue
        # Warn only if a *strictly tighter* conventional bound applies. Skip
        # when the claim already sits at the tightest displayable threshold
        # (0.001) — repeating "could state p<0.001" when the author already
        # wrote that is noise.
        tighter = _tightest_p_bound(actual_p)
        if tighter < claim_p and claim_p / max(actual_p, 1e-300) >= LOOSE_P_CLAIM_FACTOR:
            if "p-value-loose" in parse_suppressions(line_text):
                continue
            out.append(Violation(
                severity="warning",
                file=ctx.file,
                line=ctx.lineno(pm.start()),
                rule="p-value-loose",
                message=(
                    f"claim 'p<{claim_p:g}' near \\{u.macro}={u.value:+.2f} "
                    f"is loose (actual p≈{actual_p:.4g}, "
                    f"could state p<{tighter:g})"
                ),
            ))
    return out


def _check_comparative_magnitude(ctx: _SentenceCtx) -> list[Violation]:
    """Two-macro comparatives (`much stronger than`, `order of magnitude`)."""
    out: list[Violation] = []
    for cm in _COMP_RE.finditer(ctx.masked):
        phrase = cm.group(1).lower()
        # Need at least two t-stat macros in this sentence; pick the two
        # whose offsets straddle the comparative phrase. Order matters: the
        # first should be the stronger one.
        if len(ctx.usages) < 2:
            out.append(Violation(
                severity="info",
                file=ctx.file,
                line=ctx.lineno(cm.start()),
                rule="comparative-ambiguous",
                message=(
                    f"comparative '{cm.group(1)}' near sentence with "
                    f"{len(ctx.usages)} t-stat macro(s); cannot verify ratio"
                ),
            ))
            continue

        # Heuristic pairing: take the two macros bracketing the phrase (or
        # the two nearest if both are on one side).
        sorted_by_pos = sorted(ctx.usages, key=lambda u: u.offset)
        before = [u for u in sorted_by_pos if u.offset < cm.start()]
        after = [u for u in sorted_by_pos if u.offset >= cm.start()]
        if before and after:
            a, b = before[-1], after[0]
        elif len(after) >= 2:
            a, b = after[0], after[1]
        elif len(before) >= 2:
            a, b = before[-2], before[-1]
        else:
            continue

        if "order of magnitude" in phrase:
            ratio = ORDER_OF_MAGNITUDE_RATIO
            rule = "comparative-order-of-magnitude"
        else:
            ratio = MUCH_STRONGER_RATIO
            rule = "comparative-much-stronger"

        if abs(a.value) == 0 or abs(b.value) == 0:
            continue
        observed = max(abs(a.value), abs(b.value)) / min(abs(a.value), abs(b.value))
        macro_line = ctx.lineno(min(a.offset, b.offset))
        line_text = ctx.text.splitlines()[macro_line - 1] if macro_line - 1 < ctx.text.count("\n") + 1 else ""
        if observed < ratio:
            if rule in parse_suppressions(line_text):
                continue
            out.append(Violation(
                severity="error",
                file=ctx.file,
                line=ctx.lineno(cm.start()),
                rule=rule,
                message=(
                    f"claim '{cm.group(1)}' between \\{a.macro}={a.value:+.2f} "
                    f"and \\{b.macro}={b.value:+.2f}; |t|-ratio "
                    f"{observed:.2f} < required {ratio:.1f}"
                ),
            ))
    return out


def lint_sentence(
    sentence: str,
    sentence_offset: int,
    text: str,
    file: Path,
    macros: dict[str, str],
) -> list[Violation]:
    """Run all rules against one sentence and return any violations."""
    ctx = _SentenceCtx(
        sentence=sentence,
        sentence_offset=sentence_offset,
        text=text,
        file=file,
        usages=find_t_stat_usages(sentence, macros),
        masked=mask_negations(sentence),
    )
    out: list[Violation] = []
    out += _check_qualifier_rules(ctx)
    out += _check_pvalue_inequalities(ctx)
    out += _check_comparative_magnitude(ctx)
    return out


def _tightest_p_bound(p: float) -> float:
    """Return the tightest of {0.001, 0.01, 0.05, 0.10} that p satisfies."""
    for b in (0.001, 0.01, 0.05, 0.10):
        if p < b:
            return b
    return 1.0


# ---------------------------------------------------------------------------
# Register rules: the repo register, and terms of art the reader cannot check
# ---------------------------------------------------------------------------

# Text the reader never sees, blanked before the register scan with offsets
# preserved: labels, cross-references, citation keys, included paths, math,
# typewriter filenames, and LaTeX comments. Register is a property of the
# rendered sentence, so a repo word inside `\label{...}` is not a violation.
_NON_PROSE_RE = re.compile(
    r"\\(?:label|ref|eqref|cite[a-z]*|input|includegraphics|bibliography"
    r"|usepackage|newcommand|texttt|url|href)\s*(?:\[[^\]]*\])?\{[^}]*\}"
    r"|\$[^$]*\$"
    r"|(?<!\\)%.*"
)

REGISTER_RULES = [
    # The repo's word for a run
    # configuration, for its primary result, and for the estimation window.
    # Each reached the manuscript at least once.
    ("repo-register",
     re.compile(r"\barms?\b|\bheadlines?\b|\bpublished\s+(?:window|sample)\b",
                re.IGNORECASE),
     "repo register in manuscript prose"),
    # Dof and baseline tag names,
    # ALL-CAPS so ordinary English is not hit.
    ("repo-register",
     re.compile(r"\b(?:REPRODUCED|MEASURED|CHOSEN|FITTED|BASELINE)\b"),
     "repo status token in manuscript prose"),
    # A chronology the reader cannot audit. The exit is to
    # cut the timing and keep the criterion, which binds by being stated and
    # met.
    ("term-of-art",
     re.compile(r"\bpre-?registered\b|\bpre-?registration\b|\bpre-?specified\b"
                r"|\b(?:fixed|published|decided|chosen|committed|set)\s+in\s+advance\b"
                r"|\bbefore\s+(?:any|the)\s+(?:return|data|result)\w*"
                r"\s+(?:is|was|are|were)\s+examined\b"
                r"|\b(?:committed|fixed|registered|decided|chosen)\s+before\s+"
                r"the\s+(?:run|data|results?|split)\b",
                re.IGNORECASE),
     "timing claim the reader cannot audit"),
]


def _check_register(text: str, file: Path) -> list[Violation]:
    """Flag repo-internal register and unauditable terms of art in prose."""
    prose = _NON_PROSE_RE.sub(lambda m: " " * len(m.group(0)), text)
    lines = text.splitlines()
    out: list[Violation] = []
    for rule, pattern, message in REGISTER_RULES:
        for m in pattern.finditer(prose):
            line = offset_to_line(prose, m.start())
            if rule in parse_suppressions(lines[line - 1]):
                continue
            out.append(Violation("error", file, line, rule,
                                 f"{message}: {m.group(0)!r}"))
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lint_prose_qualifiers(macros: dict[str, str]) -> list[Violation]:
    """Scan every manuscript file and return all violations."""
    all_violations: list[Violation] = []
    for path in MANUSCRIPT_FILES:
        text = path.read_text()
        all_violations.extend(_check_register(text, path))
        for sentence, sent_off in split_sentences(text):
            all_violations.extend(
                lint_sentence(sentence, sent_off, text, path, macros)
            )
    return all_violations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    macros = load_macros()
    violations = lint_prose_qualifiers(macros)
    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]
    infos = [v for v in violations if v.severity == "info"]

    for v in errors:
        print(v)
    for v in warnings:
        print(v)
    for v in infos:
        print(v)

    print(
        f"\nprose-lint: {len(errors)} error(s), {len(warnings)} warning(s), "
        f"{len(infos)} info"
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
