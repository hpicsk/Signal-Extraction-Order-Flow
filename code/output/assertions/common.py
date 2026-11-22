"""Paths, the macro loader and the assertion helpers every check module uses."""

from __future__ import annotations

import re

from paths import REPO_ROOT, RESULTS_DIR, TABLES_DIR


ROOT = REPO_ROOT
RESULTS = RESULTS_DIR
NUMBERS = TABLES_DIR / "numbers.tex"
PAPER_TEX = ROOT / "qf_paper.tex"
# The documents a sentence can sit in: the article, its appendices bound in.
MANUSCRIPTS = (PAPER_TEX,)
_MACRO_RE = re.compile(r"\\newcommand\{\\(\w+)\}\{")


def submission_tex(tables: bool = True) -> list:
    """Every .tex file this submission carries: the article and the
    generated table bodies when ``tables`` is set."""
    found = sorted(ROOT.glob("*.tex"))
    if tables:
        found += sorted(TABLES_DIR.glob("*.tex"))
    return found


def load_macros() -> dict[str, str]:
    """Parse \\newcommand{\\name}{value} with brace-balanced value extraction."""
    text = NUMBERS.read_text()
    out: dict[str, str] = {}
    for match in _MACRO_RE.finditer(text):
        name = match.group(1)
        i = match.end()  # position right after the opening { of the value
        depth = 1
        start = i
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        out[name] = text[start:i]
    return out


def parse_num(s: str) -> float:
    # Strip LaTeX thin-space thousands separator before plain commas, since
    # ``\,`` is a single backslash + comma sequence and would be eaten if
    # ``,`` is removed first.
    cleaned = (
        s.replace("\\,", "").replace("{,}", "").replace(",", "")
         .replace("$", "").replace("+", "").strip()
    )
    # Handle LaTeX scientific notation: "3.5 \times 10^{-5}" → 3.5e-5
    m = re.fullmatch(
        r"\s*(-?\d+(?:\.\d+)?)\s*\\times\s*10\^\{?(-?\d+)\}?\s*",
        cleaned,
    )
    if m:
        return float(m.group(1)) * (10.0 ** int(m.group(2)))
    return float(cleaned)


def _find_manuscript_sentence(macro_label: str) -> str:
    """Locate the manuscript line(s) referencing ``\\macroLabel`` so the
    AssertionError can quote the prose the macro materialises."""
    needle = f"\\{macro_label}"
    hits = []
    for tex in MANUSCRIPTS:
        for i, line in enumerate(tex.read_text().splitlines(), start=1):
            if needle in line:
                hits.append(f"{tex.relative_to(ROOT)}:{i}: {line.strip()}")
                if len(hits) >= 2:
                    break
    return "\n  ".join(hits) if hits else "<no manuscript sentence references this macro>"


def assert_close(macro_value: str, data_value: float, *, tol: float, label: str) -> None:
    mv = parse_num(macro_value)
    if abs(mv - float(data_value)) > tol:
        sentence = _find_manuscript_sentence(label)
        raise AssertionError(
            f"Macro \\{label} = {mv} but data = {data_value:.6f} "
            f"(|diff|={abs(mv-data_value):.6f} > tol={tol}).\n"
            f"  Manuscript sentence(s) depending on this macro:\n  {sentence}"
        )


# Prose-claim helpers (no LaTeX parsing — both sides are raw floats / bools).
def _close(a: float, b: float, *, tol: float) -> bool:
    return abs(float(a) - float(b)) <= tol


def assert_prose_close(actual: float, expected: float, *, tol: float, claim: str) -> None:
    if not _close(actual, expected, tol=tol):
        raise AssertionError(
            f"Manuscript claim: \"{claim}\" -- expected ~{expected}, "
            f"got {float(actual):.6g} (|diff|={abs(float(actual)-float(expected)):.6g} > tol={tol})"
        )


def assert_true(cond: bool, *, claim: str, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"Manuscript claim: \"{claim}\" -- violated. {detail}")
