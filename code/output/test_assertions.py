#!/usr/bin/env python3
"""
Assertion suite for the manuscript: every number and claim the paper prints
is checked against ``results/`` and against the repository.

The checks live under ``assertions/``, one module per code directory whose
results they read, or a package of them for ``return_prediction``,
``signal_structure`` and ``robustness``, plus
``code/output/assertions/manuscript.py`` for checks that read only the documents and
``code/output/assertions/repository.py`` for the gates over the tree.
``code/output/assertions/registry.py`` lists them in the order they run.

Exits 0 on success and prints "OK". Raises AssertionError with the offending
macro/value pair on the first violation.

Run:
    cd code && PYTHONPATH=. python output/test_assertions.py
"""

from __future__ import annotations

from pathlib import Path

from paths import REPO_ROOT

from output.assertions.common import NUMBERS, ROOT, load_macros
from output.assertions.registry import MACRO_CHECKS, PROSE_CHECKS


def check_paths_module_is_this_repo() -> None:
    """The suite must be reading this repository's ``paths``, not another's.

    Both manuscripts in this line of work ship a ``code/paths.py``, and an
    editable install of either one puts its ``code/`` directory on
    ``sys.path`` for every interpreter in the environment. A suite run without
    ``PYTHONPATH=.`` then resolves ``paths`` to the other repository and
    verifies its results files against this repository's macros, or passes
    every check while reading numbers no sentence here quotes. The failure is
    invisible after the fact, which is what this rules out: nothing else in
    the suite can tell which repository's numbers were verified.
    """
    import paths
    here = Path(__file__).resolve().parents[2]
    assert Path(paths.__file__).resolve().is_relative_to(here), (
        f"'paths' resolved to {paths.__file__}, outside this repository "
        f"({here}). Run the suite as the Makefile does, from code/ with "
        f"PYTHONPATH=. set, so the sibling manuscript's editable install "
        f"cannot shadow it."
    )
    assert REPO_ROOT.resolve() == here, (
        f"paths.REPO_ROOT is {REPO_ROOT}, not this repository ({here})."
    )


def main() -> int:
    check_paths_module_is_this_repo()
    m = load_macros()
    print(f"Loaded {len(m)} macros from {NUMBERS.relative_to(ROOT)}")
    for label, fn in MACRO_CHECKS:
        print(f"  {label} ... ", end="", flush=True)
        fn(m)
        print("ok")
    print("OK: numbers.tex ↔ results/ manifest agrees on the verifiable subset.")
    print()
    print("Manuscript prose claims:")
    for label, fn in PROSE_CHECKS:
        print(f"  {label} ... ", end="", flush=True)
        fn(m)
        print("ok")
    print("OK: manuscript prose claims agree with results/ within tolerance.")
    print()
    print("Manuscript prose-adjacency lint:")
    from output._prose_lint import lint_prose_qualifiers
    violations = lint_prose_qualifiers(m)
    for v in violations:
        print(f"  {v}")
    errors = [v for v in violations if v.severity == "error"]
    if errors:
        raise SystemExit(
            f"FAIL: {len(errors)} prose-lint error(s); see above."
        )
    print("OK: prose qualifiers agree with adjacent macro values.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
