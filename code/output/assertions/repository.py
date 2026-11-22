"""Gates over the repository itself: the generator, the macro inventory, the
results tree, the paths and label slugs repo documents cite, script headers
and the commit history."""

from __future__ import annotations

import re
from pathlib import Path

from paths import REPO_ROOT, TABLES_DIR

from output.assertions.common import (
    MANUSCRIPTS,
    NUMBERS,
    ROOT,
    _MACRO_RE,
    assert_true,
    submission_tex,
)


# Paths a repo document names, for check_document_paths_resolve. Anchored on the
# three top-level directories a document has reason to point into, so prose that
# merely contains a slash is not mistaken for a path.
PATHLIKE = re.compile(r"(?<![\w/.-])((?:code|results|tables|figures)/[\w./*-]*[\w*])")


def _diff_line(regenerated: str, committed: str) -> tuple[int, str, str]:
    """Return (1-indexed line, committed_line, regenerated_line) for the first
    differing line, or (max_len, '<EOF>', '<EOF>') if one is a prefix of the
    other."""
    new = regenerated.splitlines()
    old = committed.splitlines()
    for i in range(min(len(new), len(old))):
        if new[i] != old[i]:
            return i + 1, old[i], new[i]
    i = min(len(new), len(old))
    return (
        i + 1,
        old[i] if i < len(old) else "<EOF>",
        new[i] if i < len(new) else "<EOF>",
    )


def check_generator_idempotent(m: dict[str, str]) -> None:
    """Covers: governance (all auto-generated macros, cross-section).

    Re-run generate_table_bodies in-memory and assert
    every generated file (tables/numbers.tex AND every body .tex file) reproduces the
    committed file byte-for-byte.

    Since tables/numbers.tex and the body files are 100% auto-generated, this single
    assertion verifies every auto-generated macro AND every auto-generated
    table cell against its source results file via the generator's own logic.
    Manual-region macros must still have explicit check_* assertions above and
    are enforced by check_macro_coverage_gate below.

    Captures every call to ``gen.write_body`` rather than only ``tables/numbers.tex``
    so a drift in any body file (a table body's loading cell, say)
    fires here instead of slipping silently into the typeset PDF.
    """
    import importlib
    import sys
    code_dir = str(REPO_ROOT / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    gen = importlib.import_module("output.generate_table_bodies")
    captured: dict[str, str] = {}
    original_write_body = gen.write_body

    def _capture(filename, content):
        captured[filename] = content

    import contextlib
    import io
    gen.write_body = _capture
    try:
        # Run the generator's own main() so every generate_*_body() helper
        # plus generate_numbers_tex() fires; captured collects every file the
        # generator would have written. Stdout is suppressed because main()
        # prints per-table progress that would clutter the test log.
        with contextlib.redirect_stdout(io.StringIO()):
            gen.main()
    finally:
        gen.write_body = original_write_body

    if "numbers.tex" not in captured:
        raise AssertionError(
            "generate_numbers_tex() did not write numbers.tex via write_body")

    drifted: list[tuple[str, int, str, str]] = []
    for filename, regenerated in sorted(captured.items()):
        committed_path = TABLES_DIR / filename
        if not committed_path.exists():
            drifted.append((filename, 0, "<missing on disk>", "<generated>"))
            continue
        committed = committed_path.read_text()
        if regenerated != committed:
            lineno, old_line, new_line = _diff_line(regenerated, committed)
            drifted.append((filename, lineno, old_line, new_line))

    if drifted:
        lines = [
            f"{len(drifted)} regenerated file(s) differ from committed contents:"
        ]
        for filename, lineno, old_line, new_line in drifted[:5]:
            lines.append(f"  {filename} (line {lineno}):")
            lines.append(f"    committed:   {old_line}")
            lines.append(f"    regenerated: {new_line}")
        if len(drifted) > 5:
            lines.append(f"  … and {len(drifted) - 5} more.")
        lines.append(
            "(Run generate_table_bodies.py to refresh, then commit the diff.)")
        raise AssertionError("\n".join(lines))


def check_every_macro_is_referenced(_m: dict[str, str]) -> None:
    """Covers: no dead code, applied to tables/numbers.tex.

    A macro exists to be printed. One that no ``.tex`` source references is a
    leftover from prose that has since been cut, and it survives silently
    because the generator regenerates it on every build and the harness's other
    checks read the results file, not the manuscript. check_macro_coverage_gate
    below gates the *manual* region instead, which is empty by construction, so
    it passed while 61 macros sat unreferenced — among them both arms of a
    winsorised horse race whose result qualifies a claim in
    subsec:return_prediction.

    The rule is exact: every ``\\newcommand`` in tables/numbers.tex is used at least
    once outside its own definition, either by the manuscript or by a generated
    table body. If prose stops quoting a value, the emitter goes
    with it in the same commit.
    """
    defined = set(_MACRO_RE.findall(NUMBERS.read_text()))
    used: set[str] = set()
    for tex in submission_tex():
        body = tex.read_text()
        if tex.name == "numbers.tex":
            # Strip the definition heads so a macro does not count as its own use.
            body = re.sub(r"\\newcommand\{\\[A-Za-z]+\}", "", body)
        used.update(re.findall(r"\\([A-Za-z]+)", body))

    orphans = sorted(defined - used)
    assert not orphans, (
        f"{len(orphans)} macro(s) in numbers.tex are referenced by no .tex "
        f"source: {orphans}\n"
        f"Delete the emitter in code/output/ (and any assertion that reads the "
        f"macro) in the same commit as the prose that stopped quoting it.")


def check_macro_coverage_gate(m: dict[str, str]) -> None:
    """Covers: governance (manual-region macros, no manuscript section).

    Coverage gate for **manual-region** macros only.

    Auto-generated macros are verified in bulk by check_generator_idempotent
    above, which regenerates every one of them and diffs. A macro is "manual"
    if it appears in tables/numbers.tex but is NOT produced by re-running
    generate_numbers_tex() in-memory. Every manual macro must be referenced by
    some function in MACRO_CHECKS or whitelisted explicitly.

    tables/numbers.tex has no manual region today -- the file's own header says do not
    edit -- so ``manual_names`` is empty and this gate passes vacuously. That is
    the intended resting state, not a defect: it is a tripwire for a future
    commit that hand-edits a value in, which no other check would see. Keep the
    docstring free of a macro count, since a count here is a second copy of a
    number the file already holds and drifts silently against it (this one read
    333 for a long while at 892).

    The whitelist is checked against tables/numbers.tex for the same reason. A
    whitelist entry naming a macro that no longer exists exempts nothing, and
    is the record of a rename nobody finished.
    """
    import importlib, sys
    code_dir = str(REPO_ROOT / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)
    gen = importlib.import_module("output.generate_table_bodies")
    captured: dict[str, str] = {}
    original_write_body = gen.write_body
    gen.write_body = lambda fn, c: captured.__setitem__(fn, c)
    try:
        gen.generate_numbers_tex()
    finally:
        gen.write_body = original_write_body
    auto_names = set(re.findall(r"\\newcommand\{\\(\w+)\}",
                                captured.get("numbers.tex", "")))
    manual_names = set(m) - auto_names

    import inspect
    # The registry imports this module, so it is read here, at call time.
    from output.assertions.registry import MACRO_CHECKS
    referenced: set[str] = set()
    for _label, fn in MACRO_CHECKS:
        if fn is check_macro_coverage_gate:
            continue
        src = inspect.getsource(fn)
        referenced.update(re.findall(r'm\[(?:f?)"(\w+)"\]', src))
        referenced.update(re.findall(r'm\.get\("(\w+)"', src))
    stale = sorted(_COVERAGE_WHITELIST - set(m))
    assert not stale, (
        f"{len(stale)} _COVERAGE_WHITELIST entries name macros numbers.tex no "
        f"longer defines, so they exempt nothing: {stale}. Delete them, or "
        f"restore the macro if the rename was unintended.")
    uncovered = sorted(manual_names - referenced - _COVERAGE_WHITELIST)
    if uncovered:
        raise AssertionError(
            f"{len(uncovered)} manual-region macros lack a check_* assertion "
            f"or whitelist entry: {uncovered[:10]}"
            f"{'…' if len(uncovered) > 10 else ''}\n"
            f"Add an assertion to a function in MACRO_CHECKS, or extend "
            f"_COVERAGE_WHITELIST in output/assertions/repository.py.")


# Macros that are intentionally not numerically asserted:
#  - Display-only macros (sample counts, "approx million" formatting, year ranges)
#  - Macros that mirror values already asserted under a different macro name
#    (e.g., the auto-generated body files compose macros that are leaves elsewhere)
#  - Macros that are derived from other already-asserted macros within the
#    manuscript text (e.g., ratios computed from t-stats that both have macros)
# This list grows whenever a new generator macro is intentionally display-only.
_COVERAGE_WHITELIST: set[str] = {
    # Year-range strings — structurally validated by the endpoints
    "ryYearsCovered", "fryYearsCovered",
    "ryNYears",
    # Display-only sample counts (load-bearing equivalents are sampForeignObs etc.)
    "sampPPDays", "sampTotalDays", "sampForeignObs",
    "panelTotalObsApprox",
}


def check_no_orphan_results(_m: dict[str, str]) -> None:
    """Covers: every committed artifact is resolved by something.

    An artifact committed under ``results/`` that no script writes and no
    checker reads is an orphan: it is a valid-looking file that a future reader
    can bind to, holding whatever run last produced it. A human-readable dump
    of a file the generator consumes, kept in step by hand, is the usual case.

    Derived from version control on both ends: the committed list from
    ``git ls-files``, the resolvers from every ``.py`` in the tree. A file
    referenced only through a composed name (``f"{prefix}_sample_size.csv"``)
    still has to appear as a literal somewhere -- the readers spell it out --
    so a genuine miss here means naming it once, not exempting it.
    """
    import subprocess

    committed = subprocess.run(
        ["git", "ls-files", "results/"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    assert committed, "git ls-files results/ returned nothing; is this a clone?"
    blob = "\n".join(
        p.read_text() for p in sorted((ROOT / "code").rglob("*.py")))
    orphans = [c for c in committed if Path(c).name not in blob]
    assert not orphans, (
        "Committed under results/ but resolved "
        "by no script and "
        "no assertion, so nothing regenerates or checks them: "
        + ", ".join(orphans))


def check_document_paths_resolve(_m: dict[str, str]) -> None:
    """Covers: every path a repo document names must resolve.

    A README command, a Makefile recipe, a bibliography comment: no program
    dereferences any of them, so a renamed or deleted script leaves them
    pointing at nothing with this suite still green.

    The document list is derived from version control rather than written down,
    so a new committed document is covered the day it lands. Paths git ignores
    are skipped: they are generated, so they resolve on a machine that has run
    the pipeline and not on a fresh clone, and requiring them would make this
    check depend on which stages have run.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "*.md", "Makefile"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    ignored = set(subprocess.run(
        ["git", "check-ignore", "--stdin"], cwd=ROOT, input="\n".join(
            m for d in tracked
            for m in PATHLIKE.findall((ROOT / d).read_text())),
        capture_output=True, text=True).stdout.split())

    missing = []
    for doc in tracked:
        for named in sorted(set(PATHLIKE.findall((ROOT / doc).read_text()))):
            if named in ignored:
                continue
            found = list(ROOT.glob(named)) if "*" in named else \
                ([ROOT / named] if (ROOT / named).exists() else [])
            if not found:
                missing.append(f"{doc}: {named}")
    assert not missing, (
        "A repo document names a path that "
        "does not resolve. "
        "Nothing dereferences these, so only this check sees them: "
        + "; ".join(missing))


def check_manuscript_file_names_resolve(_m: dict[str, str]) -> None:
    """Covers: a file name a manuscript cites must resolve.

    ``check_document_paths_resolve`` reads the Markdown documents and the
    Makefile, which spell a path from the repository root. A manuscript names
    a file the way its reader meets it, as a bare name in ``\\texttt`` with
    every underscore escaped, so ``PATHLIKE`` never sees it, and a renamed
    script leaves the manuscript naming the old file.

    Both sets are derived from version control: the names from every tracked
    ``.tex`` source, the files from everything tracked under ``code/`` and
    ``results/``. A ``\\texttt`` token counts as a file name when it ends in an
    extension the pipeline writes or runs.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
    files = {Path(p).name for p in tracked if p.startswith(("code/", "results/"))}
    missing = []
    for src in (p for p in tracked if p.endswith(".tex")):
        for token in re.findall(r"\\texttt\{([^}]*)\}", (ROOT / src).read_text()):
            name = token.replace("\\_", "_")
            if re.fullmatch(r"[\w.-]+\.(?:py|json|csv|parquet)", name) and name not in files:
                missing.append(f"{src}: {name}")
    assert not missing, (
        "A manuscript names a file that no "
        "tracked file under "
        "code/ or results/ carries, so a reader following it finds nothing: "
        + "; ".join(missing))


def check_readme_names_every_external_root(_m: dict[str, str]) -> None:
    """Covers: every name a repo file uses must resolve, both ends derived.

    ``check_document_paths_resolve`` reads the referring end and stops at the
    repository boundary: ``PATHLIKE`` is anchored on ``code/``, ``results/``,
    ``tables/`` and ``figures/``, so a tree the pipeline reads from outside the
    repository is invisible to it. The external trees are exactly the ones a
    replicator has to supply, and the README's list of them is hand-written,
    which is the shape that is true when written and silently false afterwards
    (**derive, never declare**). It went false at a vendor tree that ``make
    data`` reads and the README never asked a replicator to supply, so the
    replicator hit a ``RuntimeError`` naming a subpath they had never been
    told about.

    The subject set is derived from ``code/paths.py`` rather than restated here, so
    a tree added to the pipeline acquires the obligation the day it lands.
    """
    import ast

    src = (ROOT / "code" / "paths.py").read_text()
    roots: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(getattr(t, "id", None) == "_EXTERNAL_ROOTS" for t in node.targets):
            continue
        roots = [v.value for v in node.value.values]
    assert roots, "paths.py no longer declares _EXTERNAL_ROOTS; this check is blind"

    readme = (ROOT / "README.md").read_text()
    unnamed = [r for r in roots if r not in readme]
    assert not unnamed, (
        "code/paths.py makes the pipeline read an "
        "external tree the "
        "README never tells a replicator to supply, so the documented layout "
        "is incomplete and only this check sees it: " + ", ".join(unnamed))


def check_publishing_scripts_account_for_dof(_m: dict[str, str]) -> None:
    """Covers: the degrees-of-freedom accounting and the baseline tag of every
    publishing script.

    Every script whose numbers reach the paper must state, in its own header,
    which of its parameters were MEASURED, which CHOSEN, and which FITTED, and
    must name the alternative a referee gets in one sentence under a
    ``BASELINE:`` tag. Nothing else in this suite can see that obligation: a
    script with no accounting still emits a macro that binds, so every other
    check stays green.

    The subject set is derived, never declared. A script is *publishing* when
    it writes into ``results/`` -- it calls one of the save helpers -- and is
    run as a program rather than imported as a library, which is what the
    ``__main__`` guard distinguishes. A new analysis script therefore acquires
    the obligation the day it is added, which is the failure a hand-written
    list is written to prevent and always misses by one file.

    Filename matching is deliberately avoided: a script that composes its
    output names from a prefix would be exempt from a check that mapped read
    artifacts back to their writers by literal.

    A save reached through a helper is still a save: a predicate reading only
    the calls in the module itself exempts a script that writes through a
    helper defined in another module. The
    helper names are resolved first, across the tree, and a call to one counts
    as a save. Two directories are excluded in the predicate rather than in a
    comment, because the exclusion is a decision: ``code/output/`` is the
    generator layer, which writes the tables and macros the paper reads and
    accounts for no parameters of its own, and ``*_figure.py`` renders what a
    publishing script already measured.
    """
    import ast

    SAVERS = {"save_json", "save_csv", "to_parquet", "to_csv"}

    def direct_saves(tree: ast.AST) -> bool:
        return any(
            isinstance(n, ast.Call)
            and (n.func.attr if isinstance(n.func, ast.Attribute)
                 else getattr(n.func, "id", "")) in SAVERS
            for n in ast.walk(tree))

    sources = {}
    for py in sorted((ROOT / "code").rglob("*.py")):
        sources[py] = ast.parse(py.read_text())

    def main_guard(tree: ast.AST):
        return [n for n in tree.body
                if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                and isinstance(n.test.left, ast.Name)
                and n.test.left.id == "__name__"]

    # A name a __main__ guard calls is an entry point, not a save helper: an
    # orchestrator that runs another script's ``main`` delegates the writing,
    # and the callee carries its own accounting. Without this the predicate
    # reads every ``if __name__: main()`` as a save, because some module's
    # ``main`` does write.
    entry_points = {
        (c.func.attr if isinstance(c.func, ast.Attribute)
         else getattr(c.func, "id", ""))
        for tree in sources.values()
        for guard in main_guard(tree)
        for c in ast.walk(guard) if isinstance(c, ast.Call)
    }
    # Module-level functions anywhere under code/ whose own body reaches a
    # saver. Matched by name across the tree rather than resolved through
    # imports: the error that buys is a script wrongly asked to account for
    # itself, which fails loudly, and never a publishing script that slips.
    saving_helpers = {
        node.name
        for tree in sources.values()
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and direct_saves(node)
    } - entry_points
    SAVERS = SAVERS | saving_helpers

    def is_publishing(tree: ast.AST) -> bool:
        return direct_saves(tree) and bool(main_guard(tree))

    missing_dof, missing_baseline, undischarged = [], [], []
    for py, tree in sorted(sources.items()):
        parts = py.relative_to(ROOT).parts
        if "output" in parts or py.name.endswith("_figure.py"):
            continue
        if not is_publishing(tree):
            continue
        header = ast.get_docstring(tree) or ""
        rel = py.relative_to(ROOT).as_posix()
        if not any(t in header for t in ("MEASURED", "CHOSEN", "FITTED")):
            missing_dof.append(rel)
        if "BASELINE:" not in header:
            missing_baseline.append(rel)
        if "discharge: pending" in header:
            undischarged.append(rel)

    assert not missing_dof, (
        "A script publishing into results/ must tag every "
        "parameter MEASURED / CHOSEN / FITTED in its module docstring. "
        "Untagged: " + ", ".join(missing_dof))
    assert not missing_baseline, (
        "A script publishing into results/ must carry a "
        "BASELINE: tag naming the obvious alternative and its discharge. "
        "Untagged: " + ", ".join(missing_baseline))
    assert not undischarged, (
        "A pending BASELINE discharge under a shipped "
        "claim is "
        "an audit finding. Discharge it by argument or by run, or cut the claim "
        "it serves. Pending: " + ", ".join(undischarged))


def check_panel_readers_use_parquet(_m: dict[str, str]) -> None:
    """Covers: repo-integrity guard for the CSV->Parquet panel migration
    (a silent failure of the pipeline).

    The daily panel was migrated from CSV to Parquet; ``DAILY_DATA_ALL_INVESTORS``
    and ``DAILY_DATA_PRE2020`` now point at ``.parquet`` files. Every reader of
    those panels must use ``pd.read_parquet``. A stale ``pd.read_csv`` on a panel
    path is a hard crash on pipeline re-run, yet it survives silently because the
    cached ``results/*.json`` predate the rebuilt panel and this suite reads those
    JSONs, not the panel.

    The scan is AST-based, not line-based, because the panel constant routinely
    reaches ``read_csv`` *indirectly* — assigned to a module alias, used as a
    parameter default, or passed as a call argument. An earlier regex guard that
    only tracked module-level aliases missed two such readers
    (``_korea(path)`` in code/empirical/signal_structure/flow_persistence_decomposition.py and a
    ``load_panel(path=INPUT_PATH)`` default in a scaling script). This
    taint-propagation scan follows the constant through all three indirections.
    """
    import ast

    from paths import DAILY_DATA_ALL_INVESTORS, DAILY_DATA_PRE2020
    for p in (DAILY_DATA_ALL_INVESTORS, DAILY_DATA_PRE2020):
        assert str(p).endswith(".parquet"), (
            f"Panel path {p} is no longer .parquet; update this guard and the "
            f"readers together if the storage format changed again."
        )

    PANEL = {"DAILY_DATA_ALL_INVESTORS", "DAILY_DATA_PRE2020"}

    def tainted_value(node: ast.AST, taint: set[str]) -> bool:
        """True if ``node`` evaluates to a panel path: a tainted Name, an
        attribute ending in a panel constant, or ``str(<tainted>)``."""
        if isinstance(node, ast.Name):
            return node.id in taint
        if isinstance(node, ast.Attribute):
            return node.attr in PANEL
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "str"):
            return any(tainted_value(a, taint) for a in node.args)
        return False

    offenders = []
    for py in sorted((ROOT / "code").rglob("*.py")):
        tree = ast.parse(py.read_text())
        funcs = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        taint = set(PANEL)
        # Fixpoint: propagate the panel taint through assignments, parameter
        # defaults, and call arguments until no new tainted name appears.
        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and tainted_value(node.value, taint):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and tgt.id not in taint:
                            taint.add(tgt.id); changed = True
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = node.args
                    pos = a.posonlyargs + a.args
                    for arg, default in zip(pos[len(pos) - len(a.defaults):], a.defaults):
                        if tainted_value(default, taint) and arg.arg not in taint:
                            taint.add(arg.arg); changed = True
                    for arg, default in zip(a.kwonlyargs, a.kw_defaults):
                        if (default is not None and tainted_value(default, taint)
                                and arg.arg not in taint):
                            taint.add(arg.arg); changed = True
                elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id in funcs):
                    params = funcs[node.func.id].args
                    pos = params.posonlyargs + params.args
                    for i, arg in enumerate(node.args):
                        if (i < len(pos) and tainted_value(arg, taint)
                                and pos[i].arg not in taint):
                            taint.add(pos[i].arg); changed = True
                    for kw in node.keywords:
                        if (kw.arg and tainted_value(kw.value, taint)
                                and kw.arg not in taint):
                            taint.add(kw.arg); changed = True
        # Flag any read_csv whose first argument resolves to a panel path.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            is_read_csv = ((isinstance(fn, ast.Attribute) and fn.attr == "read_csv")
                           or (isinstance(fn, ast.Name) and fn.id == "read_csv"))
            if is_read_csv and node.args and tainted_value(node.args[0], taint):
                offenders.append(f"{py.relative_to(ROOT)}:{node.lineno}: read_csv on a panel path")
    assert not offenders, (
        "Daily panel is Parquet but pd.read_csv is used on a panel path "
        "(reproducibility invariant violated): " + "; ".join(offenders)
    )


def check_label_slugs_outside_the_document_resolve(_m: dict[str, str]) -> None:
    """Covers: every ``\\label`` slug a repo file names.

    The code cites the
    manuscript by its stable slug rather than by a rendered number, and that
    requirement is what hides the decay: the slug survives renumbering, so it
    reads as durable, while folding or retitling a section deletes it outright.
    Inside the document the typesetter reports a dangling reference; outside it
    nothing does, so a docstring or an assertion's failure message points at a
    section that no longer exists with the suite green.

    Both ends are derived. The defined set is every ``\\label`` in the
    submission's documents; the referenced set is every slug-shaped token in
    the tracked files that are not ``.tex`` -- scripts, the Makefile, the
    bibliography's comments -- since those are the ones nothing dereferences.

    A slug named after a document's file name is attributed to that document,
    and resolves only there: ``qf_paper.tex subsec:<name>`` for a label the
    article does not carry sends a reader to the wrong file. The attribution
    runs from the file name to the next semicolon, full stop, file name or
    paragraph break.

    A file that reads a tree outside this repository is quoting that tree's
    document, so its slugs resolve there and not here: the port scripts carry
    the slug of every section they drop, which is the record of the drop and
    not a pointer into this manuscript. That property is read off the file -- a
    module-level constant bound to an absolute path outside ROOT -- rather
    than from a list of filenames here, so a second port script is covered and
    a script that stops reading the other tree loses the exemption.
    """
    import subprocess

    defined = set()
    for tex in submission_tex(tables=False):
        defined |= set(re.findall(r"\\label\{([^}]+)\}", tex.read_text()))

    slug = re.compile(
        r"(?<![\w:])("
        r"(?:sec|subsec|subsubsec|tab|fig|eq|prop|cor|lem|alg|fn):[a-z][a-z0-9_]*)")
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
        check=True).stdout.split()

    external = re.compile(r"(?m)^[A-Z_]+ *= *Path\(\"(/[^\"]+)\"\)")

    labels_of = {tex.name: set(re.findall(r"\\label\{([^}]+)\}", tex.read_text()))
                 for tex in MANUSCRIPTS}
    attribution = re.compile(
        r"\b(qf_paper\.tex)\b(.*?)"
        r"(?=[;.](?:\s|$)|\bqf_paper|\n\s*\n|\"\"\"|\Z)", re.S)

    dangling = {}
    misattributed = {}
    for rel in tracked:
        if rel.endswith((".tex", ".bib")):
            continue
        try:
            text = (ROOT / rel).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if any(not Path(mo.group(1)).is_relative_to(ROOT)
               for mo in external.finditer(text)):
            continue
        for hit in slug.finditer(text):
            key = hit.group(1)
            if key not in defined:
                dangling.setdefault(key, set()).add(rel)
        for att in attribution.finditer(text):
            doc = att.group(1)
            for hit in slug.finditer(att.group(2)):
                key = hit.group(1)
                if key in defined and key not in labels_of[doc]:
                    misattributed.setdefault(f"{doc} {key}", set()).add(rel)

    assert_true(
        not dangling,
        claim="every \\label slug a repo file names resolves to a label the "
              "manuscript defines, so a docstring or a failure message cannot "
              "point at a section that was folded away",
        detail="; ".join(f"{k} ({', '.join(sorted(v))})"
                         for k, v in sorted(dangling.items())),
    )
    assert_true(
        not misattributed,
        claim="every slug a repo file names after a document's file name is "
              "labelled in that document, so the file name sends a reader to "
              "the label",
        detail="; ".join(f"{k} ({', '.join(sorted(v))})"
                         for k, v in sorted(misattributed.items())),
    )


def check_documents_carry_no_footnote(_m: dict[str, str]) -> None:
    """Covers: the documents take the manuscript's register, never the repo's.

    The manuscript states every qualification in the running text, so a reader
    reaches each one in the order the argument needs it rather than at the foot
    of the page. A footnote added later would restore the split silently: it
    typesets, it renumbers, and no binding around a macro it carries can see
    where on the page the macro sits.

    Derived from the artifacts. Every ``.tex`` the submission directory
    carries is read, the generated table bodies included, so the gate does
    not depend on a list of the files a footnote could be added to.

    ``\\footnote`` is the page-footnote command. A table's own note is set by
    the ``tabnote`` environment of the journal class and is part of the table,
    so it is outside what this gate refuses.
    """
    carriers = {}
    for tex in submission_tex():
        n = len(re.findall(r"\\footnote\b", tex.read_text()))
        if n:
            carriers[tex.relative_to(ROOT).as_posix()] = n

    assert_true(
        not carriers,
        claim="the submitted documents state every qualification in the running "
              "text and carry no footnote",
        detail="; ".join(f"{k}: {v} \\footnote" for k, v in sorted(carriers.items())),
    )


def check_criterion_timing_matches_the_history(_m: dict[str, str]) -> None:
    """Covers: a claim that a criterion was fixed before its run.

    "The commit history is the record" is itself a claim about an artifact, so
    it is derived here rather than trusted. A script header asserting that its
    criterion was fixed before the run holds only where the commit adding the
    criterion lands earlier than the commit adding the results it grades. Where
    the two arrive together there is no commit in which the criterion exists
    and its results do not, and the assertion is about a record the repository
    does not carry.

    Both ends are derived: the claiming scripts from a sweep of every header
    under code/, and each one's results files from the save calls in its own
    source. The adding commit is found through renames (``--follow``), so a
    script moved or renamed since its criterion was committed keeps its
    record rather than acquiring the rename as a fresh, empty first commit.
    Nothing else in the suite can see this -- a timing claim moves no
    number, so every binding around it stays green.

    This repository's history begins at the port, so a script that came
    through it arrives in the same commit as its results however the criterion
    was fixed, and the rule above would reject every one of them. The port
    commit is derived (the commit adding ``code/``), and a script added there
    is graded against ``criteria_record.tsv`` instead, which names the
    commits in the progenitor tree. The gate stays live in both directions: a
    claiming script added at the port and missing from the record fails, and a
    row naming a script that no longer claims its timing, or that this tree's
    own history can grade, fails too. A claiming script added after the port is
    graded by the rule above, unchanged -- the exemption covers the port, not
    the repository.
    """
    import ast
    import subprocess

    TIMING = ("pre-registered", "pre-register", "committed before",
              "fixed before", "before the run", "before the screen")

    def first_commit(path: str) -> str | None:
        out = subprocess.run(
            ["git", "log", "--follow", "--diff-filter=A", "--format=%H", "--", path],
            cwd=ROOT, capture_output=True, text=True).stdout.split()
        return out[-1] if out else None

    def files_in(commit: str) -> set[str]:
        out = subprocess.run(
            ["git", "show", "--name-only", "--format=", commit],
            cwd=ROOT, capture_output=True, text=True).stdout.split()
        return set(out)

    def port_commit() -> str | None:
        out = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H", "--", "code/"],
            cwd=ROOT, capture_output=True, text=True).stdout.split()
        return out[-1] if out else None

    record_path = ROOT / "criteria_record.tsv"
    recorded: set[str] = set()
    if record_path.exists():
        for line in record_path.read_text().splitlines():
            if line.startswith("#") or not line.strip():
                continue
            recorded.add(line.split("\t")[0])

    ported = port_commit()
    unsupported = []
    claiming_at_port: set[str] = set()
    for py in sorted((ROOT / "code").rglob("*.py")):
        rel = py.relative_to(ROOT).as_posix()
        if rel.startswith("code/output/"):
            continue          # the suite and the emitters grade nothing
        source = py.read_text()
        header = (ast.get_docstring(ast.parse(source)) or "")
        # A timing claim anywhere in the file, header or inline comment, is a
        # claim the reader can act on.
        claims = [t for t in TIMING
                  if t in header.lower() or t in source.lower()]
        if not claims:
            continue
        # The results this script grades, read off its own save calls.
        written = set(re.findall(r"[\"']([\w./-]+\.(?:json|csv))[\"']", source))
        added = first_commit(rel)
        if added is None:
            continue
        landed = files_in(added)
        together = sorted(f for f in landed
                          if f.startswith("results/")
                          and Path(f).name in {Path(w).name for w in written})
        if added == ported and together:
            # The port carried criterion and results together whatever their
            # order was upstream, so this tree's history cannot grade it. A
            # script shipping no results has nothing to grade and needs no row,
            # exactly as the rule below passes it.
            claiming_at_port.add(rel)
            if rel not in recorded:
                unsupported.append(
                    (rel, claims[0],
                     ["arrived at the port and names no row in "
                      "criteria_record.tsv"]))
            continue
        if together:
            unsupported.append((rel, claims[0], together))

    for rel in sorted(recorded - claiming_at_port):
        unsupported.append(
            (rel, "recorded",
             ["criteria_record.tsv carries a row this tree does not "
              "need: the file claims no timing, or its history can grade it"]))

    assert_true(
        not unsupported,
        claim="a script claiming its criterion was fixed before the run has a "
              "commit in which the criterion exists and its results do not",
        detail="; ".join(f"{r} says '{c}' and shipped {f} in the same commit"
                         for r, c, f in unsupported))


def check_no_comment_restates_a_constant_it_cannot_see(_m: dict[str, str]) -> None:
    """Covers: one source of truth for each number, on the code side.

    A comment naming a constant and its value is two records of one number. Where
    the file assigns that constant, the two sit together and a reader changing one
    sees the other. Where it does not, the value lives in another file, nothing
    dereferences the copy, and the comment turns false the day the constant moves
    — silently, because no import breaks and no assertion reads prose.

    Both ends are derived. The names a file may restate are the ones its own
    module body assigns, read from its syntax tree; the restatements are read
    from its comments and docstrings, never from its code, so a tuple assignment
    is not mistaken for a claim about a value. The rule carries no exemption: a
    comment that needs the number names the constant instead, and a comment that
    needs the constant's state says which state without printing it.
    """
    import ast
    import io
    import tokenize

    restated = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\s*=\s*(-?\d[\d.]*)\b")
    found = []
    for py in sorted((ROOT / "code").rglob("*.py")):
        src = py.read_text()
        tree = ast.parse(src, filename=str(py))

        assigned: set[str] = set()
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name):
                        assigned.add(sub.id)

        prose: list[tuple[int, str]] = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                prose.append((tok.start[0], tok.string))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node, clean=False)
            if doc:
                prose.append((getattr(node, "lineno", 1), doc))

        rel = py.relative_to(ROOT).as_posix()
        for line, text in prose:
            for name, value in restated.findall(text):
                if name not in assigned:
                    found.append(f"{rel}:{line} restates {name} = {value}")

    assert_true(
        not found,
        claim="no comment restates the value of a constant its own file does not "
              "assign",
        detail="; ".join(sorted(set(found))))


def check_every_bib_entry_is_cited(_m: dict[str, str]) -> None:
    """Covers: no dead code, applied to references.bib.

    The macro form of this rule is check_every_macro_is_referenced above. A
    bibliography entry has the same failure mode and no typesetter warning: a
    \\cite that resolves is checked, an entry nothing cites is not, so a key
    outlives the sentence it supported and nothing reports it. Thirteen did,
    among them petersen2009estimating under a paper that reports cluster-by-date
    standard errors and cites nobody for them, which is what an uncited entry
    looks like when the repair is a citation rather than a deletion.

    Derive both ends, and derive which bibliography each end reads: the files a
    document loads come from its own \\bibliography declaration, never from a
    fixed name here. A key is live where a document that loads its file cites
    it, and dangling means cited by a document whose own declaration reaches no
    such entry.
    """
    def keys_in(path: Path) -> set[str]:
        return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", path.read_text()))

    loaded: dict[Path, list[str]] = {}
    defined_anywhere: dict[str, set[str]] = {}
    cited_anywhere: set[str] = set()
    dangling: list[str] = []

    for tex in submission_tex(tables=False):
        body = tex.read_text()
        names = [n.strip() for decl in re.findall(r"\\bibliography\{([^}]*)\}", body)
                 for n in decl.split(",") if n.strip()]
        if not names:
            continue
        loaded[tex] = names
        reachable: set[str] = set()
        for n in names:
            bib = ROOT / f"{n}.bib"
            assert bib.exists(), f"{tex.name} loads {n}.bib, which is not in the tree"
            for k in keys_in(bib):
                reachable.add(k)
                defined_anywhere.setdefault(k, set()).add(n)
        cited: set[str] = set()
        for keys in re.findall(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}", body):
            cited.update(k.strip() for k in keys.split(",") if k.strip())
        cited_anywhere |= cited
        dangling += [f"{tex.name} cites {k}, absent from {'+'.join(names)}.bib"
                     for k in sorted(cited - reachable)]

    assert loaded, "no submitted document declares a bibliography"
    assert not dangling, (
        f"{len(dangling)} citation(s) reach no entry in the bibliography the "
        f"citing document loads: {dangling}")

    orphans = sorted(f"{k} (in {'+'.join(sorted(v))}.bib)"
                     for k, v in defined_anywhere.items() if k not in cited_anywhere)
    assert not orphans, (
        f"{len(orphans)} entr(y/ies) are cited by no .tex source: {orphans}\n"
        f"Cut the entry in the same commit as the sentence that stopped citing "
        f"it, or restore the citation where the method it names is the one used.")
