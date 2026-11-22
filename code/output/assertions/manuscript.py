"""Checks that read only the manuscript and the generated tables: front matter,
cross-references and the claims whose evidence is the document's own text."""

from __future__ import annotations

import re

from paths import TABLES_DIR
from shared_utils import CRIT_1PCT, CRIT_5PCT

from output.assertions.common import (
    MANUSCRIPTS,
    PAPER_TEX,
    ROOT,
    assert_true,
    parse_num,
    submission_tex,
)


# Quantitative Finance's instructions for authors ask for "an unstructured
# abstract of 200 words" in the "Checklist: What to Include" section.
ABSTRACT_WORD_LIMIT = 200


def check_abstract_length(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex abstract.

    The journal caps the abstract at ABSTRACT_WORD_LIMIT words and nothing else
    in this suite reads the abstract, so an edit that pushes it over arrives at
    the editor unflagged. Macros are expanded to their values before counting,
    because the count the journal applies is the one the reader sees, and an
    unexpanded ``\\sampleYears`` would score as a word either way while a
    two-token value would not. Math delimiters and braces are deleted rather
    than blanked, so ``($t = \\revCommonT$)`` counts as the three words it
    prints as; the tie in a value like ``0.93~million`` does print as a space.
    """
    tex = (ROOT / "qf_paper.tex").read_text()
    body = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S).group(1)
    body = re.sub(r"\\([A-Za-z]+)(?:\{\})?", lambda mo: m.get(mo.group(1), " "), body)
    words = len(re.sub(r"[$}{]", "", body).replace("~", " ").split())
    assert_true(
        words <= ABSTRACT_WORD_LIMIT,
        claim=f"the abstract fits the journal's {ABSTRACT_WORD_LIMIT}-word limit",
        detail=f"abstract runs to {words} words with macros expanded",
    )


def check_word_count_is_current(_m: dict[str, str]) -> None:
    """Covers: qf_paper.tex front matter, the word count the journal asks for.

    Quantitative Finance's instructions for authors ask every paper to state a
    word count. The count is a property of the prose, so any edit moves it, and
    nothing else in the suite reads it: every binding stays green while the
    first page states a count for an earlier draft. The committed file must be
    what code/output/word_count.py writes for the manuscript as it stands, and the
    manuscript must print it.
    """
    from output.word_count import WORD_COUNT_FILE, word_count_tex
    committed = (TABLES_DIR / WORD_COUNT_FILE).read_text()
    regenerated = word_count_tex()
    assert_true(
        regenerated == committed,
        claim="the word count under the JEL codes is the manuscript's current count",
        detail=f"tables/{WORD_COUNT_FILE} holds {committed.splitlines()[-1]!r}, "
               f"the manuscript now counts {regenerated.splitlines()[-1]!r}; "
               f"run make pdf and commit the file",
    )
    prose = re.sub(r"(?<!\\)%.*", "", PAPER_TEX.read_text())
    assert_true(
        "\\manuscriptWordCount" in prose,
        claim="the manuscript states its word count, as the journal asks",
        detail="qf_paper.tex no longer prints \\manuscriptWordCount",
    )


def check_taiwan_assignment_is_not_called_the_korean_one(_m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:taiwan and every other sentence of the article
    that describes the second market's regime assignment.

    The mandates fix the assignment in both markets, but the return-free
    corroboration does not carry over whole. In Korea the directionality ratio
    against a class's own sign-flip null places the two classes on opposite
    sides of one (subsec:regime_assignment); in Taiwan it keeps their ordering
    and leaves SITC below its null. A sentence saying the Taiwanese assignment
    was "fixed the same way" therefore reads as a replication where the
    statistic did not replicate, and no \\ref resolves it, so nothing but this
    check sees it go wrong.

    Derive both ends: the disclosure from sec:taiwan's own text, so that
    removing it fails here rather than making the check vacuous, and the
    offending phrasing from every Taiwan sentence the article carries.
    """
    text = PAPER_TEX.read_text()
    start = text.index("\\label{sec:taiwan}")
    nxt = re.search(r"\\section\{", text[start:])
    section = text[start:start + nxt.start()] if nxt else text[start:]
    assert_true(
        "does not separate from its own null" in section.replace("\n", " "),
        claim="sec:taiwan still discloses that SITC is not above its own "
              "sign-flip null, the part of the Korean corroboration that does "
              "not carry over",
        detail="the disclosure is gone from sec:taiwan, so the article's wording "
               "is no longer checked against anything",
    )

    sameness = re.compile(
        r"\b(fixed|assigned|made|read|established)\s+the\s+same\s+way"
        r"|\bthe\s+same\s+(assignment|statistic|signature)\b"
        r"|\bassignment\s+(is|was)\s+identical\b",
        re.IGNORECASE,
    )
    offenders = []
    for tex in MANUSCRIPTS:
        for i, line in enumerate(tex.read_text().splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                if re.search(r"Taiwan", sentence, re.IGNORECASE) and sameness.search(sentence):
                    offenders.append(f"{tex.name}:{i}: {sentence.strip()[:160]}")
    assert_true(
        not offenders,
        claim="no sentence of the article says the Taiwanese regime assignment "
              "was fixed the same way as the Korean one",
        detail="\n          ".join(offenders),
    )


# The standing of the advantage for foreign flow is one fact, and the vocabulary the manuscript
# settles it in. Subjects name the object; verdicts state where it stands.
# Both lists are the paper's own words, so a rewrite that invents a new phrasing
# escapes the tally -- which is why the failure names the paragraph rather than
# the phrase, and why the ceiling is the committed part.
_FOREIGN_SUBJECT = (
    r"advantage for foreign flow|foreign advantage|foreign result"
    r"|foreign \$S\^\{\\mathrm\{TV\}\}\$|foreign matched filter"
    r"|foreign flow's matched filter"
)
_STANDING_VERDICT = (
    r"conventional level|distinguishable from zero|loses its sign"
    r"|keeps (?:its|the) (?:predicted )?sign|does not stand equal|stand equally"
    r"|is absent|are absent|falls below|stands? (?:as )?a bound|survives on"
    r"|not on common support|weakest|fragile|bounded to"
    r"|loses significance|significant only|is not significant|significant at the 5\\% level on neither"
    r"|significant neither|significant in neither"
    r"|significant at the 5\\% level in none of"
    r"|significant on the analysis window only"
    # Bare negation beside the subject is the third verdict form, and the one a
    # keyword list keyed to standing vocabulary alone walks past.
    r"|advantage for foreign flow (?:does not|is not|no longer)"
    r"|and not the advantage for foreign flow|though not the advantage for foreign flow"
)

# Paragraphs that may state where the advantage for foreign flow stands. Every one of them
# either makes the estimate it reports (subsec:skip_day), settles it
# (subsec:size_restricted), or is read alone
# and so must carry the scope on its own (abstract, Introduction, conclusions). The budget is the state after each
# remaining site was checked against that test, and what it stops is the next
# pass adding one more, locally correct, to a fact already settled. Moving the
# robustness section's channel-by-channel estimates to the supplement and
# merging the limits by dimension took the count from 13 to 8, and restating the
# Introduction around the one claim took it to 6. The ceiling follows it down.
# That 6 held only because the tally read "separate from zero" and the paper
# wrote "separated" and "separates". With the zero verdict worded
# "distinguishable from zero" throughout, the tally also reads the KOSPI 200
# paragraph of subsubsec:kospi200_falsification, which makes the in-index
# estimate it reports, so the ceiling is 7. The tally then stripped every line
# from its first escaped percent sign on, reading "1\% level" as a comment, and
# so missed the verdicts after it; with that fixed and the abstract's wording
# ("significant only among smaller stocks") in the vocabulary, the abstract,
# one of the sites above, counts too, so the ceiling is 8. The abstract and the
# conclusions now word the verdict "significant neither" and "significant in
# neither", which the vocabulary reads, and the count stays at 8. The paper
# now names the object "the advantage for foreign flow", and the two paragraphs
# of subsec:limitations word their verdicts "significant on the analysis window
# only" and "foreign flow's matched filter does not", which the vocabulary
# reads too, so the count stays at 8. The abstract and the conclusions word the
# verdict "though not the advantage for foreign flow", which the vocabulary now
# reads, and subsec:size_restricted settles the standing that
# subsec:contrast_halves only counts, so the count is 7 and the ceiling follows.
# The count of the alternative estimations and Taiwan's foreign verdict then
# moved into subsec:size_restricted, the one report of the foreign ordering
# that place-defensive-work's question 1 asks for, so the count is 5.
FOREIGN_HALF_ADJUDICATION_BUDGET = 5

# The appendices' own ceiling, at the count they carry once the supplement
# folded into them. The sentences there do different work from the paper's:
# each appendix states the advantage for foreign flow under the perturbation its own run
# produced, because every configuration that the results data hold reaches the
# reader as a number. This
# budget is therefore a ratchet against the next addition rather than a target
# to shrink to, and an appendix that genuinely owes such a sentence raises it
# in the commit that adds the run. The same rewording of the zero verdict
# brings in the paragraph of subsec:appendix_participation_matching, which
# states the advantage for foreign flow its own run produced, so the ceiling is 3. The same
# escaped-percent fix, and a second subject phrase in the list, bring in the
# paragraphs of subsec:appendix_j_oos, subsec:appendix_zero_flow and
# subsec:appendix_size_stratified, each of which states the advantage for
# foreign flow that its own run produced, so the ceiling is 6. The count then
# stood at 4, and the stacked second-day estimates moved from subsec:skip_day
# into subsec:appendix_skip_day, so the ceiling is 5.
FOREIGN_HALF_APPENDIX_BUDGET = 5

def _adjudicating_paragraphs(tex: str) -> list[tuple[str, str]]:
    """Body paragraphs that state a verdict on where the advantage for foreign flow stands.

    Floats are stripped: a table note and a caption are read alone, so they
    carry their own qualification and are exempt the way the disclosure-repetition
    audit exempts captions.
    """
    tex = re.sub(r"\\begin\{(table|figure)\}.*?\\end\{\1\}", "", tex, flags=re.S)
    section, out = None, []
    for para in re.split(r"\n\s*\n", tex):
        headers = re.findall(r"\\(?:sub)*section\*?\{([^}]*)\}", para)
        if headers:
            section = headers[-1]
        if r"\begin{abstract}" in para:
            section = "Abstract"
        body = re.sub(r"(?<!\\)%.*", "", para)
        if re.search(_FOREIGN_SUBJECT, body, re.I) and re.search(
            _STANDING_VERDICT, body, re.I
        ):
            out.append((section, " ".join(body.split())))
    return out


def check_foreign_half_is_settled_once(_m: dict[str, str]) -> None:
    """Covers: qf_paper.tex subsec:size_restricted;
    qf_paper.tex subsec:appendix_size_restricted.

    The advantage for foreign flow is one fact, and a manuscript that
    re-decides it section by section reads as a defence rather than as a
    disclosure. Nothing else in the suite sees this: the sentences carry no
    number, so no binding covers them; no quantifier over an artifact, so the
    quantifier audit skips them; and none of the disclosure labels the prose
    skill greps, so its tally returns zero on every one. A single paragraph
    reads as ordinary and a page of them reads as anxiety, so only a count makes
    the defect visible -- the same reason the referent audit tallies rather than
    inspects. The set is derived from each document; the budget is committed.

    The body is counted under one budget, and the appendices under another.
    The body stood at its ceiling while nothing read the appendices, where the
    same verdict is stated once per perturbation, so an appendix paragraph
    re-deciding it landed unseen.
    """
    body, appendices = PAPER_TEX.read_text().split("\\appendices", 1)
    for part, text, budget in (("body", body, FOREIGN_HALF_ADJUDICATION_BUDGET),
                               ("appendices", appendices,
                                FOREIGN_HALF_APPENDIX_BUDGET)):
        hits = _adjudicating_paragraphs(text)
        by_section: dict[str, int] = {}
        for section, _ in hits:
            by_section[section] = by_section.get(section, 0) + 1
        tally = ", ".join(f"{s} x{n}" for s, n in by_section.items())
        assert_true(
            len(hits) <= budget,
            claim="the standing of the advantage for foreign flow is settled in "
                  "subsec:size_restricted and cited elsewhere, so no further "
                  "section re-decides it",
            detail=f"{part}: {len(hits)} adjudicating paragraphs against a "
                   f"budget of {budget}, in {len(by_section)} sections: {tally}",
        )


def check_float_day_counts_agree_with_notes(m: dict[str, str]) -> None:
    """Covers: every macro declares the population it measures.

    Two day counts live one apart. \\sampTotalDays is the contemporaneous
    panel and \\sampTotalDaysMinusOne the forward-return one, and a float
    whose body counts one while its note prints the other contradicts itself
    on the page while both macros stay correctly bound. Derive the pairing
    from the ``\\input`` each table block carries.
    """
    days = {"sampTotalDays": parse_num(m["sampTotalDays"]),
            "sampTotalDaysMinusOne": parse_num(m["sampTotalDaysMinusOne"])}
    # Any macro carrying a bare integer can be the note's declaration; a float
    # may count its own days under a name of its own (\sampFactorDays), and a
    # declared population is declared however it is spelled.
    integral = {}
    for name, value in m.items():
        flat = value.replace("{,}", "").replace(",", "").replace("~", "").strip()
        if flat.isdigit():
            integral[name] = int(flat)

    src = "\n".join(tex.read_text() for tex in MANUSCRIPTS)
    for block in re.findall(r"\\begin\{table\}.*?\\end\{table\}", src, re.S):
        inputs = re.findall(r"\\input\{tables/(\w+)\}", block)
        note = "".join(re.findall(r"\\tabnote\{(.*?)\}\s*\n", block, re.S))
        if not inputs:
            continue
        body = "".join((TABLES_DIR / f"{f}.tex").read_text() for f in inputs)
        printed = {int(x.replace(",", "").replace("{,}", ""))
                   for x in re.findall(r"\b\d[\d,]*\b|\d+\{,\}\d+", body)}

        # A float enters the subject set on what its BODY prints, not on what
        # its note happens to name: the yearly-stability table, since cut, declared its day count
        # in the body alone and so was skipped while it printed the panel count
        # on a table of Fama-MacBeth t-statistics.
        in_body = {int(v) for v in days.values()} & printed
        if in_body:
            declared = {integral[k] for k in integral
                        if re.search(r"\\" + k + r"(?![A-Za-z])", note)}
            assert_true(
                bool(in_body & declared),
                claim=f"the float built from {inputs} declares in its note the "
                      f"day count its own body prints",
                detail=f"body prints {sorted(in_body)}; the note names macros "
                       f"valued {sorted(declared)}")

        # One macro name is a prefix of the other, so match to a word boundary.
        named = [k for k in days
                 if re.search(r"\\" + k + r"(?![A-Za-z])", note)]
        if not named:
            continue
        for k in named:
            other = next(o for o in days if o != k)
            assert_true(
                not (days[other] in printed and days[k] not in printed),
                claim=f"the note on the float built from {inputs} names "
                      f"\\{k}, the day count its own body prints",
                detail=f"body prints {int(days[other])} and the note names "
                       f"\\{k} = {int(days[k])}",
            )


def check_appendix_span_is_derived(_m: dict[str, str]) -> None:
    """Covers: a quantifier over the paper's own appendices.

    A sentence saying how many appendices the paper carries, or spanning
    them as a literal letter range, is a second claim riding on no macro. It is
    decidable from the same file, nothing dereferences it, and the suite stays
    green while it goes stale: an appendix added later leaves the hand-typed
    count or span describing a document that no longer exists, which is what
    the check forbids.

    The repair is the form the paper once used for the same set,
    ``Appendices~\\ref{first}--\\ref{last}``, which LaTeX resolves. This check
    forbids the hand-typed form outright rather than comparing a count against
    the sections: comparing passes a sentence that is right today and re-breaks
    on the next appendix, while forbidding leaves only the form that cannot.
    """
    counts = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
              "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
              "twenty|[0-9]+")
    hand_typed = []
    for tex in MANUSCRIPTS:
        name = tex.name
        text = re.sub(r"(?<!\\)%.*", "", tex.read_text())
        for sentence in re.split(r"(?<=\.)\s+", text):
            if not re.search(r"\bappendices\b", sentence, re.I):
                continue
            for pat in (rf"\b(?:{counts})\s+(?:[a-z]+\s+)?appendices\b",
                        r"\([A-Z]--[A-Z]\)"):
                hit = re.search(pat, sentence, re.I)
                if hit:
                    hand_typed.append(f"{name}: {hit.group(0)!r}")
    n = len(re.findall(
        r"^\\section\{",
        PAPER_TEX.read_text().split("\\appendices", 1)[1], re.M))
    assert not hand_typed, (
        f"A sentence spans the "
        f"appendices by hand while the document enumerates {n} of them "
        f"itself. Span them with \\ref{{}}--\\ref{{}}: "
        + "; ".join(hand_typed))


def check_every_appendix_is_cited(_m: dict[str, str]) -> None:
    """Covers: qf_paper.tex, each appendix "cited in the text at the point it
    supports".

    The quantifier ranges over the appendices, so the set is the \\section
    commands after \\appendices. An appendix counts as cited when the body
    refers to any label inside it. A paragraph reaching every appendix at once
    announces them rather than citing one, and does not count: a cut of the
    only body sentence citing an appendix leaves such a paragraph still saying
    that every appendix is cited.

    That paragraph is found by the property that disqualifies it -- it reaches
    every appendix at once -- and not by the sentence a journal asks it to open
    with, which differs between the two versions of this manuscript and would
    make the check a wording test.
    """
    body, appendices = re.sub(r"(?<!\\)%.*", "", PAPER_TEX.read_text()).split("\\appendices", 1)

    sections = re.split(r"(?m)^(?=\\section\{)", appendices)[1:]
    labels_of = [re.findall(r"\\label\{([^}]+)\}", s) for s in sections]
    assert sections, "the paper defines no appendix \\section to range over"

    def appendices_touched(text: str) -> int:
        refs = set(re.findall(r"\\ref\{([^}]+)\}", text))
        return sum(1 for labs in labels_of if refs.intersection(labs))

    for para in re.split(r"\n\s*\n", body):
        if appendices_touched(para) == len(sections):
            body = body.replace(para, "")

    cited = set(re.findall(r"\\ref\{([^}]+)\}", body))
    uncited = [labs[0] for labs in labels_of if not cited.intersection(labs)]
    assert_true(not uncited,
                claim="each appendix is cited in the article at the point it "
                      "supports",
                detail=f"no qf_paper.tex sentence outside the announcing "
                       f"paragraph cites {', '.join(uncited)} or any label "
                       f"inside it")


def check_no_paper_sync_todo_in_submission(_m: dict[str, str]) -> None:
    """Covers: a change to code or prose reaches the other in the same commit.

    A ``TODO(paper-sync)`` marker records that one corner of the triangle has
    not caught up with another. Inside the code a reader trips over it; inside
    the submitted document nothing does, and the marker becomes the only record
    that a printed claim is unmet -- which is the state it is written to make
    temporary. Derive the marker set from the submission directory rather than
    naming the file that carries one today, so a marker left in any appendix
    is caught on the same terms.
    """
    stale = []
    for tex in submission_tex(tables=False):
        for lineno, line in enumerate(tex.read_text().splitlines(), 1):
            if "TODO(paper-sync)" in line:
                stale.append(f"{tex.name}:{lineno}")
    assert_true(
        not stale,
        claim="the submitted document carries no unresolved paper-sync marker",
        detail=f"markers: {stale}")


def check_prose_extended_robustness_count(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex sec:appendix_robustness_battery, tab:robustness_summary;
    qf_paper.tex subsec:contrast_halves, whose battery sentences read "Appendix D
    tests each class's matched filter in nine robustness checks: ... Seven of
    the nine also test S^TV, the matched filter of foreign flow ... Each class's matched filter
    stays significant at the 5% level in every check that it enters, except at
    a market-capitalization floor that keeps only the largest
    \\threshTopTickers stocks".

    That sentence and "Table D1 reports, for each of nine robustness checks, a
    statistic of the capacity-bound matched filter ... The foreign matched
    filter's statistic is significant at the 1% level in all seven checks that
    the table reports for foreign flow" are quantifiers over a set the
    paper enumerates itself, so each is asserted over that set and the set is
    derived from the table. The counts are hand-typed on both sides, the
    words in the prose and the rows and dashes in the tabular, and this table is
    written in the manuscript rather than emitted by a generator, so nothing
    else couples them: a check added or dropped, or a foreign cell filled or
    emptied, leaves a sentence false with every macro binding green. Pinned:
    the row count against the first word, the foreign cells that are not a dash
    against the second, four cells per row, and the significance of the matched
    filter's statistic in every cell, read from the macro that MATCHED_STATISTIC
    names for the row and that the cell must print.
    """
    words = dict(enumerate(
        ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen")))
    body = PAPER_TEX.read_text()

    # The tabular tab:robustness_summary labels, read back from that label.
    end = body.index(r"\label{tab:robustness_summary}")
    start = body.rindex(r"\begin{tabular}", 0, end)
    block = body[start:end].splitlines()
    after_head = block[next(i for i, ln in enumerate(block)
                            if r"\colrule" in ln) + 1:]
    rows = after_head[:next(i for i, ln in enumerate(after_head)
                            if r"\botrule" in ln) + 1]
    rows = [ln for ln in rows if "&" in ln]

    said = re.search(r"for each of (\w+) robustness checks", body)
    assert said, ("sec:appendix_robustness_battery no longer says 'for each of N "
                  "robustness checks'; re-anchor this check or delete it with the "
                  "sentence")
    word = said.group(1)
    assert_true(words.get(len(rows)) == word,
                claim="sec:appendix_robustness_battery says tab:robustness_summary "
                      f"reports 'for each of {word} robustness checks' a statistic, one "
                      "per row of the table",
                detail=f"the table carries {len(rows)} rows")

    cells = [ln.split(r"\\")[0].split("&") for ln in rows]
    wide = ["&".join(c) for c in cells if len(c) != 4]
    assert_true(not wide,
                claim="tab:robustness_summary's note says 'Each row is one robustness "
                      "check' and each class column 'reports the summary statistic for "
                      "that class', so every row is a test, a statistic for each of "
                      "the two classes and a reference",
                detail=f"{len(wide)} row(s) are not four cells: "
                       f"{[w[:60] for w in wide]}")

    # The foreign column counts the checks that the table reports for foreign
    # flow, a dash marking a check of S^MC alone.
    applies = re.search(r"in all (\w+) checks that the table reports for foreign flow", body)
    total = re.search(r"for each of (\w+) robustness checks", body)
    assert applies, ("sec:appendix_robustness_battery no longer says how many of "
                     "the checks it 'reports for foreign flow'; re-anchor this check "
                     "or delete it with the sentence")
    run = sum(1 for c in cells if len(c) == 4 and c[2].strip() != "---")
    body_n = re.search(r"\$t\$-statistic, in (\w+) robustness checks:", body)
    body_m = re.search(r"(\w+) of the (\w+) also test \$S\^", body)
    assert body_n and body_m, ("subsec:contrast_halves no longer says 'in N robustness checks:' "
                               "and 'M of the N also test S^TV'; "
                               "re-anchor this check or delete it with the sentences")
    body_said = (body_n.group(1), body_m.group(1).lower())
    assert_true(words.get(len(rows)) == body_said[0]
                and words.get(run) == body_said[1]
                and body_m.group(2) == body_said[0],
                claim="subsec:contrast_halves says the battery holds "
                      f"'{body_said[0]}' checks, '{body_said[1]}' of which "
                      "also test the matched filter of foreign flow",
                detail=f"{len(rows)} rows, {run} foreign cells that are not a dash")
    assert_true(words.get(run) == applies.group(1).lower()
                and words.get(len(rows)) == total.group(1),
                claim="sec:appendix_robustness_battery says the table reports "
                      f"'{applies.group(1)}' of its {total.group(1)} checks for "
                      "foreign flow, one per foreign cell of "
                      "tab:robustness_summary that is not a dash",
                detail=f"{run} of {len(rows)} foreign cells are not a dash")

    # The significance of each cell's matched-filter statistic. Every row must
    # have an entry, so a check added to the table fails here until its
    # statistic is named, and the named macro must be printed in its cell.
    unnamed = [c[0].strip() for c in cells
               if not any(c[0].strip().startswith(k) for k in MATCHED_STATISTIC)]
    assert_true(not unnamed,
                claim="every row of tab:robustness_summary has its matched-filter "
                      "statistic named in MATCHED_STATISTIC",
                detail=f"unnamed rows: {unnamed}")
    for c in cells:
        label = next(k for k in MATCHED_STATISTIC if c[0].strip().startswith(k))
        cap, foreign = MATCHED_STATISTIC[label]
        for macro, cell in ((cap, c[1]), (foreign, c[2])):
            if macro is not None:
                assert_true("\\" + macro in cell,
                            claim=f"the '{label}' row of tab:robustness_summary "
                                  f"prints \\{macro}",
                            detail=f"cell: {cell.strip()[:80]}")
        assert_true(abs(parse_num(m[cap])) > CRIT_5PCT,
                    claim="subsec:contrast_halves says the capacity-bound matched "
                          "filter is significant at the 5% level in every check except "
                          "the highest market-capitalization floor",
                    detail=f"'{label}': \\{cap} = {m[cap]}")
        if foreign is not None:
            assert_true(abs(parse_num(m[foreign])) > CRIT_1PCT,
                        claim="sec:appendix_robustness_battery says the foreign "
                              "statistic is significant at the 1% level in all seven "
                              "checks that the table reports for foreign flow",
                        detail=f"'{label}': \\{foreign} = {m[foreign]}")
    assert_true(abs(parse_num(m[LARGE_CAP_ONLY])) <= CRIT_5PCT,
                claim="sec:appendix_robustness_battery names the highest floor of the "
                      "threshold check as the one check in which the capacity-bound "
                      "statistic is not significant at the 5% level",
                detail=f"\\{LARGE_CAP_ONLY} = {m[LARGE_CAP_ONLY]}")


# The macro that each row of tab:robustness_summary prints for each class's
# matched filter, keyed by the row label's opening words: the capacity-bound
# statistic, then the foreign one (None where the cell is a dash). The
# threshold row prints a second capacity-bound statistic, for the highest floor, which the prose names as the exception.
MATCHED_STATISTIC = {
    "Pre-period hold-out": ("oosPPMChr", "ffPPTVhrt"),
    "Yearly stability": ("ryMinHRMCt", "fryMinHRTVt"),
    "Long-short": ("lsMCt", "fLsTVt"),
    "Winsorization": ("winMCt", "ffWinTVhrt"),
    "Within-day rank transform": ("rankCapHRMCt", "rankForHRTVt"),
    "Autocorrelation-robust": ("nwTenMCt", "fNwTenTVt"),
    "Factor-neutral": ("fnAlphaT", "fFnAlphaT"),
    "Market cap threshold": ("threshHeldMCt", None),
    "Formation price": ("vwapCapMatchedVwap", None),
}
LARGE_CAP_ONLY = "threshTopMCt"


def check_robustness_summary_references_resolve(m: dict[str, str]) -> None:
    """Covers: qf_paper.tex tab:robustness_summary.

    Every row of the summary table ends in a Reference cell pointing at the
    subsection or table holding that dimension's full result. The typesetter
    reports a dangling reference, so a deleted target is caught; an emptied one
    is not. A target reduced to a heading and a sentence still resolves, and
    the summary row then cites a layer the document no longer prints, with
    every macro binding green, because a pointer to a layer is a claim that
    carries no number.

    Derived on both sides: the referring set is read out of the table's own
    Reference cells, the defined set out of the document each cell points
    into, and what counts as printing a result out of the parameter file, so a
    row added, a target renamed or a number moved between prose and a table is
    covered without editing a list here.
    """
    body = PAPER_TEX.read_text()
    source = re.sub(r"(?<!\\)%.*", "", body)

    end = source.index(r"\label{tab:robustness_summary}")
    start = source.rindex(r"\begin{tabular}", 0, end)
    block = source[start:end].splitlines()
    after_head = block[next(i for i, ln in enumerate(block)
                            if r"\colrule" in ln) + 1:]
    rows = after_head[:next(i for i, ln in enumerate(after_head)
                            if r"\botrule" in ln) + 1]

    cited: list[tuple[str, str]] = []
    for ln in rows:
        if "&" not in ln:
            continue
        cells = ln.split(r"\\")[0].split("&")
        if len(cells) != 4:
            continue          # the four-cell shape is the check above's
        for target in re.findall(r"\\ref\{([^}]*)\}", cells[-1]):
            cited.append((cells[0].strip(), target))

    assert cited, ("tab:robustness_summary carries no Reference cell with a "
                   "\\ref; re-anchor this check or delete it with the column")

    labels = {}
    for mo in re.finditer(r"\\label\{([^}]*)\}", source):
        labels[mo.group(1)] = (source, mo.start())

    missing = sorted({t for _, t in cited if t not in labels})
    assert_true(not missing,
                claim="every Reference cell of tab:robustness_summary points at "
                      "a label the paper defines",
                detail=f"unresolved: {missing}")

    def block_of(source: str, at: int) -> str:
        """The span a label names: its float, or the run of prose to the next
        heading. The float environment is matched starred as well as plain,
        because a two-column class sets every full-width float starred, and a
        name-only match reads those as prose and grades the paragraph after
        the label instead of the table above it."""
        opens = [mo for mo in re.finditer(r"\\begin\{(table|figure)(\*?)\}", source)
                 if mo.start() < at]
        if opens:
            mo = opens[-1]
            closed = source.find("\\end{%s%s}" % mo.groups(), mo.end())
            if closed > at:
                return source[mo.start():closed]
        nxt = re.search(r"\\(?:sub)*section\{", source[at:])
        return source[at:at + nxt.start()] if nxt else source[at:]

    # A block prints its dimension's result when it carries a number: a macro
    # the parameter file defines, a generated table body, or a tabular.
    blank = []
    for row, target in cited:
        text = block_of(*labels[target])
        prints = (r"\input{tables/" in text
                  or r"\begin{tabular}" in text
                  or any("\\" + name in text for name in m))
        if not prints:
            blank.append((row, target))
    assert_true(not blank,
                claim="every Reference cell of tab:robustness_summary points at a "
                      "block that still prints a number",
                detail=f"resolves but prints nothing: {blank}")
