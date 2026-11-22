#!/usr/bin/env python3
"""Page cap on the article, read off the build that ``make pdf`` just ran.

The author capped the article at 35 pages on 2026-09-22, the length Quantitative
Finance names for a typical paper ("no more than 35 pages, inclusive of:
References, Footnotes"). The cap applies to qf_paper.pdf with its references and
bound appendices.

The count comes from the pdfTeX log line ``Output written on qf_paper.pdf (N
pages, ...)``, the typesetter's record of the file it wrote. A log that records
no written PDF fails the build rather than passing it.
"""
import re
import sys

from paths import REPO_ROOT

ARTICLE_PAGE_CAP = 35  # the author's cap on qf_paper.pdf (docstring)
LOG = REPO_ROOT / "qf_paper.log"


def article_pages() -> int:
    """Pages of qf_paper.pdf, from the last write its log records."""
    hits = re.findall(r"Output written on qf_paper\.pdf \((\d+) pages?",
                      LOG.read_text(errors="replace"))
    if not hits:
        sys.exit(f"{LOG.name} records no written qf_paper.pdf; run latexmk first")
    return int(hits[-1])


if __name__ == "__main__":
    pages = article_pages()
    if pages > ARTICLE_PAGE_CAP:
        sys.exit(f"qf_paper.pdf runs {pages} pages against the cap of "
                 f"{ARTICLE_PAGE_CAP}")
    print(f"qf_paper.pdf: {pages} pages, within the cap of {ARTICLE_PAGE_CAP}")
