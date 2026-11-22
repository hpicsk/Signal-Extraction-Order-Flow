# Quantitative Finance version — matched-filter normalization of order flow

Self-contained: the pipeline that builds the numbers, the generator that writes
the tables and macros, the assertion suite that checks every claim against
`results/`, and the documents.

## Build

    make install    # pip install -e .   (once)
    make data       # build the daily panels under results/  (needs the data root)
    make analysis   # run the empirical scripts
    make tables     # regenerate tables/*.tex and the word count, then run the assertion suite
    make pdf        # word count, latexmk, page cap
    make all        # all of the above
    make help       # every target

`make pdf` alone compiles what is committed. The journal's class and
bibliography style, `rQUF2e.cls` and `rQUF.bst` (template v2.1), sit beside the
documents. `make pdf` fails if `qf_paper.pdf` runs past its page cap
(`code/output/page_cap.py`).

## Data requirements

`make data` reads three external trees under `data/`, none of them committed.
Place or symlink each, or set `QF_DATA_ROOT` to a root that holds all three:

| path | what supplies it |
| --- | --- |
| `data/marcap/data/` | clone <https://github.com/FinanceData/marcap> |
| `data/fnguide/raw/` | FnGuide DataGuide export (subscription required) |
| `data/finmind/` | the FinMind Taiwan datasets `ohlcv/`, `price_adj/`, `instflow/`, `shares/` and `div_result/`, one parquet per stock, from the FinMind API (historical endpoints openly accessible) |
| `data/registry/` | committed; restore from git |

`code/paths.py` resolves each on first access and names what supplies it when
one is missing. Everything downstream of the panels is committed, so a clone
without the trees still runs `make tables` and `make pdf`.

## What is here

| path | what it is |
| --- | --- |
| `qf_paper.tex` | the article: five sections, the references, then the appendices, from the proof to Taiwan |
| `code/` | the pipeline: `data/`, `empirical/`, `output/` |
| `results/` | what the analysis scripts write and the generator reads |
| `tables/` | `tables/numbers.tex`, `tables/word_count.tex` and the table bodies, written by `code/output/generate_table_bodies.py` and `code/output/word_count.py` |
| `references.bib` | the bibliography |
| `rQUF2e.cls`, `rQUF.bst` | the Quantitative Finance class and bibliography style |
| `criteria_record.tsv` | the criteria fixed before the runs they grade, with the commits of the tree that this one was ported from, which record the order |

## Regenerating

Each generator writes at the path the document reads, so a re-run overwrites the
committed artifact in place and a stale result shows up as a `git diff`. Review
before committing.
