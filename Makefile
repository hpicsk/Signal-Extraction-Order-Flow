# Replication pipeline for the matched-filter normalization of order flow.
#
# Usage:
#   make help        # show targets
#   make all         # data + analysis + tables + pdf
#   make data        # build the daily panels under results/
#   make analysis    # run all empirical analyses
#   make tables      # regenerate tables/*.tex from results/
#   make pdf         # word count, then latexmk at the repository root
#   make clean       # remove generated outputs (keeps source)
#
# External data is read from one tree inside the repository, data/ (override with
# QF_DATA_ROOT). See README.md for the expected layout.

PYTHON ?= python
CODE_DIR := code
RESULTS_DIR := results
TABLES_DIR := tables
RUN := cd $(CODE_DIR) && PYTHONPATH=. $(PYTHON)

.PHONY: help all data data-universe data-main data-pre2020 data-twn data-hml \
        analysis analysis-return-prediction analysis-signal-structure \
        analysis-robustness analysis-taiwan \
        tables lint-prose pdf clean clean-results clean-tables clean-latex install

help:
	@echo "Replication targets:"
	@echo "  make install      pip install -e . (one-time)"
	@echo "  make data         build all daily panels in $(RESULTS_DIR)/"
	@echo "  make analysis     run all empirical analyses"
	@echo "  make analysis-<dir>  one directory: return-prediction, signal-structure,"
	@echo "                    robustness, taiwan"
	@echo "  make tables       regenerate $(TABLES_DIR)/*.tex"
	@echo "  make pdf          compile qf_paper.pdf"
	@echo "  make all          full pipeline end-to-end"
	@echo "  make clean        remove all generated outputs"

install:
	pip install -e .

# --------------------------------------------------------------------------
# Stage 1 — build the daily panels (read external data, write to results/).
# --------------------------------------------------------------------------

# data-twn is not optional: code/empirical/signal_structure/directionality_frequency_null.py and sec:taiwan read
# the Taiwan panel, so a build without it dies at the first Taiwan reader.
data: data-universe data-main data-pre2020 data-twn data-hml

# The point-in-time security-type universe both Korean panels are built over,
# classified from the marcap archive and the FnGuide 수정주가 export. Runs
# first: the two panels read it.
data-universe:
	$(RUN) data/kr_universe.py

data-main: data-universe
	$(RUN) data/full_daily_data_pipeline.py

data-pre2020: data-universe
	$(RUN) data/pre_period_pipeline.py

data-twn:
	$(RUN) data/twn_full_daily_data_pipeline.py

data-hml: data-main
	$(RUN) data/construct_hml_factor.py

# --------------------------------------------------------------------------
# Stage 2 — analyses (read the panels, write results/*.json and *.csv).
#
# One target per directory under code/empirical/, listing
# that directory's scripts in the order the committed results were produced.
# A line that must follow another says so. Each script's header names the
# manuscript section its numbers reach, and every file under results/ is
# named by the script that writes it, so grep over code/ finds the writer.
# --------------------------------------------------------------------------

analysis: analysis-return-prediction analysis-signal-structure \
          analysis-robustness analysis-taiwan

# code/empirical/return_prediction/ — forward returns regressed on each
# normalisation of the flow, by investor class, and that regression re-run
# under other windows, prices, samples and regimes.
analysis-return-prediction:
	# Fama-MacBeth next-day return prediction for the capacity-bound class
	# (subsec:return_prediction).
	$(RUN) empirical/return_prediction/return_prediction_analysis.py
	# Foreign-flow predictability inside and outside the KOSPI 200
	# (subsubsec:kospi200_falsification). Reads the KOSPI 200 membership panel.
	$(RUN) empirical/return_prediction/foreign_kospi200_split.py
	# Return prediction inside within-day market-cap quintiles
	# (sec:appendix_liquidity_stratified).
	$(RUN) empirical/return_prediction/liquidity_stratified_return_prediction.py
	# The contrast as one class x normalization interaction, stacked
	# over both class panels (subsec:reversal_test, tab:reversal_interaction).
	$(RUN) empirical/return_prediction/reversal_interaction.py
	# The same return prediction for every disclosed class and the institutional
	# aggregate (subsec:return_prediction).
	$(RUN) empirical/return_prediction/multi_investor_robustness.py
	# Return prediction on the second post-formation day, both markets (subsec:skip_day).
	$(RUN) empirical/return_prediction/skip_day_return_prediction.py
	# VWAP formation-price robustness (stale-closing-print test), Korea (subsec:vwap_formation, sec:appendix_robustness_battery).
	$(RUN) empirical/return_prediction/vwap_formation_robustness.py

# code/empirical/signal_structure/ — what the flow itself carries, measured
# without returns: its scaling with size and volume, its persistence, and the
# accounting bound that the independence premise runs against.
analysis-signal-structure:
	# The normalisation exponent estimated over the whole scale-invariant
	# family rather than at its two endpoints (subsec:free_exponent).
	$(RUN) empirical/signal_structure/free_exponent_profile.py
	# The accounting bound |S^TV| <= 1 that subsec:threshold_estimate argues from,
	# checked on each class's own panel.
	$(RUN) empirical/signal_structure/accounting_bound.py
	# Matched-filter advantage for every KRX-disclosed class
	# (subsec:subtype_dose_response).
	$(RUN) empirical/signal_structure/subtype_advantage.py
	# Churn-vs-net-accumulation decomposition: directionality ratios + carrier
	# elasticities, Korea main + pre-period
	# (subsec:regime_assignment; persist* macros).
	$(RUN) empirical/signal_structure/flow_persistence_decomposition.py
	# Sign-flip null for the same ratios, which prices how much of their
	# ordering the two classes' differing trading frequencies deliver
	# (subsec:appendix_directionality_frequency; dir* macros).
	$(RUN) empirical/signal_structure/directionality_frequency_null.py
	# Correlation and VIF of the two filters on the capacity-bound panel
	# (subsec:return_prediction). --regime domestic
	# is that panel, not the KRX 기관계 aggregate.
	$(RUN) empirical/signal_structure/multicollinearity.py --regime domestic

# code/empirical/robustness/ — the reversal re-estimated under other
# denominators, samples and transforms, and the audits of the universe and
# of the foreign definition.
analysis-robustness:
	# The headline re-run across market-cap thresholds (subsec:appendix_j_marketcap).
	$(RUN) empirical/robustness/threshold_sensitivity.py
	# Same reversal with day-t traded value replaced by a lagged ADV, so the
	# volume denominator no longer contains D (subsec:appendix_adv_normalisation).
	$(RUN) empirical/robustness/lagged_adv_normalisation.py
	# Class boundary of the Korean foreign series: the 기타외국인 leg the
	# definition drops (needs the data root for the DataGuide export).
	# Also measures the nonzero-flow restriction's premise in
	# subsec:appendix_panel_filters.
	$(RUN) empirical/robustness/foreign_definition_share.py
	# Same reversal on cells where the two classes' flow intensities are matched,
	# so the shape of each class's flow cannot carry it
	# (subsec:appendix_participation_matching).
	$(RUN) empirical/robustness/participation_matching.py
	# Each class's own flow persistence and the pressure it would exert inside
	# the return window, which is the channel the assignment statistic shares
	# with the confound (subsec:appendix_flow_continuation).
	$(RUN) empirical/robustness/flow_continuation.py
	# Every ordering claim re-estimated on within-day normal scores, so no
	# difference between the two filters can come from one having the wider
	# tail (subsec:appendix_j_rank).
	$(RUN) empirical/robustness/rank_transform_robustness.py
	# The three perturbations above re-estimated inside the size distribution,
	# so the size composition of common support is separated from the
	# perturbations themselves (subsec:size_restricted).
	$(RUN) empirical/robustness/size_restricted_common.py
	# Holdout, yearly-subperiod, long-short, winsorised, Newey-West and
	# factor-spanning arms, once per class (sec:appendix_robustness_battery).
	# --investor institutional is the Capacity-Bound subset (pension + insurance),
	# not the KRX 기관계 aggregate; see the module docstring of the called script.
	$(RUN) empirical/robustness/robustness_tests_extended.py --investor institutional
	$(RUN) empirical/robustness/robustness_tests_extended.py --investor foreign
	# The criteria committed before the panel was extended to 2020-2025, graded
	# on the results above. Reads the yearly arm written just before it.
	$(RUN) empirical/robustness/window_2025_criteria.py

# code/empirical/taiwan/ -- the panel and the return prediction for the two
# classes the assignment reaches (sec:taiwan).
analysis-taiwan:
	$(RUN) empirical/taiwan/twn_analysis.py
	# The contrast of eq:reversal_contrast on the same panel; its reading rule
	# was committed before the run.
	$(RUN) empirical/taiwan/twn_contrast.py

# --------------------------------------------------------------------------
# Stage 3 — regenerate LaTeX numbers/macros.
# --------------------------------------------------------------------------

tables:
	$(RUN) output/generate_table_bodies.py
	$(RUN) output/word_count.py
	$(RUN) output/test_assertions.py

lint-prose:
	$(RUN) output/_prose_lint.py

# --------------------------------------------------------------------------
# Stage 4 — compile the manuscript.
# --------------------------------------------------------------------------

# code/output/word_count.py recounts the manuscript first, because the count it prints is a
# property of the prose and moves with every edit to it. code/output/page_cap.py then fails
# the build when the article runs past its page cap.

pdf:
	$(RUN) output/word_count.py
	latexmk -pdf -interaction=nonstopmode qf_paper.tex
	$(RUN) output/page_cap.py

# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------

all: data analysis tables pdf

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

clean: clean-results clean-tables clean-latex

clean-results:
	rm -rf $(RESULTS_DIR)/_cache
	rm -f  $(RESULTS_DIR)/*.json $(RESULTS_DIR)/*.csv $(RESULTS_DIR)/*.parquet

clean-tables:
	rm -f $(TABLES_DIR)/numbers.tex
	rm -f $(TABLES_DIR)/*_body.tex
	rm -f $(TABLES_DIR)/word_count.tex
	rm -f $(TABLES_DIR)/day_count.tex $(TABLES_DIR)/obs_count.tex $(TABLES_DIR)/ticker_count.tex

clean-latex:
	latexmk -C qf_paper.tex || true
