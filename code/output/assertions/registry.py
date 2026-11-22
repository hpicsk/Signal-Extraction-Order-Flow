"""The checks code/output/test_assertions.py runs, in order, each under the label it prints.

MACRO_CHECKS pin every load-bearing ``\\newcommand`` in
``tables/numbers.tex`` to the results file it was generated
from, then run the gates over the repository; the coverage gate reads this
list and so comes last. PROSE_CHECKS pin what the manuscript says without a
macro -- directional orderings, significance bounds, quantifiers over the
paper's own artifacts, and the document's own structure -- directly to
``results/`` and to the documents.

A check lives in the module named after the code directory whose results it
reads (``return_prediction``, ``signal_structure``, ``robustness``,
``data``), in ``manuscript`` when it reads only the documents, in ``theory``
when it decides a step of the paper's argument by computation, and in
``repository`` when it reads the tree itself. ``return_prediction``,
``signal_structure`` and ``robustness`` are packages, each divided further by
what its directory's scripts measure.
"""

from output.assertions.return_prediction.reversal import (
    check_prose_aggregate_alignment,
    check_prose_capacity_return_prediction,
    check_prose_foreign_return_prediction,
    check_prose_contrast_halves_rows,
    check_prose_pre_period_reversal,
    check_prose_reversal_interaction,
    check_prose_zero_flow,
    check_return_prediction,
    check_reversal_word_needs_both_halves,
)
from output.assertions.return_prediction.horizons import (
    check_prose_skip_day,
    check_prose_vwap_formation,
)
from output.assertions.return_prediction.subsamples import (
    check_kospi200_falsification,
    check_prose_kospi200_horse_race,
    check_prose_liquidity_stratified,
)
from output.assertions.signal_structure.scaling import (
    check_estimation_samples,
    check_prose_sample_size_lower_bound,
    check_prose_multicollinearity,
    check_sample_sizes,
)
from output.assertions.signal_structure.directionality import (
    check_directionality_frequency_null,
    check_flow_persistence,
    check_prose_flow_persistence,
)
from output.assertions.signal_structure.matched_filter import (
    check_prose_free_exponent,
    check_prose_accounting_bound,
)
from output.assertions.signal_structure.subtypes import (
    check_prose_capacity_concentration,
    check_prose_subinvestor_sharpest,
    check_prose_subtype_advantage,
)
from output.assertions.robustness.filter_channels import (
    check_prose_flow_continuation,
    check_prose_lagged_adv_normalisation,
)
from output.assertions.robustness.common_support import (
    check_prose_participation_matching,
    check_prose_size_restricted,
)
from output.assertions.robustness.battery import (
    check_extended_robustness_foreign,
    check_extended_robustness_yearly,
    check_window_2025_verdicts,
    check_long_short,
    check_prose_extended_robustness_foreign,
    check_prose_extended_robustness_inst,
    check_prose_mismatched_sign_tails,
    check_prose_rank_transform,
    check_prose_threshold_sensitivity,
)
from output.assertions.robustness.definitions import (
    check_investment_trust_inclusion,
    check_prose_foreign_definition,
    check_prose_universe_composition,
)
from output.assertions.theory import check_argument_steps
from output.assertions.taiwan import (
    check_prose_taiwan,
    check_taiwan,
    check_twn_contrast,
)
from output.assertions.data import (
    check_hml_floors_do_not_bind,
    check_matched_filter_algebra,
    check_min_stocks_floor_matches_its_appendix,
    check_prose_no_trade_sessions,
    check_prose_panel_survivorship,
    check_threshold_grid_summary_floors_do_not_bind,
    check_turnover_descriptive_is_the_panel,
    check_union_stacking_is_the_panels_with_a_return,
)
from output.assertions.manuscript import (
    check_abstract_length,
    check_word_count_is_current,
    check_taiwan_assignment_is_not_called_the_korean_one,
    check_appendix_span_is_derived,
    check_every_appendix_is_cited,
    check_float_day_counts_agree_with_notes,
    check_foreign_half_is_settled_once,
    check_no_paper_sync_todo_in_submission,
    check_prose_extended_robustness_count,
    check_robustness_summary_references_resolve,
)
from output.assertions.repository import (
    check_document_paths_resolve,
    check_documents_carry_no_footnote,
    check_every_bib_entry_is_cited,
    check_every_macro_is_referenced,
    check_generator_idempotent,
    check_label_slugs_outside_the_document_resolve,
    check_macro_coverage_gate,
    check_manuscript_file_names_resolve,
    check_no_comment_restates_a_constant_it_cannot_see,
    check_no_orphan_results,
    check_panel_readers_use_parquet,
    check_criterion_timing_matches_the_history,
    check_publishing_scripts_account_for_dof,
    check_readme_names_every_external_root,
)


MACRO_CHECKS = [
    ("Return prediction macros", check_return_prediction),
    ("Flow persistence decomposition (round-trip vs net accumulation)", check_flow_persistence),
    ("Directionality ratio vs trading-frequency null (subsec:appendix_directionality_frequency)",
     check_directionality_frequency_null),
    ("Taiwan panel and return prediction (sec:taiwan)", check_taiwan),
    ("Sample sizes", check_sample_sizes),
    ("Panel total vs per-class estimation samples (subsec:appendix_panel_filters)",
     check_estimation_samples),
    ("Extended robustness — yearly (capacity-bound)", check_extended_robustness_yearly),
    ("Extended robustness — yearly (foreign)", check_extended_robustness_foreign),
    ("Criteria for the 2020-2025 window (window_2025_criteria.py)", check_window_2025_verdicts),
    ("Criteria for the class with investment trusts (investment_trust_inclusion.py)",
     check_investment_trust_inclusion),
    ("KOSPI 200 falsification (foreign)", check_kospi200_falsification),
    ("Long-short + factor neutrality", check_long_short),
    ("Matched-filter algebra", check_matched_filter_algebra),
    # Repo-integrity guard: panel readers use Parquet.
    ("Publishing scripts account for their degrees of freedom and baselines",
     check_publishing_scripts_account_for_dof),
    ("Turnover descriptives are the panel's", check_turnover_descriptive_is_the_panel),
    ("Cross-section floor unbound on the full panels, and the stratified cells keep the days it leaves them (subsec:appendix_panel_filters)",
     check_min_stocks_floor_matches_its_appendix),
    ("HML tail and split floors bind on no formation year of the window",
     check_hml_floors_do_not_bind),
    ("The union stacking is the two classes' stock-days with a next-day return (subsec:reversal_test)",
     check_union_stacking_is_the_panels_with_a_return),
    ("Size-sweep summary floors leave every grid point summarised",
     check_threshold_grid_summary_floors_do_not_bind),
    ("No orphan artifacts under results/", check_no_orphan_results),
    ("Claims that a criterion preceded its run match the history",
     check_criterion_timing_matches_the_history),
    ("No unresolved paper-sync marker in the submitted document",
     check_no_paper_sync_todo_in_submission),
    ("Repo documents name paths that resolve", check_document_paths_resolve),
    ("No comment restates a constant its own file does not assign",
     check_no_comment_restates_a_constant_it_cannot_see),
    ("Manuscripts name files that resolve", check_manuscript_file_names_resolve),
    ("README names every external tree paths.py reads",
     check_readme_names_every_external_root),
    ("Panel readers use Parquet", check_panel_readers_use_parquet),
    # Bulk verification of every auto-generated macro.
    ("Generator idempotency (all auto-generated macros)", check_generator_idempotent),
    ("Every macro in numbers.tex is referenced by a .tex source",
     check_every_macro_is_referenced),
    ("Every entry in references.bib is cited by a .tex source",
     check_every_bib_entry_is_cited),
    # Coverage gate must run last so it sees the full MACRO_CHECKS list.
    ("Macro coverage gate, manual region", check_macro_coverage_gate),
]


PROSE_CHECKS = [
    ("Steps of the argument that a computation decides (sec:theory, subsec:proof_matched_filter)",
     check_argument_steps),
    ("FM sample-size lower bound", check_prose_sample_size_lower_bound),
    ("Return prediction next-day directionals", check_prose_capacity_return_prediction),
    ("Foreign horse-race S^MC sign flip", check_prose_foreign_return_prediction),
    ("KRX aggregate alignment", check_prose_aggregate_alignment),
    ("Flow persistence (round-trip vs accumulation) directionals", check_prose_flow_persistence),
    ("KOSPI 200 horse-race + univariate directionals", check_prose_kospi200_horse_race),
    ("Reversal as a class × normalisation interaction", check_prose_reversal_interaction),
    ("Contrast on the pre-period window (subsec:contrast_halves)",
     check_prose_pre_period_reversal),
    ("Each row of tab:contrast_halves as subsec:contrast_halves reads it",
     check_prose_contrast_halves_rows),
    ("'Reversal' only where both classes' advantages are significant (eq:reversal_contrast)",
     check_reversal_word_needs_both_halves),
    ("Contrast under a lagged traded-value denominator (sec:appendix_nonuniform_channels)",
     check_prose_lagged_adv_normalisation),
    ("Contrast on matched net participation rates (subsec:appendix_participation_matching)",
     check_prose_participation_matching),
    ("Flow persistence and the pressure it would exert (subsec:appendix_flow_continuation)",
     check_prose_flow_continuation),
    ("Free normalisation exponent theta (subsec:free_exponent)",
     check_prose_free_exponent),
    ("Common-support perturbations inside the size distribution (subsec:size_restricted)",
     check_prose_size_restricted),
    ("Extended robustness dimensions match the summary table "
     "(sec:appendix_robustness_battery)",
     check_prose_extended_robustness_count),
    ("Every Reference cell of the robustness summary still prints a number "
     "(tab:robustness_summary)",
     check_robustness_summary_references_resolve),
    ("Ordering claims under a within-day rank transform (subsec:appendix_j_rank)",
     check_prose_rank_transform),
    ("Matched-filter advantage across the disclosed classes (subsec:subtype_dose_response)",
     check_prose_subtype_advantage),
    ("The accounting bound |S^TV| <= 1 (subsec:threshold_estimate)",
     check_prose_accounting_bound),
    ("Nonzero-flow sample restriction (subsec:appendix_zero_flow)", check_prose_zero_flow),
    ("Concentration inside the capacity-bound class (subsec:appendix_aggregate_alignment)",
     check_prose_capacity_concentration),
    ("Size-stratified return prediction (sec:appendix_liquidity_stratified)",
     check_prose_liquidity_stratified),
    ("Panel survivorship premise (subsec:data)",
     check_prose_panel_survivorship),
    ("Sessions without a trade are dropped, and returns span them (subsec:data, sec:taiwan)",
     check_prose_no_trade_sessions),
    ("Universe composition — the security types the panel excludes (sec:appendix_data)",
     check_prose_universe_composition),
    ("Foreign definition — the excluded 기타외국인 leg (sec:appendix_data)",
     check_prose_foreign_definition),
    ("Collinearity of the two filters", check_prose_multicollinearity),
    ("Return prediction on the second day after the flow (subsec:skip_day)", check_prose_skip_day),
    ("VWAP formation-price robustness (subsec:vwap_formation)", check_prose_vwap_formation),
    ("Extended robustness (capacity-bound) — pre-period + winsor + NW", check_prose_extended_robustness_inst),
    ("Extended robustness (foreign) — long-short + alpha", check_prose_extended_robustness_foreign),
    ("Mismatched sign under both tail treatments (sec:appendix_robustness_battery)", check_prose_mismatched_sign_tails),
    ("Threshold sensitivity (subsec:appendix_j_marketcap) — headline anchor, grid coverage, both sides of the boundary, every grid point of all three statistics", check_prose_threshold_sensitivity),
    ("Abstract fits the journal's word limit", check_abstract_length),
    ("Word count printed is the manuscript's current count", check_word_count_is_current),
    ("Appendix span is derived, not hand-typed", check_appendix_span_is_derived),
    ("Every appendix is cited from the main paper", check_every_appendix_is_cited),
    ("Label slugs named outside the document resolve", check_label_slugs_outside_the_document_resolve),
    ("The documents carry no footnote", check_documents_carry_no_footnote),
    ("No sentence calls the capacity-bound subset the sharpest sub-investor", check_prose_subinvestor_sharpest),
    ("A float's note names the day count its own body prints", check_float_day_counts_agree_with_notes),
    ("The advantage for foreign flow is settled once and cited elsewhere", check_foreign_half_is_settled_once),
    ("Taiwan directional claims and panel construction (sec:taiwan)", check_prose_taiwan),
    ("The contrast on the Taiwan panel (tab:twn_contrast)", check_twn_contrast),
    ("Taiwan's assignment is not described as the Korean one",
     check_taiwan_assignment_is_not_called_the_korean_one),
]
