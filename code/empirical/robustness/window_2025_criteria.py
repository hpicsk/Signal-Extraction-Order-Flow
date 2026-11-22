#!/usr/bin/env python3
"""Criteria for extending the Korean panel from 2020--2024 to 2020--2025.

The panel ended in 2024 when the headline results were read, and the extension
adds a sixth calendar year. Extending a sample after its result is visible is a
degree of freedom on the sample axis. Each claim the extension can overturn
therefore carries a sign and a threshold fixed here, committed before the
commit that moves END_DATE_MAIN and before any 2025 observation entered the
panel. Every verdict reaches the reader as the comparison it decided, whichever
way it falls.

The criteria grade the 2020--2025 panel, the one sample the manuscript reports.
Six restate a claim of the manuscript on the longer window, and one adds the
new year to a quantifier that ranges over years.

K1  capacity-bound horse race (subsec:return_prediction): t_MC^HR above the 1%
    critical value and t_TV^HR negative.
K2  foreign horse race (subsec:return_prediction): t_TV^HR above the 1%
    critical value and t_MC^HR negative.
K3  stacked reversal on common support, univariate (subsec:reversal_test):
    b4 - b3 positive with t above the 1% critical value.
K4  return-free ordering (subsec:regime_assignment): capacity-bound excess
    directionality above one and foreign below one, each at the randomisation
    floor of its sign-flip null.
K5  foreign half on common support, univariate (subsec:contrast_halves):
    (b1 + b3) - (b2 + b4) negative with t below minus the 5% critical value.
    The abstract qualifies this half by size and by window, and on 2020--2024
    it clears in levels on common support and nowhere else.
K6  stacked contrast on the second day after the flow, [t+1, t+2], common
    support, univariate (subsec:skip_day): reversal t below the 5% critical
    value and foreign half positive. On 2020--2024 the reversal is absent
    there and the foreign half positive, so this criterion fixes that sign
    before 2025 is read.
K7  yearly stability (subsec:appendix_j_oos): capacity-bound t_MC^HR above the
    5% critical value in every calendar year of the window, 2025 included.

BASELINE: the 2020--2025 panel graded as one sample | obvious alternative: a
          referee says the new year should be graded alone, as a forward
          holdout | discharge: argument and K7. The manuscript reports one
          window, so its claims are graded where they are made, and the one
          claim that ranges over years reads the new year alone.

Degrees of freedom
------------------
CRIT_1PCT, CRIT_5PCT  CHOSEN -- the two-sided 1% and 5% normal critical values
                      the manuscript's significance statements use, imported
                      from shared_utils.regression.
WINDOW_YEARS          CHOSEN -- 2020 through 2025. The window closes at a
                      calendar-year boundary because the yearly-stability
                      table reports whole years, and 2026 is partial.
floor                 MEASURED -- 2 / (n_draws + 1), the smallest two-sided
                      p-value the sign-flip null's draw count can return, read
                      from results/directionality_frequency_null.json.

Inputs
------
    results/multi_investor_robustness_results.json
    results/reversal_interaction_results.json
    results/directionality_frequency_null.json
    results/extended_robustness_results.json

Output
------
    results/window_2025_verdicts.json
"""

import json

from paths import RESULTS_DIR
from shared_utils import print_header, save_json
from shared_utils.regression import CRIT_1PCT, CRIT_5PCT

WINDOW_YEARS = [str(y) for y in range(2020, 2026)]
"""CHOSEN -- the calendar years the extended panel spans."""

OUTPUT = RESULTS_DIR / "window_2025_verdicts.json"


def _load(name: str) -> dict:
    return json.loads((RESULTS_DIR / name).read_text())


def _check_window(ri: dict, dfn: dict, yearly: dict) -> None:
    """Refuse to grade a result that was not estimated on the extended window.

    Three of the four inputs stamp the window they cover, and a verdict read
    off the 2020--2024 files would grade the old sample under the new name.
    """
    last = ri["common"]["date_last"]
    if not last.startswith(WINDOW_YEARS[-1]):
        raise ValueError(f"reversal_interaction_results.json ends {last}, "
                         f"not in {WINDOW_YEARS[-1]}")
    window = dfn["korea"]["window"]
    if window != f"{WINDOW_YEARS[0]}-{WINDOW_YEARS[-1]}":
        raise ValueError(f"directionality_frequency_null.json covers {window}")
    if sorted(yearly) != WINDOW_YEARS:
        raise ValueError(f"extended_robustness_results.json reports years "
                         f"{sorted(yearly)}, not {WINDOW_YEARS}")


def evaluate() -> list[dict]:
    """Every criterion's statistic, its comparison, and whether it held."""
    mi = _load("multi_investor_robustness_results.json")["return_predictions"]
    ri = _load("reversal_interaction_results.json")
    dfn = _load("directionality_frequency_null.json")
    yearly = _load("extended_robustness_results.json")["yearly_subperiods"]
    _check_window(ri, dfn, yearly)

    cap, forg = mi["capacity_bound"], mi["foreign"]
    first = ri["common"]["fama_macbeth_univariate"]
    second = ri["common_second_day"]["fama_macbeth_univariate"]
    k200 = dfn["korea"]
    floor = 2.0 / (dfn["config"]["n_draws"] + 1)
    year_t = {y: yearly[y]["horse_race"]["MC"]["t_stat"] for y in WINDOW_YEARS}

    return [
        {"id": "K1", "claim": "capacity-bound t_MC^HR > CRIT_1PCT and t_TV^HR < 0",
         "values": {"t_MC_HR": cap["horse_race_MC"]["t_stat"],
                    "t_TV_HR": cap["horse_race_TV"]["t_stat"]},
         "passed": cap["horse_race_MC"]["t_stat"] > CRIT_1PCT
                   and cap["horse_race_TV"]["t_stat"] < 0},
        {"id": "K2", "claim": "foreign t_TV^HR > CRIT_1PCT and t_MC^HR < 0",
         "values": {"t_TV_HR": forg["horse_race_TV"]["t_stat"],
                    "t_MC_HR": forg["horse_race_MC"]["t_stat"]},
         "passed": forg["horse_race_TV"]["t_stat"] > CRIT_1PCT
                   and forg["horse_race_MC"]["t_stat"] < 0},
        {"id": "K3", "claim": "stacked reversal b4 - b3 > 0 with t > CRIT_1PCT",
         "values": {"reversal": first["reversal"]["mean"],
                    "t": first["reversal"]["t"]},
         "passed": first["reversal"]["mean"] > 0
                   and first["reversal"]["t"] > CRIT_1PCT},
        {"id": "K4", "claim": "capacity-bound excess > 1 > foreign excess, "
                              "each at the randomisation floor",
         "values": {"capacity_excess": k200["capacity_bound"]["excess"],
                    "foreign_excess": k200["foreign"]["excess"],
                    "capacity_p": k200["capacity_bound"]["p_two_sided"],
                    "foreign_p": k200["foreign"]["p_two_sided"],
                    "floor": floor},
         "passed": k200["capacity_bound"]["excess"] > 1.0 > k200["foreign"]["excess"]
                   and k200["capacity_bound"]["p_two_sided"] <= floor + 1e-12
                   and k200["foreign"]["p_two_sided"] <= floor + 1e-12},
        {"id": "K5", "claim": "foreign half < 0 with t < -CRIT_5PCT",
         "values": {"foreign_half": first["advantage_foreign"]["mean"],
                    "t": first["advantage_foreign"]["t"]},
         "passed": first["advantage_foreign"]["mean"] < 0
                   and first["advantage_foreign"]["t"] < -CRIT_5PCT},
        {"id": "K6", "claim": "day-two reversal t < CRIT_5PCT and foreign half > 0",
         "values": {"reversal_t": second["reversal"]["t"],
                    "foreign_half": second["advantage_foreign"]["mean"],
                    "foreign_half_t": second["advantage_foreign"]["t"]},
         "passed": second["reversal"]["t"] < CRIT_5PCT
                   and second["advantage_foreign"]["mean"] > 0},
        {"id": "K7", "claim": "capacity-bound t_MC^HR > CRIT_5PCT in every year",
         "values": year_t,
         "passed": all(t > CRIT_5PCT for t in year_t.values())},
    ]


def main() -> None:
    print_header("Criteria for the 2020-2025 window")
    verdicts = evaluate()
    for v in verdicts:
        print(f"  {v['id']}  {'HOLDS' if v['passed'] else 'FAILS':5s}  {v['claim']}")
        print(f"        {v['values']}")
    save_json({"window": f"{WINDOW_YEARS[0]}-{WINDOW_YEARS[-1]}",
               "crit_1pct": CRIT_1PCT, "crit_5pct": CRIT_5PCT,
               "verdicts": verdicts}, OUTPUT)


if __name__ == "__main__":
    main()
