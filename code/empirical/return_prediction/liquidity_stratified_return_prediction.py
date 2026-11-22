#!/usr/bin/env python3
"""
Return prediction stratified by market capitalisation.

The absence of the advantage for foreign flow among the largest stocks, which
qf_paper.tex subsec:size_restricted reports and subsec:limitations states,
bounds the range of the cross-section over which
the matched-filter advantage holds. This script runs the return regression
that limit rests on: within-day
market-cap quintiles, each class's own matched and mismatched filter,
next-day returns. The same partition then locates the headline sample on
that axis, because the common-support restriction of subsec:reversal_test
admits only stock-days a capacity-bound institution trades and is therefore
a size restriction as well as a participation one. Both feed
sec:appendix_liquidity_stratified and the body sentences that cite it.

Degrees of freedom
------------------
N_QUINTILES     CHOSEN — 5, the conventional quintile sort. Not tuned here.
RETURN_HORIZON  CHOSEN — R_t1, the horizon subsec:return_prediction
                reports.
BREAKPOINTS     CHOSEN — formed within day on the full daily cross-section, not
                within each class's nonzero-flow subset, so a quintile means the
                same universe for both classes and the two are comparable bin
                for bin.
cov_type        CHOSEN — the Fama-MacBeth default. At h = 1 the forward returns
                do not overlap, so no overlap correction applies.
MIN_STOCKS_PER_DAY  MEASURED elsewhere — inherited from shared_utils.config,
                the value every FM specification in the paper uses.

BASELINE: within-day quintile bins of market capitalization | obvious
alternative: a continuous signal x size interaction, which does not depend on
bin edges | discharge: argument. The appendix reads two cells, the smallest
quintile where capacity-bound flow is too scarce to measure and the largest
where the foreign advantage is absent, and a slope locates neither.

Output: ../results/liquidity_stratified_results.json
"""

from shared_utils import (
    MIN_STOCKS_PER_DAY_DEFAULT,
    load_all_investors_panel,
    print_header,
    save_json,
)
from shared_utils.data import within_day_quantile
from shared_utils.regression import fit_fama_macbeth
from paths import RESULTS_DIR

RESULTS_PATH = RESULTS_DIR / "liquidity_stratified_results.json"

N_QUINTILES = 5
QUINTILE_LABELS = [f"Q{i}" for i in range(1, N_QUINTILES + 1)]
RETURN_HORIZON = "R_t1"

# Each class's matched filter is fixed in advance from its documented scaling
# role (qf_paper.tex subsec:regime_assignment), not from the returns tested here.
MATCHED_FILTER = {"capacity_bound": "MC", "foreign": "TV"}


def _build_stratifiers(df):
    """Attach the within-day market-cap quintile label to ``df``.

    Breakpoints are formed on the full daily cross-section (see BREAKPOINTS
    above).
    """
    df = df.copy()
    df["size_q"] = within_day_quantile(df, "M", QUINTILE_LABELS)
    print(f"  size_q: {df['size_q'].notna().sum():,} of {len(df):,} stock-days "
          f"labelled ({df['size_q'].isna().sum():,} without an M value)")
    return df


def _fm(df, cols, d_col):
    """Day-standardised FM of ``RETURN_HORIZON`` on ``cols``, nonzero flow only."""
    res, panel = fit_fama_macbeth(
        df, cols, RETURN_HORIZON,
        min_stocks_per_day=MIN_STOCKS_PER_DAY_DEFAULT,
        row_filter=lambda sub: sub[sub[d_col] != 0],
    )
    if res is None:
        return None
    out = {c: {"coef_bp": float(res.params[c]) * 1e4,
               "t_stat": float(res.tstats[c]),
               "p_value": float(res.pvalues[c])} for c in cols}
    out["n_days"] = int(panel["date"].nunique())
    out["n_obs"] = int(len(panel))
    return out


def _by_quintile(df, investor, q_col):
    """The univariate MC and TV fits within each bin, the two that
    tab:liquidity_stratified prints."""
    smc, stv, d_col = (f"S_MC_{investor}", f"S_TV_{investor}", f"D_{investor}")
    out = {}
    for q in QUINTILE_LABELS:
        sub = df[df[q_col] == q]
        cell = {}
        for label, cols in (("univariate_MC", [smc]),
                            ("univariate_TV", [stv])):
            fit = _fm(sub, cols, d_col)
            if fit is not None:
                cell[label] = fit
        if cell:
            out[q] = cell
    return out


def _size_composition(df):
    """Where in the size distribution the common-support stacking sits.

    subsec:reversal_test estimates the reversal on the stock-days both classes
    trade, and a capacity-bound institution reaches only a minority of sessions
    (subsec:appendix_zero_flow), so that restriction is a size restriction too. This
    block measures how far it moves the sample, as the share of stock-days each
    sample places in each within-day market-cap quintile. The reference is the
    foreign class's own nonzero-flow days, which are the foreign leg of the
    union stacking the same section reports beside it.
    """
    common = (df["D_capacity_bound"] != 0) & (df["D_foreign"] != 0)
    out = {}
    for name, mask in (("common_support", common),
                       ("foreign_nonzero", df["D_foreign"] != 0)):
        q = df.loc[mask, "size_q"]
        out[name] = {
            "n_stock_days": int(len(q)),
            "share": {label: float((q == label).mean()) for label in QUINTILE_LABELS},
        }
    return out


def main():
    print_header("SIZE-STRATIFIED RETURN PREDICTION", level="banner")
    df = _build_stratifiers(load_all_investors_panel())
    print(f"Panel: {len(df):,} rows, {df['date'].nunique():,} days, "
          f"{df['ticker'].nunique():,} stocks")

    results = {"by_size": {}}
    for investor in MATCHED_FILTER:
        results["by_size"][investor] = _by_quintile(df, investor, "size_q")
    results["size_composition"] = _size_composition(df)

    results["config"] = {
        "n_quintiles": N_QUINTILES,
        "return_horizon": RETURN_HORIZON,
        "min_stocks_per_day": MIN_STOCKS_PER_DAY_DEFAULT,
        "matched_filter": MATCHED_FILTER,
        "panel_rows": int(len(df)),
        "panel_dates": int(df["date"].nunique()),
    }
    save_json(results, RESULTS_PATH)

    for axis_key, axis_name in (("by_size", "market-cap quintile"),):
        for investor, matched in MATCHED_FILTER.items():
            mismatched = "TV" if matched == "MC" else "MC"
            print(f"\n{investor} by {axis_name} "
                  f"(matched = S^{matched}, mismatched = S^{mismatched}):")
            print(f"{'Q':<4} {'matched bp':>11} {'t':>7} "
                  f"{'mismatched bp':>14} {'t':>7} {'days':>6}")
            for q, cell in results[axis_key][investor].items():
                m = cell[f"univariate_{matched}"][f"S_{matched}_{investor}"]
                x = cell[f"univariate_{mismatched}"][f"S_{mismatched}_{investor}"]
                print(f"{q:<4} {m['coef_bp']:>+11.2f} {m['t_stat']:>+7.2f} "
                      f"{x['coef_bp']:>+14.2f} {x['t_stat']:>+7.2f} "
                      f"{cell['univariate_' + matched]['n_days']:>6}")

    print("\nMarket-cap quintile composition (share of each sample's stock-days):")
    print(f"{'sample':<16} {'stock-days':>11} "
          + " ".join(f"{q:>7}" for q in QUINTILE_LABELS))
    for name, comp in results["size_composition"].items():
        print(f"{name:<16} {comp['n_stock_days']:>11,} "
              + " ".join(f"{100 * comp['share'][q]:>6.1f}%" for q in QUINTILE_LABELS))


if __name__ == "__main__":
    main()
