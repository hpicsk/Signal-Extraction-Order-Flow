"""Matplotlib styling shared across paper figures."""

import matplotlib.pyplot as plt

PAPER_RCPARAMS = {
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
}


def apply_paper_rcparams() -> None:
    plt.rcParams.update(PAPER_RCPARAMS)
