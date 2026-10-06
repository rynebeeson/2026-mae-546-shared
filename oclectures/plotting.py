"""Shared matplotlib styling for the lecture notebooks."""
import matplotlib.pyplot as plt

# Colorblind-friendly (Okabe-Ito) palette
COLORS = {
    'dymos': '#0072B2',      # blue   — numerical solution
    'analytic': '#D55E00',   # orange — analytic / reference solution
    'secondary': '#009E73',  # green
    'accent': '#CC79A7',     # magenta
    'gray': '#888888',
}


def apply_style():
    """Apply a consistent, lecture-friendly plot style."""
    plt.rcParams.update({
        'figure.figsize': (9, 4),
        'figure.dpi': 110,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 11,
        'axes.titlesize': 12,
        'legend.frameon': False,
        'lines.linewidth': 2,
    })
