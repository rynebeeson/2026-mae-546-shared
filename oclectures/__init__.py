"""Shared helpers (plotting, widgets, problem builders) for the optimal
control lecture notebooks."""
from .plotting import COLORS, apply_style
from .nlp_tools import describe_nlp, total_jacobian, plot_jacobian
from .utils import quiet, redirect_outputs_to_tmp

__all__ = ['COLORS', 'apply_style', 'describe_nlp', 'total_jacobian',
           'plot_jacobian', 'quiet', 'redirect_outputs_to_tmp']
