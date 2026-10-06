"""Tools for inspecting the NLP that dymos/OpenMDAO hand to the optimizer:
design-variable/constraint summaries and total-Jacobian visualization."""
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def _shorten(name):
    """Strip dymos/OpenMDAO path clutter from a variable name for display."""
    for prefix in ('traj.', 'traj0.', 'phase0.', 'phases.'):
        name = name.replace(prefix, '')
    # 'linkages.arc0:x_final|arc1:x_initial' -> 'defect arc0|arc1: x'
    name = re.sub(r'linkages\.(\w+):(\w+)_final\|(\w+):\2_initial',
                  r'defect \1|\3: \2', name)
    replacements = [
        ('control_values:', ''),
        ('controls:', ''),
        ('states:', ''),
        ('timeseries.', ''),
        ('collocation_constraint.defects:', 'defect: '),
        ('continuity_comp.defect_control_rates:', 'rate cont.: '),
        ('continuity_comp.defect_controls:', 'cont.: '),
        ('continuity_comp.defect_', 'cont.: '),
        ('final_boundary_constraints.final_value:', 'final '),
        ('initial_boundary_constraints.initial_value:', 'initial '),
        ('boundary_vals.', 'final '),
        ('_rate', ' rate'),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    return name


def describe_nlp(prob, title=None):
    """Print the design variables, constraints, and objective of the NLP seen
    by the optimizer attached to ``prob``."""
    dvs = prob.driver.get_design_var_values()
    cons = prob.driver.get_constraint_values()
    objs = prob.driver.get_objective_values()

    if title:
        print(f'=== {title} ===\n')

    print('Design variables (z):')
    n_dv = 0
    for name, val in dvs.items():
        print(f'  {_shorten(name):<42s} size {np.asarray(val).size:>4d}')
        n_dv += np.asarray(val).size

    print('\nConstraints (c(z)):')
    n_con = 0
    for name, val in cons.items():
        print(f'  {_shorten(name):<42s} size {np.asarray(val).size:>4d}')
        n_con += np.asarray(val).size

    print('\nObjective (f(z)):')
    for name in objs:
        print(f'  {_shorten(name)}')

    print(f'\n--> NLP with {n_dv} variables and {n_con} constraints')
    return n_dv, n_con


def total_jacobian(prob):
    """Compute the dense total Jacobian of ALL driver responses — the
    objective first, then every constraint, *including linear ones* — with
    respect to the design variables.

    (A bare ``prob.compute_totals()`` omits constraints that were declared
    linear, since the optimizer computes their constant Jacobian only once;
    here we request everything explicitly so the picture matches the full
    NLP.)

    Returns
    -------
    J : ndarray
        Dense (1 + n_constraints) x n_desvars Jacobian.
    of_blocks, wrt_blocks : list of (name, size)
        Row / column block structure, in the order they appear in ``J``.
    """
    wrt_names = list(prob.driver.get_design_var_values())
    of_names = (list(prob.driver.get_objective_values())
                + list(prob.driver.get_constraint_values()))
    totals = prob.compute_totals(of=of_names, wrt=wrt_names)
    of_sizes = {of: totals[(of, wrt_names[0])].shape[0] for of in of_names}
    wrt_sizes = {wrt: totals[(of_names[0], wrt)].shape[1] for wrt in wrt_names}
    J = np.block([[totals[(of, wrt)] for wrt in wrt_names] for of in of_names])
    of_blocks = [(_shorten(n), of_sizes[n]) for n in of_names]
    wrt_blocks = [(_shorten(n), wrt_sizes[n]) for n in wrt_names]
    return J, of_blocks, wrt_blocks


def plot_jacobian(J, of_blocks, wrt_blocks, ax=None, title='', tol=1e-12,
                  cmap='viridis', colorbar=True, obj_blocks=1):
    """Sparsity/magnitude plot of a total Jacobian with labeled blocks.

    Nonzero entries are colored by log-magnitude (with a colorbar); exact
    zeros stay white. Thin gray lines mark individual matrix entries, heavier
    lines the boundaries between variable/constraint blocks, and a dark rule
    separates the objective row(s) (the first ``obj_blocks`` row blocks, as
    ordered by ``total_jacobian``) from the constraints.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    A = np.abs(J)
    masked = np.ma.masked_where(A <= tol, A)
    vmin, vmax = masked.min(), masked.max()
    if not np.isfinite(vmin) or vmin <= 0 or vmin == vmax:
        norm = None
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(masked, norm=norm, cmap=cmap, interpolation='nearest',
                   aspect='auto')
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(r'$|\partial (f, c) \, / \, \partial z|$', fontsize=9)
        cb.ax.tick_params(labelsize=8)

    # Per-entry grid (only when the matrix is small enough to read it)
    if max(J.shape) <= 64:
        ax.set_xticks(np.arange(-0.5, J.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, J.shape[0], 1), minor=True)
        ax.grid(which='minor', color='0.9', lw=0.3)
        ax.tick_params(which='minor', length=0)

    # Block boundaries; a heavier rule below the objective row(s)
    r = 0
    for k, (_, size) in enumerate(of_blocks[:-1]):
        r += size
        heavy = (k == obj_blocks - 1)
        ax.axhline(r - 0.5, color='0.25' if heavy else '0.55',
                   lw=1.8 if heavy else 0.9)
    c = 0
    for _, size in wrt_blocks[:-1]:
        c += size
        ax.axvline(c - 0.5, color='0.55', lw=0.9)

    # Block labels at block centers
    centers, labels, pos = [], [], 0
    for name, size in of_blocks:
        centers.append(pos + size / 2 - 0.5)
        labels.append(f'{name}  ({size})' if size > 1 else name)
        pos += size
    ax.set_yticks(centers, labels, fontsize=8)
    centers, labels, pos = [], [], 0
    for name, size in wrt_blocks:
        centers.append(pos + size / 2 - 0.5)
        labels.append(f'{name}  ({size})' if size > 1 else name)
        pos += size
    ax.set_xticks(centers, labels, fontsize=8, rotation=45, ha='right')

    ax.set_xlabel('design variables $z$', fontsize=9)
    ax.set_ylabel('objective $f$, constraints $c$', fontsize=9)
    density = 100 * np.count_nonzero(A > tol) / A.size
    ax.set_title(f'{title}\n{J.shape[0]} x {J.shape[1]}, '
                 f'{density:.1f}% nonzero', fontsize=10)
    ax.grid(False)
    return ax
