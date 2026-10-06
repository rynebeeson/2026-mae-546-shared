"""Generate the brachistochrone geometry schematic embedded in notebook 01.

Run:  python make_schematic.py   (from this directory, in the dymos-lectures env)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

from oclectures import apply_style, COLORS

apply_style()


def bezier(p0, p1, p2, s):
    s = np.atleast_1d(s)[:, None]
    return (1 - s)**2 * p0 + 2 * (1 - s) * s * p1 + s**2 * p2


P0, P1, P2 = np.array([0., 10.]), np.array([1.2, 4.8]), np.array([10., 5.])
wire = bezier(P0, P1, P2, np.linspace(0, 1, 100))

fig, ax = plt.subplots(figsize=(8, 5.2))
ax.plot(wire[:, 0], wire[:, 1], color=COLORS['gray'], lw=2.5,
        label='wire (shape to be found)')

# Bead and its velocity vector (tangent to the wire)
s0 = 0.42
bead = bezier(P0, P1, P2, s0)[0]
tang = (bezier(P0, P1, P2, s0 + 1e-4) - bezier(P0, P1, P2, s0 - 1e-4))[0]
tang /= np.linalg.norm(tang)
ax.plot(*bead, 'o', ms=13, color=COLORS['dymos'], zorder=5)
ax.annotate('', xy=bead + 2.2 * tang, xytext=bead,
            arrowprops=dict(arrowstyle='-|>', lw=2, color=COLORS['dymos']))
ax.text(*(bead + 2.2 * tang + np.array([0.1, -0.45])), r'$v$',
        color=COLORS['dymos'], fontsize=14)

# Downward vertical reference and the control angle theta
ax.plot([bead[0], bead[0]], [bead[1], bead[1] - 2.6], '--', color='0.4', lw=1.2)
theta_deg = np.degrees(np.arctan2(tang[0], -tang[1]))  # from downward vertical
ax.add_patch(Arc(bead, 2.9, 2.9, theta1=270, theta2=270 + theta_deg,
                 color='0.25', lw=1.5))
lbl = np.radians(270 + theta_deg / 2)
ax.text(bead[0] + 1.9 * np.cos(lbl), bead[1] + 1.9 * np.sin(lbl),
        r'$\theta$', fontsize=15)

# Gravity
ax.annotate('', xy=(8.8, 7.2), xytext=(8.8, 9.2),
            arrowprops=dict(arrowstyle='-|>', lw=2, color=COLORS['analytic']))
ax.text(9.05, 8.1, r'$g$', color=COLORS['analytic'], fontsize=14)

# Endpoints
ax.plot(*P0, 's', ms=8, color='k')
ax.plot(*P2, 's', ms=8, color='k')
ax.annotate('$A = (0, 10)$ m, released from rest', P0, xytext=(0.5, 10.3),
            fontsize=11)
ax.annotate('$B = (10, 5)$ m', P2, xytext=(8.3, 4.15), fontsize=11)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-0.8, 11.6)
ax.set_ylim(3.6, 11)
ax.set_aspect('equal')
ax.grid(False)
ax.set_title(r'The control $\theta(t)$: direction of the velocity, '
             'measured from the downward vertical')
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig('brachistochrone_schematic.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print('wrote brachistochrone_schematic.png')
