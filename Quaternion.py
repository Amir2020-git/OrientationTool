# Algorithm summary:
# 1) Find cutoff nearest neighbors
# 2) Read two neighbors' orientations as quaternions
# 3) Quaternion multiplication to get misorientation
# 4) Apply 24 cubic symmetries; take the minimum misorientation
# 5) Average ~12 neighbors per particle
# 6) Write "average_misorientation" per particle

from ovito.data import CutoffNeighborFinder
from ovito.modifiers import PolyhedralTemplateMatchingModifier  # kept for context; Orientation must exist upstream
import numpy as np
import math

# 24 proper rotations of the cubic point group (unit quaternions w, x, y, z)
CUBIC_SYM = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.5,  0.5,  0.5,  0.5],
    [0.5, -0.5, -0.5, -0.5],
    [0.5, -0.5,  0.5,  0.5],
    [0.5,  0.5, -0.5,  0.5],
    [0.5,  0.5,  0.5, -0.5],
    [0.5, -0.5, -0.5,  0.5],
    [0.5,  0.5, -0.5, -0.5],
    [0.5, -0.5,  0.5, -0.5],
    [0.0,  0.0,  1.0,  0.0],
    [0.0,  0.0,  0.0,  1.0],
    [0.0,  1.0,  0.0,  0.0],
    [0.0,  0.707106781186548, 0.0, -0.707106781186548],
    [0.0,  0.707106781186548, 0.0,  0.707106781186548],
    [0.707106781186548, 0.0,  0.707106781186548, 0.0],
    [0.707106781186548, 0.0, -0.707106781186548, 0.0],
    [0.0,  0.0,  0.707106781186548, -0.707106781186548],
    [0.707106781186548, 0.707106781186548, 0.0, 0.0],
    [0.707106781186548,-0.707106781186548, 0.0, 0.0],
    [0.0,  0.0,  0.707106781186548, 0.707106781186548],
    [0.0,  0.707106781186548, -0.707106781186548, 0.0],
    [0.707106781186548, 0.0, 0.0, -0.707106781186548],
    [0.0,  0.707106781186548,  0.707106781186548, 0.0],
    [0.707106781186548, 0.0, 0.0,  0.707106781186548]
], dtype=float)

# Fixed reordering and conjugation (to match your original convention)
_REORDER = (3, 0, 1, 2)          # map input (w,x,y,z?) into expected order
_CONJ    = np.array([1, -1, -1, -1], dtype=float)

def _quat_mul(a, b):
    """Hamilton product of quaternions a*b, each as [w,x,y,z]."""
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array([
        w0*w1 - x0*x1 - y0*y1 - z0*z1,
        w0*x1 + x0*w1 + y0*z1 - z0*y1,
        w0*y1 - x0*z1 + y0*w1 + z0*x1,
        w0*z1 + x0*y1 - y0*x1 + z0*w1
    ], dtype=float)

def _misorientation_deg(q, h):
    """Minimum misorientation angle (degrees) over 24 cubic symmetries for pair q,h."""
    # Reorder and conjugate as in your original
    r = np.array([q[j] for j in _REORDER], dtype=float) * _CONJ
    a = np.array([h[j] for j in _REORDER], dtype=float)
    b = _quat_mul(r, a)
    # Apply the 24 symmetry operators on the left: d = c * b
    d0_vals = []
    for c in CUBIC_SYM:
        d = _quat_mul(c, b)
        d0_vals.append(d[0])
    # Clamp to avoid numeric domain errors in arccos
    d0_max = np.clip(np.max(d0_vals), -1.0, 1.0)  # max(d0) minimizes angle
    theta = 2.0 * math.acos(d0_max)
    return math.degrees(theta)

def modify(frame, data, output):
    cutoff = 3.5
    finder = CutoffNeighborFinder(cutoff, data)

    # Orientation must exist (typically provided by PolyhedralTemplateMatchingModifier upstream)
    if 'Orientation' not in output.particle_properties:
        raise RuntimeError("Orientation property not found. "
                           "Apply PolyhedralTemplateMatchingModifier before this modifier.")

    orientations = output.particle_properties['Orientation'].marray
    n = data.number_of_particles

    # Create result property once and fill it
    avg_prop = output.create_user_particle_property("average_misorientation", "float").marray

    for index in range(n):
        theta_degs = []
        for neigh in finder.find(index):
            q = orientations[index]
            h = orientations[neigh.index]
            theta_degs.append(_misorientation_deg(q, h))

        if not theta_degs:
            avg_deg = float('nan')  # no neighbors within cutoff
        else:
            avg_deg = float(np.mean(theta_degs))

        # Map to your sentinel values, preserving your thresholds
        if avg_deg > 15:
            avg_prop[index] = 15000.0
        elif 3 < avg_deg <= 15:
            avg_prop[index] = 3000.0
        else:
            avg_prop[index] = avg_deg
