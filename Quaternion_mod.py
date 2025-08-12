#!/usr/bin/env python3
"""
Compute average cubic misorientation (in degrees) for each particle using OVITO.

Usage:
  python misorientation.py --input INPUT_FILE --output OUTPUT_CSV --cutoff 3.5

Notes:
- Requires OVITO (pip: `pip install ovito` or use OVITO's bundled Python).
- By default, runs Polyhedral Template Matching (PTM) to generate 'Orientation'.
- If your input already has an 'Orientation' property, you can skip PTM with --no-ptm.
"""

import sys
import math
import argparse
import numpy as np

try:
    from ovito.io import import_file, export_file
    from ovito.data import CutoffNeighborFinder
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
except Exception as e:
    print("ERROR: Could not import OVITO Python API. Install with `pip install ovito` "
          "or run this using OVITO's Python interpreter. Details:\n", e, file=sys.stderr)
    sys.exit(1)

# 24 proper rotations of the cubic point group as unit quaternions (w, x, y, z)
CUBIC_SYM = np.array([
    [1,0,0,0],
    [0.5, 0.5, 0.5, 0.5],
    [0.5,-0.5,-0.5,-0.5],
    [0.5,-0.5, 0.5, 0.5],
    [0.5, 0.5,-0.5, 0.5],
    [0.5, 0.5, 0.5,-0.5],
    [0.5,-0.5,-0.5, 0.5],
    [0.5, 0.5,-0.5,-0.5],
    [0.5,-0.5, 0.5,-0.5],
    [0,0,1,0],
    [0,0,0,1],
    [0,1,0,0],
    [0, np.sqrt(0.5), 0,-np.sqrt(0.5)],
    [0, np.sqrt(0.5), 0, np.sqrt(0.5)],
    [np.sqrt(0.5), 0, np.sqrt(0.5), 0],
    [np.sqrt(0.5), 0,-np.sqrt(0.5), 0],
    [0,0, np.sqrt(0.5),-np.sqrt(0.5)],
    [np.sqrt(0.5), np.sqrt(0.5), 0,0],
    [np.sqrt(0.5),-np.sqrt(0.5), 0,0],
    [0,0, np.sqrt(0.5), np.sqrt(0.5)],
    [0, np.sqrt(0.5),-np.sqrt(0.5), 0],
    [np.sqrt(0.5), 0,0,-np.sqrt(0.5)],
    [0, np.sqrt(0.5), np.sqrt(0.5), 0],
    [np.sqrt(0.5), 0,0, np.sqrt(0.5)]
], dtype=float)

def q_conj(q):
    # q = (w, x, y, z)
    return np.array([q[0], -q[1], -q[2], -q[3]])

def q_mul(a, b):
    # Hamilton product, both shape (4,)
    w = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
    x = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2]
    y = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1]
    z = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    return np.array([w,x,y,z])

def misorientation_deg(qi, qj, sym=CUBIC_SYM):
    """Smallest rotation angle (deg) between orientations qi and qj
       under both-side cubic symmetry: min_{g1,g2 in G} angle(g1 * qi^{-1} qj * g2^{-1})
    """
    # Normalize (defensive)
    qi = qi / np.linalg.norm(qi)
    qj = qj / np.linalg.norm(qj)
    # relative rotation
    b = q_mul(q_conj(qi), qj)
    # search over both-side symmetry
    min_ang = math.pi
    for g1 in sym:
        for g2 in sym:
            d = q_mul(g1, q_mul(b, q_conj(g2)))
            # enforce positive scalar (q ~ -q)
            d0 = abs(d[0] / np.linalg.norm(d))
            d0 = np.clip(d0, -1.0, 1.0)
            ang = 2.0 * math.acos(d0)
            if ang < min_ang:
                min_ang = ang
    return math.degrees(min_ang)

def build_pipeline(input_path, run_ptm=True):
    node = import_file(input_path)
    if run_ptm:
        node.modifiers.append(PolyhedralTemplateMatchingModifier(output_orientation=True))
    return node

def compute_average_misorientation(data, cutoff):
    """Creates/overwrites 'average_misorientation' particle property in 'data'."""
    if 'Orientation' not in data.particles:
        raise RuntimeError(
            "No 'Orientation' property found. "
            "Run with PTM (default) or provide an input that already contains 'Orientation'."
        )

    # Create the output property once
    avg_prop = data.particles_.create_property("average_misorientation", dtype=float)
    q_arr = np.asarray(data.particles['Orientation'].array)  # (N,4) (w,x,y,z)

    finder = CutoffNeighborFinder(cutoff, data)
    N = data.particles.count

    for i in range(N):
        mis_list = []
        for neigh in finder.find(i):
            j = neigh.index
            mis_list.append(misorientation_deg(q_arr[i], q_arr[j]))
        avg_prop.marray[i] = float(np.mean(mis_list)) if mis_list else float('nan')

def parse_args():
    p = argparse.ArgumentParser(description="OVITO: average cubic misorientation per particle")
    p.add_argument("--input", "-i", required=True, help="Input structure file (e.g., .cfg, .dump, .xyz)")
    p.add_argument("--output", "-o", default="misorientation.csv",
                   help="Output CSV path (default: misorientation.csv)")
    p.add_argument("--cutoff", "-r", type=float, default=3.5, help="Neighbor cutoff distance (default: 3.5)")
    p.add_argument("--no-ptm", action="store_true",
                   help="Do NOT run PTM; requires existing 'Orientation' property in input")
    return p.parse_args()

def main():
    args = parse_args()

    # Build pipeline and compute
    node = build_pipeline(args.input, run_ptm=not args.no_ptm)
    data = node.compute()

    # Compute average misorientation as a particle property
    compute_average_misorientation(data, cutoff=args.cutoff)

    # Export a tidy CSV table
    # (You can add more columns if needed; these names follow OVITO's property syntax.)
    columns = [
        "Particle Identifier",
        "Position.X", "Position.Y", "Position.Z",
        "average_misorientation"
    ]
    export_file(data, args.output, "txt/csv", columns=columns)
    print(f"Done. Wrote: {args.output}")

if __name__ == "__main__":
    main()
