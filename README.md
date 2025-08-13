Average Misorientation — OVITO Python Modifier

This OVITO Python modifier computes, for each particle, the average crystallographic misorientation (in degrees) to its cutoff neighbors, taking the minimum angle over the 24 proper rotations of the cubic point group. It then writes the result to a per-particle user property named average_misorientation.

What it does?
1. Finds neighbors within a user-set cutoff (default 3.5 Å)
2. For each neighbor pair (i, j), reads quaternions from the Orientation property
3. Computes misorientation by quaternion multiplication and applies all 24 cubic symmetries
4. Takes the minimum angle over symmetries (in degrees)
5. Averages the per-neighbor minima for each particle
6. Stores the average in average_misorientation (special sentinel values optional)
