# OrientationTool

# Algorithm summary:
# 1. Find the cutoff nearest neighbors
# 2. Invoke the 2 orientation values for two neighbors in the form of quaternions
# 3. Quaternion multiplications for calculating misorientation
# 4. Apply 24 symmetry group to find the minimum misorientation
# 5. Average of ~12 neighbors misorientation for each particle
# 6. Adding this average_misorientation as a new property of the particle
