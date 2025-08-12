#Algorithm summary:
#1. Find the cutoff nearest neighbors
#2. Invoke the 2 orientation values for two neighbors in the form of quaternions
#3. Quaternion multiplications for calculating misorientation
#4. Apply 24 symmetry group to find the minimum misorientation
#5. Average of ~12 neighbors misorientation for each particle
#6. Adding this average_misorientation as a new property of the particle


from ovito.io import import_file, export_file
from ovito.data import NearestNeighborFinder
from ovito.data import CutoffNeighborFinder
from ovito.data import *
from ovito.modifiers import *
from ovito.modifiers import PolyhedralTemplateMatchingModifier
import numpy as np
from numpy import linalg as LA
import math
#from pyquaternion import *


# Load input simulation file.
#node = import_file("dump.sig3_minimization_AfterFix_0.cfg")
#data = node.source

#Defining 24 symmetry groups
cubic_gcs = np.array([
[		 1,			 0,			 0,			 0],
[0.500000000000000,	 0.500000000000000,	 0.500000000000000,	 0.500000000000000],
[0.500000000000000,	-0.500000000000000,	-0.500000000000000,	-0.500000000000000],
[0.500000000000000,	-0.500000000000000,	 0.500000000000000,	 0.500000000000000],
[0.500000000000000,	 0.500000000000000,	-0.500000000000000,	 0.500000000000000],
[0.500000000000000,	 0.500000000000000,	 0.500000000000000,	-0.500000000000000],
[0.500000000000000,	-0.500000000000000,	-0.500000000000000,	 0.500000000000000],
[0.500000000000000,	 0.500000000000000,	-0.500000000000000,	-0.500000000000000],
[0.500000000000000,	-0.500000000000000,	 0.500000000000000,	-0.500000000000000],
[		 0,			 0,			 1,			 0],
[		 0,			 0,		 	 0,			 1],
[		 0,			 1,			 0,			 0],
[		 0,	 0.707106781186548,			 0,	-0.707106781186548],
[		 0,	 0.707106781186548,			 0,	 0.707106781186548],
[0.707106781186548,			 0,	 0.707106781186548,	 		 0],
[0.707106781186548,			 0,	-0.707106781186548,			 0],
[		 0,			 0,	 0.707106781186548,	-0.707106781186548],
[0.707106781186548,	 0.707106781186548,			 0,			 0],
[0.707106781186548,	-0.707106781186548,			 0,			 0],
[		 0,			 0,	 0.707106781186548,	 0.707106781186548],
[		 0,	 0.707106781186548,	-0.707106781186548,			 0],
[0.707106781186548,	 		 0,			 0,	-0.707106781186548],
[		 0,	 0.707106781186548,	 0.707106781186548,			 0],
[0.707106781186548,			 0,			 0,	 0.707106781186548]
])

xyz = np.copy(cubic_gcs)

def modify(frame, data , output):
	cutoff=3.5
	finder = CutoffNeighborFinder (cutoff, data)
	# Loop over all input particles:
	for index in range(data.number_of_particles):
		min_degree_list=[]
		print("Cutoff neighbors of particle %i:" % index)
		# Iterate over the neighbors of the current particle, starting with the closest:
		for neigh in finder.find(index):

			q=output.particle_properties['Orientation'].marray[index]
			h=output.particle_properties['Orientation'].marray[neigh.index]
		
			conj=np.array([1, -1, -1, -1])
			#reordering the quaternions
			myorder = [3, 0, 1, 2]
			q_reordered = [q[j] for j in myorder]
			h_reordered = [h[s] for s in myorder]

			#conj=np.array([-1, -1, -1, 1])
			r=np.multiply(q_reordered,conj)
			#r=q
			a=h_reordered
			#product=(r*a)
			b0 = r[0] * a[0] - r[1] * a[1] - r[2] * a[2] - r[3] * a[3]
			b1 = r[0] * a[1] + r[1] * a[0] + r[2] * a[3] - r[3] * a[2]
			b2 = r[0] * a[2] - r[1] * a[3] + r[2] * a[0] + r[3] * a[1]
			b3 = r[0] * a[3] + r[1] * a[2] - r[2] * a[1] + r[3] * a[0]
			b=np.array([b0, b1, b2, b3])
			thetaList=[]
			# apply 24 symmetry group
			for i in range (24):
				c=xyz[i]
				#product=(c*b)
				d0 = c[0] * b[0] - c[1] * b[1] - c[2] * b[2] - c[3] * b[3]
				d1 = c[0] * b[1] + c[1] * b[0] + c[2] * b[3] - c[3] * b[2]
				d2 = c[0] * b[2] - c[1] * b[3] + c[2] * b[0] + c[3] * b[1]
				d3 = c[0] * b[3] + c[1] * b[2] - c[2] * b[1] + c[3] * b[0]
				d=np.array([d0, d1, d2, d3])
				#if d0>1 or d0<-1:
					#print('d0 = ',d0)
				d0_limited=np.clip(d0,-1,1)
				theta=2.0*np.arccos(d0_limited)
				thetaDegree=math.degrees(theta)

				thetaList.append(thetaDegree)
				#print(i, thetaDegree)
				#print(d0,d0_limited)
			minDegree=min(thetaList)
			min_degree_list.append(minDegree)
			#print(thetaList)
			print(minDegree)
			
		#print(min_degree_list)
		average_minDegree=np.average(min_degree_list)
		
		if average_minDegree > 15:
			average_minDegree=15000
		elif average_minDegree > 3 and average_minDegree <= 15:
			average_minDegree=3000
		
		print('average misoriention = ',average_minDegree)
		fundamental_property = output.create_user_particle_property("average_misorientation", "float").marray
		fundamental_property[index] = average_minDegree

