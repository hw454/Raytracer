#!/usr/bin/env python3
# Hayley Wragg 2019-03-20
''' The code saves the values for the parameters in a ray tracer '''
import numpy as np

#FIXME Allow input of ray number change yes or no. Then save the direction from the source.
# These should then be imported into the main program from a saved numpy file rather than computing every time.
# Compute Directions in here if Nra changes not in main program

Nra=300 # Minimum 3
Nre=6
h=0.25
RTPar=np.array([Nra,Nre,h])
np.save('Parameters/Raytracing.npy',RTPar)

# Obstacles are all triangles in 3D.
triangle1 =np.array([(0.0,0.0,0.0),(3.0, 0.0,0.0),(1.5,1.5,0.0)])
triangle2=np.array([(0.0,0.0,0.0),(3.0, 0.0,0.0),(1.0,1.0,3.0)])
triangle3=np.array([(0.0,0.0,0.0),(1.5, 1.5,0.0),(1.0,1.0,3.0)])
triangle4=np.array([(1.5,1.5,0.0),(1.0, 1.0,3.0),(3.0,0.0,0.0)])
Oblist=3*np.array([triangle1,triangle2,triangle3,triangle4])

#- Outer Boundary -
# 3D co-ordinates forming a closed boundary.
# Wall 1
OuterBoundary1 =np.array([(0.0,0.0,1.0),(3.0,0.0,1.0),(3.0,0.0,0.0)])
OuterBoundary2 =np.array([(3.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,1.0)])
# Wall 2
OuterBoundary3 =np.array([(3.0,0.0,0.0),(3.0,3.0,0.0),(3.0,3.0,1.0)])
OuterBoundary4 =np.array([(3.0,0.0,1.0),(3.0,3.0,1.0),(3.0,0.0,0.0)])
# Wall 3
OuterBoundary5 =np.array([(3.0,3.0,1.0),(0.0,3.0,0.0),(3.0,3.0,0.0)])
OuterBoundary6 =np.array([(3.0,3.0,1.0),(3.0,0.0,0.0),(3.0,0.0,1.0)])
# Wall 4
OuterBoundary7 =np.array([(0.0,3.0,1.0),(0.0,3.0,0.0),(0.0,0.0,0.0)])
OuterBoundary8 =np.array([(0.0,3.0,1.0),(0.0,0.0,0.0),(0.0,0.0,1.0)])
# Ceiling
OuterBoundary9 =np.array([(0.0,0.0,1.0),(3.0,3.0,1.0),(3.0,0.0,1.0)])
OuterBoundary10=np.array([(0.0,0.0,1.0),(0.0,3.0,1.0),(3.0,3.0,1.0)])
# Floor
OuterBoundary11=np.array([(0.0,0.0,0.0),(3.0,3.0,0.0),(3.0,0.0,0.0)])
OuterBoundary12=np.array([(0.0,0.0,0.0),(0.0,3.0,0.0),(3.0,3.0,0.0)])
OuterBoundary=5*np.array([OuterBoundary1,OuterBoundary2, OuterBoundary3,
 OuterBoundary4,OuterBoundary5,OuterBoundary6,OuterBoundary7,
 OuterBoundary8, OuterBoundary9, OuterBoundary10, OuterBoundary11,OuterBoundary12])

Tx=np.array([5.75,5.25,1.5]) # -Router location -co-ordinate of three real numbers
#(the third is zero when modelling in 2D).

np.save('Parameters/Obstacles.npy',Oblist)
np.save('Parameters/OuterBoundary.npy',OuterBoundary)
np.save('Parameters/Origin.npy',Tx)


