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

raylist       =np.empty([Nra+1, Nre+1,4])
deltheta      =(-2+np.sqrt(2.0*(Nra)))*(ma.pi/(Nra-2))
    xysteps       =int(2.0*ma.pi/deltheta)
    zsteps        =int(ma.pi/deltheta-2)
    Nra           =xysteps*zsteps+2
    # ^^ Due to need of integer steps the input number of rays can not
    # always be used if everything is equally spaced ^^
    theta1        =deltheta*np.arange(xysteps)
    theta2        =deltheta*np.arange(1,zsteps+1)
    xydirecs      =np.transpose(r*np.vstack((np.cos(theta1),np.sin(theta1))))
    z             =r*np.tensordot(np.cos(theta2),np.ones(xysteps),axes=0)
    directions    =np.zeros((Nra,4))
    directions[0] =np.array([0.0,0.0,r,0.0])
    directions[-1]=np.array([0.0,0.0,-r,0.0])
    # Form the xyz co-ordinates matrix
    #FIXME try to form this without a loop
    for j in range(1,zsteps+1):
      st=(j-1)*xysteps+1
      ed=(j)*xysteps+1
      sinalpha=np.sin(theta2[j-1])
      coords=np.c_[sinalpha*xydirecs,z[j-1]]
      directions[st:ed]=np.c_[coords,np.zeros(xysteps)]
    # Iterate through the rays find the ray reflections
    # FIXME the rays are independent of each toher so this is easily parallelisable
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =np.append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      raystart.multiref(s,Nre)
      raylist[it]=raystart.points[0:-2]

np.save('Parameters/Obstacles.npy',Oblist)
np.save('Parameters/OuterBoundary.npy',OuterBoundary)
np.save('Parameters/Origin.npy',Tx)


