#!/usr/bin/env python3
# Hayley Wragg 2019-03-20
''' The code saves the values for the parameters in a ray tracer '''
import numpy as np
import sys

#FIXME Allow input of ray number change yes or no. Then save the direction from the source.
# These should then be imported into the main program from a saved numpy file rather than computing every time.
# Compute Directions in here if Nra changes not in main program
def DeclareParameters():
  print('Saving the parameters in ParameterInput.py')
  Nra=200
  Nre=5
  h=1

  # Obstacles are all triangles in 3D.
  triangle1 =np.array([(0.0,0.0,0.0),(3.0, 0.0,0.0),(0.0,0.0,3.0)])
  triangle2 =np.array([(0.0,0.0,3.0),(3.0, 0.0,3.0),(0.0,0.0,0.0)])
  triangle3=np.array([(0.0,0.0,0.0),(0.0, 3.0,0.0),(0.0,0.0,3.0)])
  triangle4 =np.array([(0.0,0.0,3.0),(0.0, 3.0,0.0),(0.0,3.0,3.0)])
  triangle5 =np.array([(0.0,0.0,0.0),(3.0, 3.0,0.0),(0.0,3.0,0.0)])
  triangle6 =np.array([(0.0,0.0,0.0),(3.0, 3.0,0.0),(3.0,0.0,0.0)])
  triangle7 =np.array([(0.0,0.0,3.0),(3.0, 3.0,3.0),(3.0,0.0,3.0)])
  triangle8 =np.array([(0.0,0.0,3.0),(3.0, 3.0,3.0),(0.0,3.0,3.0)])
  triangle9 =np.array([(3.0,0.0,3.0),(3.0, 3.0,3.0),(3.0,3.0,0.0)])
  triangle10=np.array([(3.0,0.0,3.0),(3.0, 0.0,0.0),(3.0,3.0,0.0)])
  triangle11=np.array([(3.0,3.0,0.0),(3.0, 3.0,3.0),(0.0,3.0,3.0)])
  triangle12=np.array([(3.0,3.0,0.0),(0.0, 3.0,3.0),(0.0,3.0,0.0)])
  Oblist=2*np.array([triangle1,triangle2,triangle3,triangle4,triangle5,triangle6,triangle7,triangle8,triangle9,triangle10,triangle11,triangle12])

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

  Tx=np.array([6.75,6.25,1.5]) # -Router location -co-ordinate of three real numbers
  #(the third is zero when modelling in 2D).
  deltheta      =(np.sqrt(2.0))*(np.pi/(np.sqrt(Nra)+np.sqrt(2))) # Calculate angle spacing
  xysteps       =int(2.0*np.pi/deltheta)
  zsteps        =int(np.pi/deltheta-2)
  Nra           =xysteps*zsteps+2
  # ^^ Due to need of integer steps the input number of rays can not
  # always be used if everything is equally spaced ^^
  theta1        =deltheta*np.arange(xysteps)
  theta2        =deltheta*np.arange(1,zsteps+1)
  xydirecs      =np.transpose(np.vstack((np.cos(theta1),np.sin(theta1))))
  z             =np.tensordot(np.cos(theta2),np.ones(xysteps),axes=0)
  directions    =np.zeros((Nra,4))
  directions[0] =np.array([0.0,0.0,1.0,0.0])
  directions[-1]=np.array([0.0,0.0,-1.0,0.0])
  # Form the xyz co-ordinates matrix
  #FIXME try to form this without a loop
  for j in range(1,zsteps+1):
      st=(j-1)*xysteps+1
      ed=(j)*xysteps+1
      sinalpha=np.sin(theta2[j-1])
      coords=np.c_[sinalpha*xydirecs,z[j-1]]
      directions[st:ed]=np.c_[coords,np.zeros(xysteps)]
  RTPar=np.array([Nra,Nre,h])
  np.save('Parameters/Raytracing.npy',RTPar)
  print('Number of requested rays ', Nra)
  print('Number of reflections ', Nre)
  print('Mesh spacing ', h)
  np.save('Parameters/Directions.npy',directions)
  np.save('Parameters/Obstacles.npy',Oblist)
  np.save('Parameters/OuterBoundary.npy',OuterBoundary)
  np.save('Parameters/Origin.npy',Tx)
  print('Origin of raytracer ', Tx)
  print('------------------------------------------------')
  print('Geometrical parameters saved')
  print('------------------------------------------------')
  return 1

def ObstacleCoefficients():
  '''Retrieve the information saved in Declare parameters. Then define the obstacle coefficients. '''
  print('Saving the material parameters in ParameterInput.py')
  Oblist        =np.load('Parameters/Obstacles.npy')
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')
  Oblist        =np.concatenate((Oblist,OuterBoundary),axis=0)
  Nob=len(Oblist)
  RTPar=np.load('Parameters/Raytracing.npy')
  Nra=RTPar[0]
  Nre=RTPar[1]
  h=RTPar[2]
  Na=int(Nob*Nre+1)
  Nb=int((Nre)*(Nra)+1)                        # Number of columns on each Mesh element
  Nob=len(Oblist)                              # The Number of obstacle.
  mur=np.full((Nob,1), complex(1.0,0))         # Currently mur is
                                               # the same for every obstacle.
                                               # Array created to get functions correct.
  epsr=np.full((Nob,1),complex(3.6305,7.41E-2))# Currently epsr is the
                                               # same for every obstacle
  sigma=np.full((Nob,1),1.0E-2)                # For this test sigma is the
                                               # same for every obstacle

  # PHYSICAL CONSTANTS
  mu0=4*np.pi*1E-6
  c=2.99792458E+8
  eps0=1/(mu0*c**2)#8.854187817E-12
  Z0=(mu0/eps0)**0.5 #120*np.pi
  Freespace=np.array([mu0,eps0,Z0,c])

  # CALCULATE PARAMETERS
  frequency=2*np.pi*2.43E+9                       # 2.43 GHz
  gamma=np.sqrt(np.divide(complex(0,frequency*mu0)*mur,np.multiply(sigma,eps0*frequency*complex(0,1)*epsr)))
  Znob=Z0*np.divide((1+gamma),(1-gamma)   )   # Characteristic impedance of the obstacles
  #Znob=np.transpose(np.tile(Znob,(Nb,1)))    # Tile the obstacle coefficient number to be the same size as a mesh array.
  refindex=np.sqrt(np.multiply(mur,epsr))     # Refractive index of the obstacles

  np.save('Parameters/FreeSpace.npy',Freespace)
  np.save('Parameters/frequency.npy',frequency)
  print('Permittivity of free space ', mu0)
  print('Permeability of free space ', eps0)
  print('Characteristic Impedence ', Z0)
  print('Speed of light ', c)
  print('Number of obstacles',Nob)
  print('Relative Impedance of the obstacles ', Znob)
  print('Refractive index of the obstacles ', refindex)
  Znob=np.tile(Znob,Nre)                      # The number of rows is Nob*Nre+1. Repeat Nob
  Znob=np.insert(Znob,0,complex(0.0,0.0))     # Use a zero for placement in the LOS row
  refindex=np.tile(refindex,Nre)              # The number of rows is Nob*Nre+1. Repeat refindex
  refindex=np.insert(refindex,0,complex(0,0)) # Use a zero for placement in the LOS row
  np.save('Parameters/Znob.npy',Znob)
  np.save('Parameters/refindex.npy',refindex)
  print('------------------------------------------------')
  print('Material parameters saved')
  print('------------------------------------------------')
  return 1

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  out=DeclareParameters()
  out=ObstacleCoefficients()

  exit()



