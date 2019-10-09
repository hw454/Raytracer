#!/usr/bin/env python3
# Hayley Wragg 2019-03-20
''' The code saves the values for the parameters in a ray tracer '''
import numpy as np
import sys
import os

def DeclareParameters():
  '''All input parameters for the ray-launching method are entered in
  this function which will then save them inside a Parameters folder.

  * Nra - Number of rays

    .. note::

       Due to need of integer steps the input number of rays can not
       always be used if everything is equally spaced.

  * Nre - Number of reflections
  * Ns - Number of steps to split longest axis.
  * l1 - Interior obstacle scale
  * l2 - Boundary scale.
  * triangle1 - First interior obstacle
  * ...
  * triangleN - Last interior obstacle
  * OuterBoundary1 - First obstacle forming the boundary of the \
  environment
  * ...
  * OuterBoundaryN - Last obstacle forming the boundary of the \
  environment.

  :return: 0 if successfully completed.

  '''
  # -------------------------------------------------------------------
  # INPUT PARAMETERS FOR RAY LAUNCHER----------------------------------
  # -------------------------------------------------------------------

  print('Saving ray-launcher parameters')
  Nra=100 # Number of rays
  Nre=5  # Number of reflections
  Ns=30   # Number of steps on longest axis.
  l1=2   # Interior obstacle scale
  l2=9   # Outer Boundary length scale

  # Obstacles are all triangles in 3D.
  triangle1 =np.array([(0.0,0.0,0.0),(1.0, 0.0,0.0),(0.0,0.0,1.0)])
  triangle2 =np.array([(0.0,0.0,1.0),(1.0, 0.0,1.0),(0.0,0.0,0.0)])
  triangle3 =np.array([(0.0,0.0,0.0),(0.0, 1.0,0.0),(0.0,0.0,1.0)])
  triangle4 =np.array([(0.0,0.0,1.0),(0.0, 1.0,0.0),(0.0,1.0,1.0)])
  triangle5 =np.array([(0.0,0.0,0.0),(1.0, 1.0,0.0),(0.0,1.0,0.0)])
  triangle6 =np.array([(0.0,0.0,0.0),(1.0, 1.0,0.0),(1.0,0.0,0.0)])
  triangle7 =np.array([(0.0,0.0,1.0),(1.0, 1.0,1.0),(1.0,0.0,1.0)])
  triangle8 =np.array([(0.0,0.0,1.0),(1.0, 1.0,1.0),(0.0,1.0,1.0)])
  triangle9 =np.array([(1.0,0.0,1.0),(1.0, 1.0,1.0),(1.0,1.0,0.0)])
  triangle10=np.array([(1.0,0.0,1.0),(1.0, 0.0,0.0),(1.0,1.0,0.0)])
  triangle11=np.array([(1.0,1.0,0.0),(1.0, 1.0,1.0),(0.0,1.0,1.0)])
  triangle12=np.array([(1.0,1.0,0.0),(0.0, 1.0,1.0),(0.0,1.0,0.0)])

  #- Outer Boundary -
  # 3D co-ordinates forming a closed boundary.
  # Wall 1
  OuterBoundary1 =np.array([(0.0,0.0,0.3),(1.0,0.0,0.3),(1.0,0.0,0.0)])
  OuterBoundary2 =np.array([(1.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.3)])
  # Wall 2
  OuterBoundary3 =np.array([(1.0,0.0,0.0),(1.0,1.0,0.0),(1.0,1.0,0.3)])
  OuterBoundary4 =np.array([(1.0,0.0,0.3),(1.0,1.0,0.3),(1.0,0.0,0.0)])
  # Wall 3
  OuterBoundary5 =np.array([(1.0,1.0,0.3),(0.0,1.0,0.0),(1.0,1.0,0.0)])
  OuterBoundary6 =np.array([(1.0,1.0,0.3),(1.0,0.0,0.0),(1.0,0.0,0.3)])
  # Wall 4
  OuterBoundary7 =np.array([(0.0,1.0,0.3),(0.0,1.0,0.0),(0.0,0.0,0.0)])
  OuterBoundary8 =np.array([(0.0,1.0,0.3),(0.0,0.0,0.0),(0.0,0.0,0.3)])
  # Ceiling
  OuterBoundary9 =np.array([(0.0,0.0,0.3),(1.0,1.0,0.3),(1.0,0.0,0.3)])
  OuterBoundary10=np.array([(0.0,0.0,0.3),(0.0,1.0,0.3),(1.0,1.0,0.3)])
  # Floor
  OuterBoundary11=np.array([(0.0,0.0,0.0),(1.0,1.0,0.0),(1.0,0.0,0.0)])
  OuterBoundary12=np.array([(0.0,0.0,0.0),(0.0,1.0,0.0),(1.0,1.0,0.0)])

  # -Router location -co-ordinate of three real numbers
  Tx=np.array([0.45,0.417,0.1])

  # CONSTRUCT THE ARRAYS FOR STORING OBSTACLES
  Oblist=(1.0/l1)*np.array([triangle1,triangle2,triangle3,triangle4,triangle5,
  triangle6,triangle7,triangle8,triangle9,triangle10,triangle11,
  triangle12])

  OuterBoundary=np.array([OuterBoundary1,OuterBoundary2,
  OuterBoundary3, OuterBoundary4,OuterBoundary5,OuterBoundary6,
  OuterBoundary7, OuterBoundary8, OuterBoundary9, OuterBoundary10,
  OuterBoundary11,OuterBoundary12])

  # -------------------------------------------------------------------
  # CALCULATED PARAMETERS TO SAVE
  # -------------------------------------------------------------------

  # CALCULATE RELATIVE MESHWIDTH
  roomlengthscale=l2*abs(np.amax(OuterBoundary)-np.amin(OuterBoundary)) # SCALE WITHIN THE UNIT CO-ORDINATES.
  h=1.0/Ns

  # CALCULATE ANGLE SPACING
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

  # COMBINE THE RAY-LAUNCHER PARAMETERS INTO ONE ARRAY
  RTPar=np.array([Nra,Nre,h,roomlengthscale])

  # FORM THE ARRAY OF INITIAL RAY DIRECTIONS
  directions    =np.zeros((Nra,4))
  directions[0] =np.array([0.0,0.0,1.0,0.0])
  directions[-1]=np.array([0.0,0.0,-1.0,0.0])
  #FIXME try to form this without a loop
  for j in range(1,zsteps+1):
      st=(j-1)*xysteps+1
      ed=(j)*xysteps+1
      sinalpha=np.sin(theta2[j-1])
      coords=np.c_[sinalpha*xydirecs,z[j-1]]
      directions[st:ed]=np.c_[coords,np.zeros(xysteps)]

  print('Number of requested rays ', Nra)
  print('Number of reflections ', Nre)
  print('Mesh spacing ', h)
  print('Origin of raytracer ', Tx)

  # --------------------------------------------------------------------
  # SAVE THE PARAMETERS IN A FOLDER TITLED `Parameters`
  # --------------------------------------------------------------------
  if not os.path.exists('./Parameters'):
    os.makedirs('./Parameters')
  np.save('Parameters/Raytracing.npy',RTPar)
  np.save('Parameters/Directions.npy',directions)
  np.save('Parameters/Obstacles.npy',Oblist)
  np.save('Parameters/OuterBoundary.npy',OuterBoundary)
  np.save('Parameters/Origin.npy',Tx)
  print('------------------------------------------------')
  print('Geometrical parameters saved')
  print('------------------------------------------------')
  return 0

def ObstacleCoefficients():
  ''' Input the paramters for obstacles and the antenna. To ensure \
  arrays are of the right length for compatibility for the \
  ray-launcher retrieve the ray-launching parameters in \
  :py:func:`DeclareParameters()`

  Load:

  * 'Obstacles.npy'     -Co-ordinates of obstacles in the room
  * 'OuterBoundary.npy' - Co-ordinates of the walls of the room
  * 'Raytracing.npy'    -[Nra (number of rays), Nre (number of reflections), \
  h (relative meshwidth)]

  Calculate:

  * Nob=len([Obstacles,OuterBoundary])

  Input:

  * `Freespace` -[mu0 (permeability of air), \
  eps0 (permittivity of air),Z0 (characteristic impedance of air), \
  c (speed of light)]
  * `frequency` - :math:`\\omega` angular frequency of the wave out \
  the antenna.
  * `mur`       - :math:`\\mu_r` The relative permeability for all obstacles. \
  This should be an array with the same number of terms as the number \
  of obstacles Nob.
  * `epsr`     - :math:`\\epsilon_r` The relative permittivity for each obstacle. \
  This should be an array with the same number of terms as the number \
  of obstacles Nob.
  * `sigma`     - :math:`\\sigma` The electrical conductivity of the obstacles. \
  This should be an array with the same number of terms as the number \
  of obstacles.
  * `Gt`        - The gains of the antenna. The should be an array with \
  the same number of terms as the number of rays Nra.

  Calculate:

  * `eps0`   - :math:`\\epsilon_0=\\frac{1}{\\mu_0 c^2}`  permittivity of \
  freespace.
  * `Z0`     - :math:`Z_0=\\sqrt{\\frac{\\mu_0}{\\epsilon_0}}` characteristic \
  impedance of freespace.
  * `refindex` - The refreactive index \
  :math:`n=\\sqrt{\\mu_r\\epsilon_r}`
  * `Znobrat`- The relative impedance of the obstacles given by,
    :math:`\\hat{Z}_{Nob}=\\frac{Z_{Nob}}{Z_0}`. The impedance of each \
    obstacle :math:`Z_{Nob}` is given by \
    :math:`Z_{Nob}=\\sqrt{\\frac{i\\omega\\mu_0\\mu_r}{\\sigma+i\\epsilon_0\\epsilon_r}}`.


  The Znobrat and refindex terms are then reformatted so that they \
  repeat Nre times with an extra term. The extra term corresponds to \
  the line of sight path. This makes them the same length as a column \
  in a matrix in a :py:class:`DictionarySparseMatrix.DS`. \
  Each term corresponds to a possible obstacle reflection combination.

  The Gains matrix is also reformated to that it repeats (Nre+1) times. \
  This corresponds to every possible ray reflection number combination \
  This makes them the same length as a row in a matrix in a \
  :py:class:`DictionarySparseMatrix.DS`. \
  Each term corresponds to a possible obstacle reflection combination.

  Save:
  * `frequency.npy`- The angular frequency :math:`\\omega`.
  * `refindex.npy` - The refractive index of the obstacles.
  * `Znobrat.npy`  - The relative characteristic impedance.
  * `TxGains.npy`  - The gains of the antenna.
  * `Freespace.npy`- The freespace parameters.

  .. code::

     Freespace=np.array([mu0,eps0,Z0,c])

  :return: 0 if successfully completed.

  '''
  print('Saving the physical parameters for obstacles and antenna')

  # -------------------------------------------------------------------
  # RETRIEVE RAY LAUNCHER PARAMETERS FOR ARRAY LENGTHS-----------------
  # -------------------------------------------------------------------
  if not os.path.exists('./Parameters/'):
    os.makedirs('./Parameters/')
  RTPar         =np.load('Parameters/Raytracing.npy')
  Oblist        =np.load('Parameters/Obstacles.npy')
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)
  Nra=int(RTPar[0])                           # Number of rays
  Nre=int(RTPar[1])                           # Number of reflections
  Nob=len(Oblist)                             # The Number of obstacle.

  # -------------------------------------------------------------------
  # INPUT PARAMETERS FOR POWER CALCULATIONS----------------------------
  # -------------------------------------------------------------------

  # Gains of the rays
  Gt=np.ones((Nra,1),dtype=np.complex128)

  # PHYSICAL CONSTANTS Do not change for obstacles for frequency
  mu0=4*np.pi*1E-6
  c=2.99792458E+8
  frequency=2*np.pi*2.79E+08                   # 2.79 GHz #FIXME make this a table and choose a frequency option

  # Relative Constants for the obstacles
  mur=np.full((Nob,1), complex(3.0,0))         # For this test mur is
                                               # the same for every obstacle.
                                               # Array created to get functions correct.
  epsr=np.full((Nob,1),complex(2949.3, 0.1065))# For this test epsr is the
                                               # same for every obstacle

  sigma=np.full((Nob,1),1.0E-4)                # For this test sigma is the
                                               # same for every obstacle
  # CALCULATE FREESPACE PARAMETERS

  eps0=1/(mu0*c**2)  #8.854187817E-12
  Z0=(mu0/eps0)**0.5 #120*np.pi Characteristic impedance of free space.

  # STORE FREESPACE PARAMETERS IN ONE ARRAY
  Freespace=np.array([mu0,eps0,Z0,c])

  # CALCULATE OBSTACLE PARAMETERS
  top=complex(0,frequency*mu0)*mur
  bottom=sigma+complex(0,eps0*frequency)*epsr
  Znob =np.sqrt(top/bottom)                    # Wave impedance of the obstacles
  Znobrat=Znob/Z0
  refindex=np.sqrt(np.multiply(mur,epsr))     # Refractive index of the obstacles
  # CLEAR THE TERMS JUST FOR CALCULATION
  del top, bottom,Znob

  # PRINT THE PARAMETERS
  print('Permittivity of free space ', mu0)
  print('Permeability of free space ', eps0)
  print('Characteristic Impedence ', Z0)
  print('Speed of light ', c)
  print('Number of obstacles',Nob)
  print('Relative Impedance of the obstacles ', Znobrat.T)
  print('Refractive index of the obstacles ', refindex.T)

  # Make the refindex, impedance and gains vectors the right length to
  # match the matrices.
  Znobrat=np.tile(Znobrat,Nre)                    # The number of rows is Nob*Nre+1. Repeat Nob
  Znobrat=np.insert(Znobrat,0,complex(0.0,0.0))     # Use a zero for placement in the LOS row
  refindex=np.tile(refindex,Nre)
  refindex=np.insert(refindex,0,1+0j)
  Gt=np.tile(Gt,(Nre+1,1))

  # --------------------------------------------------------------------
  # SAVE THE PARAMETERS
  # --------------------------------------------------------------------

  np.save('Parameters/TxGains.npy', Gt)
  np.save('Parameters/Freespace.npy',Freespace)
  np.save('Parameters/frequency.npy',frequency)
  np.save('Parameters/Znobrat.npy',Znobrat)
  np.save('Parameters/refindex.npy',refindex)
  print('------------------------------------------------')
  print('Material parameters saved')
  print('------------------------------------------------')
  return 0

if __name__=='__main__':
  np.set_printoptions(precision=3)
  print('Running  on python version')
  print(sys.version)
  out=DeclareParameters()
  out=ObstacleCoefficients()

  exit()



