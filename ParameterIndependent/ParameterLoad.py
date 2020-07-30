#!/usr/bin/env python3
# Hayley Wragg 2019-03-20
''' The code saves the values for the parameters in a ray tracer '''
import numpy as np
import math as ma
import sys
import os
import pickle
import openpyxl as wb

def DeclareParameters():
  '''All input parameters for the ray-launching method are entered in
  this function which will then save them inside a Parameters folder.

  * deltheta- The array of angle spacings.

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

  #print('Saving ray-launcher parameters')
  deltheta=np.pi*np.array([1/3])#,1/5,1/7,1/8,1/9,1/12,1/14,1/16,1/18,1/19,1/20,1/22,1/25,1/36])
  nrays=len(deltheta)
  Nra=np.ones((1,nrays),dtype=int)
  Nra=Nra[0]
  Nre=2            # Number of reflections
  Ns=5             # Number of steps on longest axis.
  split=4          # Number of steps through each mesh square
  l1=2.0           # Interior obstacle scale
  l2=3.0           # Outer Boundary length scale
  InnerOb=0        # Indicator of whether the inner obstacles should be used
  NtriOut=np.array([])# This will be the number of triangles forming each plane surface in the outer boundary
  NtriOb=np.array([]) # This will be the number of triangles forming each plane surface in the obstacle list
  ## Obstacles are all triangles in 3D.
  xmi=0.5
  xma=1.0
  ymi=0.5
  yma=1.0
  zmi=0.0
  zma=0.5
  Oblist=BoxBuild(xmi,xma,ymi,yma,zmi,zma)
  # In a box all surfaces are formed of two triangles
  Nbox=2*np.ones(6)
  NtriOb=np.append(NtriOb,Nbox)

  #- Outer Boundary -
  # 3D co-ordinates forming a closed boundary.
  xmi=0.0
  xma=1.0
  ymi=0.0
  yma=1.0
  zmi=0.0
  zma=1.0
  OuterBoundary=BoxBuild(xmi,xma,ymi,yma,zmi,zma)
  # In a box all surfaces are formed of two triangles
  Nbox=2*np.ones(6)
  NtriOut=np.append(NtriOut,Nbox)

  # -Router location -co-ordinate of three real numbers
  Tx=np.array([0.5,0.5,0.5])*l2

  runplottype= str('PerfectRefCentre')

  LOS=0     # LOS=1 for LOS propagation, LOS=0 for reflected propagation
  PerfRef=1 # Perfect reflection has no loss and ignores angles.

  # CONSTRUCT THE ARRAYS FOR STORING OBSTACLES
  Oblist=(1.0/l1)*Oblist

  OuterBoundary=l2*OuterBoundary

  # -------------------------------------------------------------------
  # CALCULATED PARAMETERS TO SAVE
  # -------------------------------------------------------------------

  # CALCULATE RELATIVE MESHWIDTH
  roomlengthscale=abs(np.amax(OuterBoundary)-np.amin(OuterBoundary)) # SCALE WITHIN THE UNIT CO-ORDINATES.
  #OuterBoundary=OuterBoundary/roomlengthscale
  #Oblist=Oblist/roomlengthscale
  h=1.0/Ns

  if not os.path.exists('./Parameters'):
    os.makedirs('./Parameters')

  # CALCULATE ANGLE SPACING
  for j in range(0,nrays):
    xysteps       =int(ma.ceil(abs(2.0*np.pi/deltheta[j])))
    theta1        =np.linspace(0.0,2*np.pi,num=int(xysteps), endpoint=False) # Create an array of all the angles
    deltheta[j]=theta1[1]-theta1[0]
    zsteps        =int(ma.ceil(abs(np.pi/(deltheta[j]))))
    Nra[j]=2
    Nraout=np.array([])
    start=0
    for k in range(-int(zsteps/2),int(zsteps/2)+1):
      mid=(np.cos(deltheta[j])-np.sin(k*deltheta[j])**2)/(np.cos(deltheta[j]*k)**2)
      if abs(mid)>1:
        pass
      else:
        bot=ma.acos(mid)
        xyk=int(2*np.pi/bot)
        if xyk<=1:
          break
        Nra[j]+=xyk
        theta1        =np.linspace(0.0,2*np.pi,num=int(xyk), endpoint=False) # Create an array of all the angles
        co=np.cos(k*deltheta[j])
        si=np.sin(k*deltheta[j])
        updirecs     =np.c_[co*np.cos(theta1),co*np.sin(theta1),si*np.ones((xyk,1))]
        #downdirecs   =np.c_[co*np.cos(theta1),co*np.sin(theta1),-si*np.ones((xyk,1))]
        if start==0:
          coords=updirecs
          start=1
        else:
          #coords  =np.r_[coords,downdirecs]
          coords  =np.r_[updirecs,coords]
    if len(coords)<=1:
      Nraout=Nra[j+1:]
      pass
    else:
      Nraout=Nra
      directions=np.zeros((Nra[j],4))
      directions[1:-1]=np.c_[coords,np.zeros((Nra[j]-2,1))]
      directions[0] =np.array([0.0,0.0, 1.0,0.0])
      directions[-1]=np.array([0.0,0.0,-1.0,0.0])
      directionname=str('Parameters/Directions'+str(int(j))+'.npy')
      np.save(directionname,directions)

  # # For comparing vector code to loop version
  # directions2=np.zeros((zsteps*xysteps+2,4))
  # for phi in range(0,zsteps):
    # c=np.cos(theta2[phi])
    # s=np.sin(theta2[phi])
    # for the in range(0,xysteps):
        # x=np.cos(theta1[the])*s
        # y=np.sin(theta1[the])*s
        # z=c
        # j=phi*xysteps+the+1
        # directions2[j]=np.array([x,y,z,0.0])
  # directions2[0] =np.array([0.0,0.0, 1.0,0.0])
  # directions2[-1]=np.array([0.0,0.0,-1.0,0.0])
  # print(np.sum(directions-directions2))

  # COMBINE THE RAY-LAUNCHER PARAMETERS INTO ONE ARRAY
  RTPar=np.array([Nre,h,roomlengthscale,split])

  print('Number of rays ', Nraout,'Number of reflections ', Nre,'Mesh spacing ', h)
  print('Angle spacing ', deltheta)
  #print('Origin of raytracer ', Tx)

  # --------------------------------------------------------------------
  # SAVE THE PARAMETERS IN A FOLDER TITLED `Parameters`
  # --------------------------------------------------------------------
  np.save('Parameters/Raytracing.npy',RTPar)
  np.save('Parameters/Nra.npy',Nraout)
  np.save('Parameters/delangle.npy',deltheta)
  np.save('Parameters/Obstacles.npy',Oblist)
  np.save('Parameters/NtriOb.npy',NtriOb)
  np.save('Parameters/InnerOb.npy',InnerOb)
  np.save('Parameters/OuterBoundary.npy',OuterBoundary)
  np.save('Parameters/NtriOut.npy',NtriOut)
  np.save('Parameters/Ns.npy',Ns)
  np.save('Parameters/Origin.npy',Tx)
  np.save('Parameters/LOS.npy',LOS)
  np.save('Parameters/PerfRef.npy',PerfRef)

  text_file = open('Parameters/runplottype.txt', 'w')
  n = text_file.write(runplottype)
  text_file.close()
  #print('------------------------------------------------')
  #print('Geometrical parameters saved')
  #print('------------------------------------------------')
  return 0

def ObstacleCoefficients(index=0):
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
  #print('Saving the physical parameters for obstacles and antenna')

  # -------------------------------------------------------------------
  # RETRIEVE RAY LAUNCHER PARAMETERS FOR ARRAY LENGTHS-----------------
  # -------------------------------------------------------------------
  if not os.path.exists('./Parameters/'):
    os.makedirs('./Parameters/')
  RTPar         =np.load('Parameters/Raytracing.npy')
  Nre,h,L,split       =RTPar
  Oblist        =np.load('Parameters/Obstacles.npy')
  OuterBoundary =np.load('Parameters/OuterBoundary.npy')
  Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)
  Nra           =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      nra=np.array([Nra])
  else:
      nra=len(Nra)
  Nre=int(RTPar[0])                           # Number of reflections
  Nob=np.load('Parameters/Nob.npy')          # The Number of obstacle.

  # -------------------------------------------------------------------
  # INPUT PARAMETERS FOR POWER CALCULATIONS----------------------------
  # -------------------------------------------------------------------

  # PHYSICAL CONSTANTS Do not change for obstacles for frequency
  mu0=4*np.pi*1E-6
  c=2.99792458E+8

  # ANTENNA PARAMETERS
  #-----------------------------------------------------------------------
  # Gains of the rays
  for j in range(0,nra):
    Gt=np.ones((Nra[j],1),dtype=np.complex128)
    gainname=str('Parameters/Tx'+str(Nra[j])+'Gains'+str(index)+'.npy')
    np.save(gainname, Gt)
  frequency=2*np.pi*2.79E+08                   # 2.79 GHz #FIXME make this a table and choose a frequency option
  khat          =frequency*L/c                 # Non-dimensional wave number
  lam           =(2*np.pi)/khat                # Non-dimensional wavelength
  Antpar        =np.array([khat,lam,L])
  Pol      =np.array([1.0,0.0])

  # OBSTACLE CONTSTANTS
  #----------------------------------------------------------------------
  # Relative Constants for the obstacles
  mur=np.full((Nob,1), complex(3.0,0))         # For this test mur is
                                               # the same for every obstacle.
                                               # Array created to get functions correct.
  epsr=np.full((Nob,1),complex(3.824,0.013))   # For this test epsr is the - For total absorption complex(2949.3, 0.1065))
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
  #Znobrat=np.ones((Nob,1),dtype=np.complex128)
  Znobrat=Znob/Z0
  #refindex=np.ones((Nob,1),dtype=np.complex128)
  refindex=np.zeros((Nob,1),dtype=np.complex128)          # Perfect Relection
  refindex[0]=1 #np.sqrt(np.multiply(mur[0],epsr[0]))     # Refractive index of the obstacles
  refindex[1]=1
  # CLEAR THE TERMS JUST FOR CALCULATION
  del top, bottom,Znob

  # --------------------------------------------------------------------
  # SAVE THE PARAMETERS
  # --------------------------------------------------------------------
  np.save('Parameters/Freespace'+str(index)+'.npy',Freespace)
  np.save('Parameters/frequency'+str(index)+'.npy',frequency)
  np.save('Parameters/lam'+str(index)+'.npy',lam)
  np.save('Parameters/khat'+str(index)+'.npy',khat)
  np.save('Parameters/Antpar'+str(index)+'.npy',Antpar)
  np.save('Parameters/Znobrat'+str(index)+'.npy',Znobrat[:,0])
  np.save('Parameters/refindex'+str(index)+'.npy',refindex[:,0])
  np.save('Parameters/Pol'+str(index)+'.npy',Pol)
  #print('------------------------------------------------')
  #print('Material parameters saved')
  #print('------------------------------------------------')
  del Freespace, frequency,lam,khat, Antpar, Znobrat,refindex,Pol,index
  return 0

def BoxBuild(xmi,xma,ymi,yma,zmi,zma):
  ''' Input the inimum and maximum x,y, and z co-ordinates which will form a Box.
  :param xmi: The minimum x co-ordinate.
  :param xma: The maximum x co-ordinate.
  :param ymi:The minimum y co-ordinate.
  :param yma: The maximum y co-ordinate.
  :param zmi: The minimum z co-ordinate.
  :param zma: The maximum z co-ordinate.

  .. code::

       Box=[T0,T1,...T12]
       TJ=[p0J,p1J,p2J]
       p0J=[x0J,y0J,z0J]
       p1J=[x1J,y1J,z1J]
       p2J=[x2J,y2J,x2J]

  :rtype: 12 x 3 x 3 numpy array.
  :returns: Box
  '''
  # The faces in the y=ymi plane
  triangle1 =np.array([(xmi,ymi,zmi),(xma,ymi,zmi),(xmi,ymi,zma)])
  triangle2 =np.array([(xmi,ymi,zma),(xma,ymi,zma),(xma,ymi,zmi)])
  # The faces in the x=xmi plane
  triangle3 =np.array([(xmi,ymi,zmi),(xmi,yma,zmi),(xmi,ymi,zma)])
  triangle4 =np.array([(xmi,ymi,zma),(xmi,yma,zmi),(xmi,yma,zma)])
  # The faces in the z=zmi plane
  triangle5 =np.array([(xmi,ymi,zmi),(xma,yma,zmi),(xmi,yma,zmi)])
  triangle6 =np.array([(xmi,ymi,zmi),(xma,yma,zmi),(xma,ymi,zmi)])
  # The faces in the z=zma plane
  triangle7 =np.array([(xmi,ymi,zma),(xma,yma,zma),(xma,ymi,zma)])
  triangle8 =np.array([(xmi,ymi,zma),(xma,yma,zma),(xmi,yma,zma)])
  # The faces in the x=xma plane
  triangle9 =np.array([(xma,ymi,zma),(xma,yma,zma),(xma,yma,zmi)])
  triangle10=np.array([(xma,ymi,zma),(xma,ymi,zmi),(xma,yma,zmi)])
  # The faces in the y=yma plane
  triangle11=np.array([(xma,yma,zmi),(xma,yma,zma),(xmi,yma,zma)])
  triangle12=np.array([(xma,yma,zmi),(xmi,yma,zmi),(xmi,yma,zma)])
  # Put the triangular faces into an array called Box.
  Box=np.array([triangle1,triangle2,triangle3,triangle4,triangle5,
  triangle6,triangle7,triangle8,triangle9,triangle10,triangle11,
  triangle12])
  return Box

if __name__=='__main__':
  np.set_printoptions(precision=3)
  print('Running  on python version')
  print(sys.version)
  Sheetname='InputSheet.xlsx'
  out=DeclareParameters(Sheetname)
  out=ObstacleCoefficients(Sheetname)

  exit()



