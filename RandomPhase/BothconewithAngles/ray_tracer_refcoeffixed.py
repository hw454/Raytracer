#!/usr/bin/env python3
# Hayley Wragg 2018-05-29
''' Code to trace rays around a room using cones to account for
spreading. This version does not remove points inside an object.'''
import numpy as np
import matplotlib.pyplot as mp
import HayleysPlotting as hp
import reflection as ref
import intersection as ins
import linefunctions as lf
#import ray_tracer_test as rtest
import geometricobjects as ob
import roommesh as rmes
import math
import time


if __name__=='__main__':
  np.set_printoptions(precision=2,threshold=1e-12,suppress=True)
  # INPUT PARAMETERS FROM MEASUREMENTS

  # PLASTERBOARD DRY WALL
  #epsr=2.19*complex(1,1.11E-02) Plasterboard Drywall

  # RED BRICK DRY WALL
  #frequency=5.8E+8 # 5.8GHz, ref coef is 1/0.646 for 45 degree polarisation, 1/0.2512 for vertical polarisation
  #frequency=2.3E+9                                   # 2.3 GHz or 2*math.pi*(2.3E+9) for angular frequency
  #powerstreg=1                                   # The initial signal power in db
  ## The spacing is now found inside the uniform ray tracer function   # Spacing in the grid spaces.
  #bounds= np.array([10**-9, 10**2])               # The bounds within which the signal power is useful
  #mur=1.0                                        # Relative Permeability of Red brick dry Wall
  #epsr=5.86*complex(1,1.16E-01)                  # Relative Permittivity of Red brick dry
  #sigma=0.001
  ##measuredref=10**(-4.937E-2)
  #gamma=measuredref=10**(-4.4349E-1)
  #np.sqrt((complex(0,frequency*mur*mu0))/(complex(sigma,eps0*epsr*frequency)))

  # WOOD
  frequency=2*np.pi*2.43E+9                       # 2.43 GHz
  powerstreg=3.16                                 # mW taken from -5 dBm
  bounds= np.array([10**-9, 10**2])               # The bounds within which the signal power is useful
  mur=1.0
  epsr=complex(3.6305,7.41E-2)                    # Dry Wood
  #gamma= no measured coef
  sigma=1.0E-2

  # PHYSICAL CONSTANTS
  mu0=4*np.pi*1E-6
  c=2.99792458E+8
  eps0=1/(mu0*c**2)#8.854187817E-12
  Z1=(mu0/eps0)**0.5 #120*np.pi

  # CALCULATE PARAMETERS
  streg=complex(((1/(epsr*eps0))*powerstreg)**0.5,0.0)          # Initial field strength
  gamma=np.sqrt((complex(0,frequency*mur*mu0))/(complex(sigma,eps0*epsr*frequency)))
  #refloss=1/0.2512
  #refloss2=1/0.6457            # Max reflection coefficient
  Z2=Z1*(1+gamma)/(1-gamma)     # Characteristic impedance of the obstacles
  refindex=np.sqrt(mur*epsr)    # Refractive index of the obstacles

  # DEFINE PARAMETERS FOR THE CODE
  #m=int(math.ceil(np.log(powerstreg/bounds[0])/np.log(1/measuredref)))     # number of reflections observed
  m=10                       # number of reflections observed
  #n=10                      # number of rays emitted from source
  l=5                        # number of n's
  n=10                       # Initial number of rays will be 2*n
  ave=5                      # number of runs to average over
  origin=(5,1)               # source of the signal
  i=1                        # The figure number for the room plot
  origin1=(5,1)              # source of the signal
  origin2=(0,2)              # source of the signal
  outsidepoint1=(4,1)        # point for testing whether another point is inside or outside an object
  outsidepoint2=(5,2)        # point for testing whether another point is inside or outside an object
  outsidepoint3=(1,3)        # point for testing whether another point is inside or outside an object
  outsidepoint4=(0,3)        # point for testing whether another point is inside or outside an object
  print('Ray tracer begun for ',m,'number of reflections')
  # CONSTRUCT THE OBJECTS
  # Define the walls and construct the room
  wall1=ob.Wall_segment(np.array([-1.0,-1.0]),np.array([ 6.0,-1.0]))
  wall2=ob.Wall_segment(np.array([-1.0,-1.0]),np.array([-1.0, 4.0]))
  wall3=ob.Wall_segment(np.array([-1.0, 4.0]),np.array([ 6.0, 4.0]))
  wall4=ob.Wall_segment(np.array([ 6.0, 4.0]),np.array([ 6.0,-1.0]))
  # Define the object1 in the room
  Box1=ob.Wall_segment(np.array([-0.5, 0.0]),np.array([-0.5, 1.0]))
  Box2=ob.Wall_segment(np.array([-0.5, 1.0]),np.array([ 0.0, 1.0]))
  Box3=ob.Wall_segment(np.array([ 0.0, 1.0]),np.array([ 0.0, 0.0]))
  Box4=ob.Wall_segment(np.array([ 0.0, 0.0]),np.array([-0.5, 0.0]))
  # Define Object 2 in the room
  sofa1=ob.Wall_segment(np.array([ 1.0, 0.0]), np.array([ 1.0, 2.0]))
  sofa2=ob.Wall_segment(np.array([ 1.0, 2.0]), np.array([ 3.0, 2.0]))
  sofa3=ob.Wall_segment(np.array([ 3.0, 2.0]), np.array([ 3.0, 0.0]))
  sofa4=ob.Wall_segment(np.array([ 3.0, 0.0]), np.array([ 2.5, 0.0]))
  sofa5=ob.Wall_segment(np.array([ 2.5, 0.0]), np.array([ 2.5, 1.0]))
  sofa6=ob.Wall_segment(np.array([ 2.5, 1.0]), np.array([ 1.5, 1.0]))
  sofa7=ob.Wall_segment(np.array([ 1.5, 1.0]), np.array([ 1.5, 0.0]))
  sofa8=ob.Wall_segment(np.array([ 1.5, 0.0]), np.array([ 1.0, 0.0]))
  # Contain all the edges of the room
  obstacles=(wall1,wall2,wall3,wall4,Box1,Box2,Box3,Box4,sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)
  box=(Box1,Box2,Box3,Box4)
  sofa=(sofa1,sofa2,sofa3,sofa4,sofa5,sofa6,sofa7,sofa8)
  Room=ob.room((obstacles[0]))
  Room.roomconstruct(obstacles)
  Room.add_inside_objects(box)
  Room.add_inside_objects(sofa)
  # Declare variable to store theta n and n
  thetaNa=np.zeros((l,4))
  thetaNb=np.zeros((l,4))
  thetafile1=("Theta_values_location_1.txt")
  f=open(thetafile1,"w+")
  f.close()
  thetafile2=("Theta_values_location_2.txt")
  f=open(thetafile2,"w+")
  f.close()
  for j in range(0,l):
      print('number of rays', n)
      #Attempt at spreading the initial signal strength. This is actually accounted for in C_lambda streg=stregstart/n
      i,spacing,grid1a=Room.uniform_ray_tracer(origin1,outsidepoint1,outsidepoint2,n,ave,i,frequency,streg,m,Z2,Z1,refindex)
      i=i+2
      filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
      f=open(filename,"w+")
      (x,y)=Room.time
      f.write("Run times for first source location %.8f, %.8f" % (x,y))
      #f.write("Estimated P value" % y)
      f.close()
      #i,spacing,grid=Room.uniform_ray_tracer(origin,n,i+1,frequency,streg,m,refloss)
      i,spacing,grid1b=Room.uniform_ray_tracer(origin2,outsidepoint3,outsidepoint4,n,ave,i,frequency,streg,m,Z2,Z1,refindex)
      f=open(filename,"a+")
      for x in Room.time:
        f.write("Run times for second source location %.8f" % x)
      f.close()
      n=2*n
      print('number of rays', n)
      #Attempt at spreading the initial signal strength. This is actually accounted for in C_lambda streg=stregstart/n
      i,spacing,grid2a=Room.uniform_ray_tracer(origin1,outsidepoint1,outsidepoint2,n,ave,i,frequency,streg,m,Z2,Z1,refindex)
      mp.figure(i+1)
      resi_grid=grid1a.meshdiff(grid2a)
      thetaNa[j][0]=resi_grid[6][6]
      thetaNa[j][1]=n/2
      thetaNa[j][2]=-np.log(thetaNa[j][0])/np.log(thetaNa[j][1])
      if j>0:
        thetaNa[j][3]=np.log(thetaNa[j-1][0]/thetaNa[j][0])/np.log(2)
      f=open(thetafile1,"a+")
      (x,y,z,k)=thetaNa[j][:]
      f.write("%.8f, %.8f, %.8f, %.8f \n" % (x,y,z,k))
      f.close()
      mp.title('Residual- for %s rays and %s rays' %(n/2,n))
      mp.savefig('../../../../ConeFigures/ConeResidualan'+str(n/2)+'and2n'+str(n)+'.png',bbox_inches='tight')
      i=i+2
      filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
      f=open(filename,"w+")
      (x,y)=Room.time
      f.write("Run times for first source location %.8f, %.8f" % (x,y))
      #f.write("Estimated P value" % y)
      f.close()
      #i,spacing,grid=Room.uniform_ray_tracer(origin,n,i+1,frequency,streg,m,refloss)
      i,spacing,grid2b=Room.uniform_ray_tracer(origin2,outsidepoint3,outsidepoint4,n,ave,i,frequency,streg,m,Z2,Z1,refindex)
      mp.figure(i+1)
      resi_grid=grid1b.meshdiff(grid2b)
      thetaNb[j][0]=resi_grid[6][6]
      thetaNb[j][1]=n/2
      thetaNb[j][2]=-np.log(thetaNb[j][0])/np.log(thetaNb[j][1])
      if j>0:
        thetaNb[j][3]=np.log(thetaNb[j-1][0]/thetaNb[j][0])/np.log(2)
      f=open(thetafile2,"a+")
      (x,y,z,k)=thetaNb[j][:]
      f.write("%.8f, %.8f, %.8f, %.8f \n" % (x,y,z,k))
      f.close()
      mp.title('Residual- for %s rays and %s rays' %(n/2,n))
      mp.savefig('../../../../ConeFigures/ConeResidualbn'+str(n/2)+'and2n'+str(n)+'.png',bbox_inches='tight')
      f=open(filename,"a+")
      for x in Room.time:
        f.write("Run times for second source location %.8f" % x)
      f.close()
      #f.write("Estimated P value" % y)
  mp.figure(i+2)
  mp.plot(thetaNa[0:-1][0],thetaNa[0:-1][1])
  mp.title('Difference in Log Field_(0,0) for n rays and 2n rays against n')
  mp.savefig('../../../../ConeFigures/thetaNagainstNloc1.png')
  mp.figure(i+3)
  mp.plot(thetaNa[0:-1][0],thetaNa[0:-1][1])
  mp.title('Difference in Log Field_(0,0) for n rays and 2n rays against n')
  mp.savefig('../../../../ConeFigures/thetaNagainstNloc2.png')
  #mp.show()
  # TEST err=rtest.ray_tracer_test(Room, origin)
  # TEST PRINT print('error after rtest on room', err)
  exit()
