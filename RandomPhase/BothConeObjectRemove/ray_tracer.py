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

  np.set_printoptions(precision=5,threshold=1e-12,suppress=True)
  # DEFINE PARAMETERS
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
  # Define the ray tracing parameters
  # DEFINE PARAMETERS FOR THE CODE
  #m=int(math.ceil(np.log(powerstreg/bounds[0])/np.log(1/measuredref)))     # number of reflections observed
  m= 5                     # number of reflections observed
  #n=10                       # number of rays emitted from source
  jmax=4                  # number of n's
  n=50                       # Initial number of rays will be 2*n
  ave=5                      # number of runs to average over
  origin=(5,1)               # source of the signal
  i=1                        # The figure number for the room plot
  origin1=(5,1)              # source of the signal
  origin2=(0,2)              # source of the signal
  outsidepoint1=(4,1)        # point for testing whether another point is inside or outside an object
  outsidepoint2=(5,2)        # point for testing whether another point is inside or outside an object
  outsidepoint3=(1,3)        # point for testing whether another point is inside or outside an object
  outsidepoint4=(0,4)        # point for testing whether another point is inside or outside an object
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
  # The spacing is now found inside the uniform ray tracer function spacing=0.25  # Spacing in the grid spaces.

  # Reflection Coefficient
  refloss=1/0.2512
  # Declare variable to store theta n and n
  thetaNaNoRND=np.zeros((jmax,6))
  thetaNbNoRND=np.zeros((jmax,6))
  thetaNaRND=np.zeros((jmax,6))
  thetaNbRND=np.zeros((jmax,6))
  thetafile1=("Theta_values_location_1.txt")
  f1=open(thetafile1,"w+")
  f1.write("theta values at location 1 ")
  f1.close()
  thetafile2=("Theta_values_location_1RND.txt")
  f2=open(thetafile2,"w+")
  f2.write("theta values at location 2 with RND phase")
  f2.close()
  thetafile3=("Theta_values_location_2.txt")
  f3=open(thetafile3,"w+")
  f3.write("theta values at location 2")
  f3.close()
  thetafile4=("Theta_values_location_2RND.txt")
  f4=open(thetafile4,"w+")
  f4.write("theta values at location 2 with RND phase")
  f4.close()
  for jit in range(1,jmax):
      #n=j*100
      print('number of rays', n)
      #Attempt at spreading the initial signal strength. This is actually accounted for in C_lambda streg=stregstart/n
      i,spacing,grid1aNoRND, grid1aRND=Room.uniform_ray_tracer(origin1,outsidepoint1,outsidepoint2,n,ave,i,frequency,streg,m,refloss)
      grid1aNoRND=np.ma.filled(grid1aNoRND,0.0)
      grid1aRND=np.ma.filled(grid1aRND,0.0)
      np.save('Heatmapgrids/HeatmapaRND'+str(m)+'Refs'+str(n)+'n.npy',grid1aRND)
      np.save('Heatmapgrids/HeatmapaNoRND'+str(m)+'Refs'+str(n)+'n.npy',grid1aNoRND)
      i=i+2
      #i,spacing=Room.uniform_ray_tracer_bounded(origin,n,i+1,frequency,streg,m,bounds,refloss)
      filename=("RuntimesN"+str(n)+"Delta"+str(int(spacing*100))+ ".txt")
      f=open(filename,"w+")
      (x,y)=Room.time
      f.write("Run times for first source location %.8f, %.8f" % (x,y))
      #f.write("Estimated P value" % y)
      f.close()
      #i,spacing,grid=Room.uniform_ray_tracer(origin,n,i+1,frequency,streg,m,refloss)
      i,spacing,grid1bNoRND, grid1bRND=Room.uniform_ray_tracer(origin2,outsidepoint3,outsidepoint4,n,ave,i,frequency,streg,m,refloss)
      grid1bNoRND=np.ma.filled(grid1bNoRND,0.0)
      grid1bRND=np.ma.filled(grid1bRND,0.0)
      np.save('Heatmapgrids/HeatmapbRND'+str(m)+'Refs'+str(n)+'n.npy',grid1bRND)
      np.save('Heatmapgrids/HeatmapbNoRND'+str(m)+'Refs'+str(n)+'n.npy',grid1bNoRND)
      f=open(filename,"a+")
      for x in Room.time:
        f.write("Run times for second source location %.8f" % x)
      f.close()
      # Power
      thetaNaNoRND[jit][0]=grid1aNoRND[2][2]#(8.854187817e-12)*(np.absolute(grid1aNoRND.grid[2][2])**2)
      thetaNbNoRND[jit][0]=grid1bNoRND[2][2]#(8.854187817e-12)*(np.absolute(grid1bNoRND.grid[2][2])**2)
      thetaNaRND[jit][0]=grid1aRND[2][2]#(8.854187817e-12)*(np.absolute(grid1aRND.grid[2][2])**2)
      thetaNbRND[jit][0]=grid1bRND[2][2]#(8.854187817e-12)*(np.absolute(grid1bRND.grid[2][2])**2)
      #n
      thetaNaNoRND[jit][2]=n
      thetaNbNoRND[jit][2]=n
      thetaNaRND[jit][2]=n
      thetaNbRND[jit][2]=n
      # Power/n
      thetaNaNoRND[jit][5]=thetaNaNoRND[jit][0]/thetaNaNoRND[jit][2]
      thetaNaRND[jit][5]=thetaNaRND[jit][0]/thetaNaRND[jit][2]
      thetaNbNoRND[jit][5]=thetaNbNoRND[jit][0]/thetaNbNoRND[jit][2]
      thetaNbRND[jit][5]=thetaNbRND[jit][0]/thetaNbRND[jit][2]
      if (jit >0):
        #resi_grida=grid1aNoRND.meshdiff(grid2aNoRND)
        #resi_grida_RND=grid1aRND.meshdiff(grid2aRND)
        #resi_gridb=grid1bNoRND.meshdiff(grid2bNoRND)
        #resi_gridb_RND=grid1bRND.meshdiff(grid2bRND)
        #np.save('Resigrids/ResiaRND'+str(m)+'Refs'+str(n)+'n.npy',resi_grida_RND)
        #np.save('Resigrids/ResiaNoRND'+str(m)+'Refs'+str(n)+'n.npy',resi_grida)
        #np.save('Resigrids/ResibRND'+str(m)+'Refs'+str(n)+'n.npy',resi_gridb_RND)
        #np.save('Resigrids/ResibNoRND'+str(m)+'Refs'+str(n)+'n.npy',resi_gridb)
        ##Diff between ppower for n/2 and n
        #thetaNaNoRND[jit][1]=abs(resi_grida[2][2])
        #thetaNaRND[jit][1]=abs(resi_grida_RND[2][2])
        #thetaNbNoRND[jit][1]=abs(resi_gridb[2][2])
        #thetaNbRND[jit][1]=abs(resi_gridb_RND[2][2])
        # log(dif)/log(n)
        thetaNaNoRND[jit][3]=-np.log(thetaNaNoRND[jit][1])/np.log(thetaNaNoRND[jit][2])
        thetaNaRND[jit][3]=-np.log(thetaNaRND[jit][1])/np.log(thetaNaRND[jit][2])
        thetaNbNoRND[jit][3]=-np.log(thetaNbNoRND[jit][1])/np.log(thetaNbNoRND[jit][2])
        thetaNbRND[jit][3]=-np.log(thetaNbRND[jit][1])/np.log(thetaNbRND[jit][2])
        # log(dif(n)/dif(2n))/log(2)approx alpha
        if (jit>1):
          thetaNaNoRND[jit][4]=np.log(thetaNaNoRND[jit-1][1]/thetaNaNoRND[jit][1])/np.log(2)
          thetaNaRND[jit][4]=np.log(thetaNaRND[jit-1][1]/thetaNaRND[jit][1])/np.log(2)
          thetaNbNoRND[jit][4]=np.log(thetaNbNoRND[jit-1][1]/thetaNbNoRND[jit][1])/np.log(2)
          thetaNbRND[jit][4]=np.log(thetaNbRND[jit-1][1]/thetaNbRND[jit][1])/np.log(2)
      else: i=i+4
      grid2aNoRND=grid1aNoRND
      grid2aRND=grid1aRND
      grid2bNoRND=grid1bNoRND
      grid2bRND=grid1bNoRND
      print('Output for first location '+str(n/2)+'rays')
      print(thetaNaNoRND[jit][:])
      f2=open(thetafile1,'a')
      (x,y,z,k,l,q)=thetaNaNoRND[jit][:]
      f2.write('%.8f, %.8f, %.8f, %.8f , %.8f, %.8f' % (x,y,z,k,l,q))
      f2.close()
      print('Output for random first location '+str(n/2)+'rays')
      print(thetaNaRND[jit][:])
      f4=open(thetafile2,'a')
      (x,y,z,k,l,q)=thetaNbRND[jit][:]
      f4.write('%.8f, %.8f, %.8f, %.8f , %.8f, %.8f' % (x,y,z,k,l,q))
      f4.close()
      print('Output for second location '+str(n/2)+'rays')
      print(thetaNbNoRND[jit][:])
      f2=open(thetafile3,'a')
      (x,y,z,k,l,q)=thetaNbNoRND[jit][:]
      f2.write('%.8f, %.8f, %.8f, %.8f , %.8f, %.8f' % (x,y,z,k,l,q))
      f2.close()
      print('Output for random second location '+str(n/2)+'rays')
      print(thetaNbNoRND[jit][:])
      f4=open(thetafile4,'a')
      (x,y,z,k,l,q)=thetaNbRND[jit][:]
      f4.write('%.8f, %.8f, %.8f, %.8f , %.8f, %.8f' % (x,y,z,k,l,q))
      f4.close()
      n=2*n
      #f.write("Estimated P value" % y)
  n=n/2
  thetaNaNoRND=np.transpose(thetaNaNoRND)
  thetaNbNoRND=np.transpose(thetaNbNoRND)
  thetaNaRND=np.transpose(thetaNaRND)
  thetaNbRND=np.transpose(thetaNbRND)
  np.save('OutputbNoRND'+str(m)+'Refs'+str(n)+'n.npy',thetaNbNoRND)
  np.save('OutputbRND'+str(m)+'Refs'+str(n)+'n.npy',thetaNbRND)
  np.save('OutputaNoRND'+str(m)+'Refs'+str(n)+'n.npy',thetaNaNoRND)
  np.save('OutputaRND'+str(m)+'Refs'+str(n)+'n.npy',thetaNaRND)
 # print(thetaNaNoRND)
  #print(thetaNbNoRND)
  #print(thetaNaRND)
  #print(thetaNbRND)
  mp.show()
  # TEST err=rtest.ray_tracer_test(Room, origin)
  # TEST PRINT print('error after rtest on room', err)
  exit()
