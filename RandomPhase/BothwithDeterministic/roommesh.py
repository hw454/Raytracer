#!/usr/bin/env python3
# Hayley Wragg 2017-07-18
''' Code to construct the mesh of the room '''

from math import atan2,hypot,sqrt,copysign
import numpy as np
import reflection as ref
import intersection as ins
import linefunctions as lf
import HayleysPlotting as hp
import matplotlib.pyplot as mp
import math as ma
from math import sin,cos,atan2,log
import numpy.linalg as lin
import random as rnd

class roommesh:
  ' a mesh representing a room'
  def __init__(s,xbounds,ybounds,spacing):
    ''' s has bounds which are the min and max x and y values and a grid
     storing the strength values '''
    s.bounds=np.vstack(
      (np.array(xbounds,dtype=np.float),
       np.array(ybounds,dtype=np.float),
    ))
    s.grid=np.zeros((abs(int((ybounds[1]-ybounds[0])/spacing)+1),abs(int((xbounds[1]-xbounds[0])/spacing)+1)),dtype=complex)
     # Co-ordinate followed by initial field strength
  def __getitem__(s,i,j):
    return s.grid[i][j]
  def __str__(s):
    return 'room mesh('+str(list(s.grid))+')'
  def __getspacing__(s):
    return abs((s.bounds[0][1]-s.bounds[0][0])/(len(s.grid[1])-1))
  def __getxrange__(s):
    return abs(int(s.bounds[0][1]/s.__getspacing__()))
  def __getyrange__(s):
    return abs(int(s.bounds[1][1]/s__getspacing__()))
  def __xmin__(s):
    return s.bounds[0][0]
  def __xmax__(s):
    return s.bounds[0][1]
  def __ymin__(s):
    return s.bounds[1][0]
  def __ymax__(s):
    return s.bounds[1][1]
  def singleraydeter(s,ray,iterconsts,f):
    ''' The field strength at the start of the ray is start assign this
    value to a mesh square and iterate through the ray '''
    streg=iterconsts[0]
    totdist=iterconsts[1]
    # Get the spacing between co-ordinates
    space=s.__getspacing__()
    # Find the position of the start of the ray
    j=int((ray[0][0]- s.__xmin__())/space+1)
    i=int((s.__ymax__()-ray[0][1])/space+1)
    # Find the direction of the ray
    direc=lf.Direction(ray)
    # Compute  the number of steps from one end of the ray to the other
    # If ray1=ray0+lam*d then n=lam/space
    n=int(np.dot((ray[1]-ray[0]),direc)/(np.dot(direc,direc)*space))
    point0=ray[0]
    #Compute the loss using that the length of each step should be the same.
    tmp=np.array([[0,space],[space,0]])
    alpha=(1/(lin.norm(direc)**2))*min(np.absolute(np.dot(tmp,direc)[0]),np.absolute(np.dot(tmp,direc)[1]))
    deldist=lf.length(np.array([(0,0),alpha*direc]))
    if deldist>np.sqrt(2*(space**2)):
        print('length greater than mesh')
    # Step through the ray decreasing the field strength
    for x in range(0,n-1):
      # Find the matrix position of the next point
      i2=int((s.__ymax__()-point0[1])/space)
      j2=int((point0[0]- s.__xmin__())/space)
      loss=((totdist+deldist)/totdist)**2
      # Check the strength hasn't run out
      #watstreg=10.0**(streg/10.0)
      #if watstreg <= 0:
         #return streg
      if i2 == i and j2 == j:
      # If a ray is going through the same box again take away the loss
      #but do not add the start again.
        #For db s.grid[i2][j2]-=loss
        s.grid[i2][j2]=s.grid[i2][j2]/loss
      else:
      # If a ray does not go through the same box twice add the signal
      #strength to the box
        s.grid[i2][j2]+=streg
        i=i2
        j=j2
      # Compute the field strength after loss
      #In db streg=streg-loss
      #phase=rnd.uniform(0,2)
      k=2*ma.pi*f/(299792458)
      phase=k*totdist
      phasechange=np.exp(ma.pi*phase*complex(0,1))
      streg=streg*phasechange/loss
      # Find the distance to the next step
      totdist=totdist+deldist
      # Set the starting point for the next iteration
      point0=space*direc+point0
    # Return the strength ready for the next ray
    return np.array([streg,totdist])
  def singleray(s,ray,iterconsts,f):
    ''' The field strength at the start of the ray is start assign this
    value to a mesh square and iterate through the ray '''
    streg=iterconsts[0]
    totdist=iterconsts[1]
    # Get the spacing between co-ordinates
    space=s.__getspacing__()
    # Find the position of the start of the ray
    j=int((ray[0][0]- s.__xmin__())/space+1)
    i=int((s.__ymax__()-ray[0][1])/space+1)
    # Find the direction of the ray
    direc=lf.Direction(ray)
    # Compute  the number of steps from one end of the ray to the other
    # If ray1=ray0+lam*d then n=lam/space
    n=int(np.dot((ray[1]-ray[0]),direc)/(np.dot(direc,direc)*space))
    point0=ray[0]
    #Compute the loss using that the length of each step should be the same.
    tmp=np.array([[0,space],[space,0]])
    alpha=(1/(lin.norm(direc)**2))*min(np.absolute(np.dot(tmp,direc)[0]),np.absolute(np.dot(tmp,direc)[1]))
    deldist=lf.length(np.array([(0,0),alpha*direc]))
    if deldist>np.sqrt(2*(space**2)):
        print('length greater than mesh')
    # Step through the ray decreasing the field strength
    for x in range(0,n):
      # Find the matrix position of the next point
      i2=int((s.__ymax__()-point0[1])/space)
      j2=int((point0[0]- s.__xmin__())/space)
      loss=(totdist+deldist)/totdist
      # Check the strength hasn't run out
      #watstreg=10.0**(streg/10.0)
      #if watstreg <= 0:
         #return streg
      if i2 == i and j2 == j:
      # If a ray is going through the same box again take away the loss
      #but do not add the start again.
        #For db s.grid[i2][j2]-=loss
        s.grid[i2][j2]=s.grid[i2][j2]/loss
      else:
      # If a ray does not go through the same box twice add the signal
      #strength to the box
        s.grid[i2][j2]+=streg
        i=i2
        j=j2
      # Compute the field strength after loss
      #In db streg=streg-loss
      streg=streg/loss
      # Find the distance to the next step
      totdist=totdist+deldist
      # Set the starting point for the next iteration
      point0=space*direc+point0
    # Return the strength ready for the next ray
    return np.array([streg,totdist])
  def bound(s,bounds):
    s.grid[np.absolute(s.grid)>bounds[1]]=bounds[1]
    s.grid[np.absolute(s.grid)<bounds[0]]=bounds[0]
    return
  def meshdiff(s,r):
    z1=s.grid
    z2=r.grid
    diffz=np.subtract(z1,z2)
    diffz=np.absolute(diffz)
    diffz=diffz/np.absolute(z1)
    #z1=10*np.log10(np.absolute(s.grid))
    ##print(r.grid)
    #z2=10*np.log10(np.absolute(r.grid))
    np.seterr(divide='ignore')
    mp.imshow(diffz, cmap='viridis', interpolation='nearest',vmin=0,vmax=5) #,extent=extent)
    mp.colorbar()
  def plot(s):
     ''' Plot a heatmap of the strength values '''
     np.seterr(divide='ignore')
     z=np.absolute(s.grid)
     #Convert to db
     z=10*np.ma.log10(z)
     extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
     mp.imshow(z, cmap='inferno', interpolation='nearest',extent=extent,vmin=-110,vmax=30)
     #mp.colorbar()
     return
  def plotbounded(s,bounds):
     ''' Plot a heatmap of the strength values '''
     z=np.absolute(s.grid)
     #Convert to db
     z=10*np.ma.log10(z)
     extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
     mp.imshow(z, cmap='viridis', interpolation='nearest',extent=extent,vmin=-110,vmax=30)
     #mp.colorbar()
     return
  def hist(s,i):
     z=s.grid
     z=10*np.ma.log10(np.absolute(z))
     mp.figure(i)
     h=np.histogram(z)
     h2=np.cumsum(h[0])
     #print(max(h2))
     h2=h2*(1.0/max(h2))
     mp.plot(h[1][:-1],h2)
     c=np.array([h[1][:-1],h2])
     mp.ylabel('Cumulative frequency')
     mp.xlabel('Field strength in dBm')
     mp.figure(i+1)
     mp.hist(z.flatten(),bins='auto')
     mp.ylabel('# of squares with value in range')
     mp.xlabel('Field strength in dBm')
     #mp.plot(h[1][:-1],h[0]) Plots the histogram as a line
     return c
  def histbounded(s,i):
     z=np.absolute(s.grid)
     z=10*np.ma.log10(z)
     mp.figure(i)
     h=np.histogram(z)
     h2=np.cumsum(h[0])
     #print(max(h2))
     h2=h2*(1.0/max(h2))
     mp.plot(h[1][:-1],h2)
     c=np.array([h[1][:-1],h2])
     mp.ylabel('Cumulative frequency')
     mp.xlabel('Field strength in dBm')
     mp.figure(i+1)
     mp.ylabel('# of squares with value in range')
     mp.xlabel('Field strength in dBm')
     mp.hist(z.flatten(),bins='auto')
     #mp.plot(h[1][:-1],h[0]) Plots the histogram as a line
     return c
  def teststrength(s):
    ray=np.array([[0.0,0.0],[10.0,10.0]])
    start=100000
    s.singleray(ray,start)
    #print(s.strengthvalues(ray,start))
    return
  def testmesh(s):
    s.constructmesh(0.0,10.0,0.0,10.0,1.0)
    return

