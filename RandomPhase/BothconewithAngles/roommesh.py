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
  def __init__(s,inside_points,objectcorners,xbounds,ybounds,spacing):
    ''' s has bounds which are the min and max x and y values and a grid
     storing the strength values '''
    s.bounds=np.vstack(
      (np.array(xbounds,dtype=np.float),
       np.array(ybounds,dtype=np.float),
    ))
    s.grid=np.zeros((abs(int((ybounds[1]-ybounds[0])/spacing)+1),abs(int((xbounds[1]-xbounds[0])/spacing)+1)),dtype=complex)
     # Co-ordinate followed by initial field strength
    s.emptycords=inside_points
    s.objectcorners=objectcorners
    #FIXME
  def __getitem__(s,i,j):
    return s.grid[i][j]
  def index_points(s,l):
    spacing=s.__getspacing__
    npo=s.objectcorners[l][0]          # The point that object starts in the co-ordinate list
    npend=s.objectcorners[l][1]
    j=int((s.emptycords[npo][0]-s.__xmin__)/spacing)
    i=int((s.__ymax__-s.emptycords[npo][1])/spacing)
    n=s.emptycords[l].shape[0]
    m=s.emptycords[l].shape[1]
    insidelist=np.empty((n,m), dtype=int)
    k=0
    for x in s.emptycords[npo:npend]:
      j=int((x[0]- s.__xmin__())/spacing)
      i=int((s.__ymin__()-x[1])/spacing)
      insidelist[k]=np.array([(i,j)])
    return insidelist
  def completeinsidelist(s,l):
    space=s.__getspacing__()
    # Find the position for the end and start of this object
    npo=s.objectcorners[l][0]
    npend=s.objectcorners[l][1]
    j=int((s.emptycords[npo][0]-s.__xmin__())/space)
    i=int((s.__ymax__()-s.emptycords[npo][1])/space)
    # Find the starting point and the neighbouring points
    P0=s.emptycords[npo]
    P1=s.emptycords[npo+1]
    P2=s.emptycords[npend]
    if npend-npo>3:
      P3=s.emptycords[npo+2]
    # Find the direction of the first line to work along and the two
    # lines which will shift P0 and P1 down.
    d1=lf.Direction(np.array([P0,P1]))
    d2=lf.Direction(np.array([P0,P2]))
    d3=lf.Direction(np.array([P1,P3]))
    # Find the number of points in the first line
    n=int(lin.norm(P1-P0)/space)
    # Find the number of steps down FIXME
    bigN=int(n**2)
    # Initialise the completelist
    completeinsidelist=np.empty((n*(bigN+1),2),dtype=int)
    # Add the starting point to the list
    completeinsidelist[0]=np.array([(i,j)])
    n2=int(lin.norm(P2-P0)/space)
    for y in range(1,n2+1):
      for x in range(1,n+1):
        nextP=P0+space*x*d1
        j=int((nextP[0]- s.__xmin__())/space)
        i=int((s.__ymax__()-nextP[1])/space)
        completeinsidelist[(y-1)*n+x]=np.array([(i,j)])
        print((y-1)*n+x,i,j)
      P0=P0+space*d2
      j=int((nextP[0]- s.__xmin__())/space)
      i=int((s.__ymax__()-nextP[1])/space)
      completeinsidelist[(y+1)*n]=np.array([(i,j)])
      print(y*n,i,j)
      P1=P1+space*P3
      n=int(lin.norm(P1-P0)/space)
      #FIXME stop when the next point is reached
    return completeinsidelist[0:int((n-1)*(n2-1))]
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
  def __greatesti__(s):
    return s.grid.shape[0]-1
  def __greatestj__(s):
    return s.grid.shape[1]-1
  def singlerayrndsum(s,ray,iterconsts,f,nrays):
    ''' The field strength at the start of the ray is start assign this
    value to a mesh square and iterate through the ray '''
    streg=iterconsts[0]
    totdist=np.real(iterconsts[1])
    # Get the spacing between co-ordinates
    space=s.__getspacing__()
    # Find the position of the start of the ray
    j=int((ray[0][0]- s.__xmin__())/space)
    i=int((s.__ymax__()-ray[0][1])/space)
    # Find the direction of the ray
    direc=lf.Direction(ray)
    normal=np.array([direc[1],-direc[0]])
    # Find the maximum i and j
    jmax=int((direc[0]*space+ray[1][0]- s.__xmin__())/space)
    imax=int((s.__ymax__()-ray[1][1]-direc[1]*space)/space)
    maxraylength=lf.length(np.array([ray[0],ray[1]]))
    #Compute the loss using that the length of each step should be the same.
    tmp=np.array([[0,space],[space,0]])
    alpha=(1/(lin.norm(direc)**2))*0.5*(np.absolute(np.dot(tmp,direc)[0])+np.absolute(np.dot(tmp,direc)[1]))
    deldist=lf.length(np.array([(0.0,0.0),alpha*direc]))
    # Compute  the number of steps from one end of the ray to the other
    # If ray1=ray0+lam*d then n=lam/space
    n=int(1+lin.norm(ray[1]-ray[0])/deldist)
    #n=int(1+np.dot((ray[1]-ray[0]),direc)/deldist)
    point0=ray[0]
    if deldist>np.sqrt(2*(space**2)):
        print('length greater than mesh')
    # RAY STRENGTH ITERATION
    # Step through the ray decreasing the field strength
    for x in range(0,2*n):
      # Find the number of squares for the cone
      #Nsqs=int(((ma.pi*totdist)/(nrays*space))*sqrt(2))
      #Nsqs=int(2*totdist*ma.sin(2*ma.pi/nrays)/(space))
      #Nsqs=4*sqrt(2)*totdist/(space*nrays)
      conedist=totdist*ma.tan(ma.pi/nrays)
      Nsqs=int(1+(conedist/deldist))
      # Find the matrix position of the next point
      i2=int((s.__ymax__()-point0[1])/space)
      j2=int((point0[0]- s.__xmin__())/space)
      if i2>s.__greatesti__() or  j2>s.__greatestj__():
        return np.array([streg,totdist])
      elif i2<0 or j2<0:
        return np.array([streg,totdist])
      else:
        if abs(i-i2)>abs(i-imax) and abs(j-j2)>abs(j-jmax):
          return np.array([streg,totdist])
 #       if abs(i-i2)==abs(i-imax) and abs(j-j2)==abs(j-jmax):
  #        return np.array([streg,totdist])
        loss=((totdist+deldist)/totdist)**2
        if i2 == i and j2 == j and x>1:
          # If a ray is going through the same box again take away the loss
          # but do not add the start again.
          # However check you are not at the start of the ray
          # For db s.grid[i2][j2]-=loss
          s.grid[i2][j2]=s.grid[i2][j2]/loss
        else:
          # If a ray does not go through the same box twice add the signal
          #strength to the box
          # Add streg to N/2 squares in +ve and -ve direction.
            s.grid[i2][j2]+=streg
            # CONE CALCULATION
            # Assign the value out to the squares in the cone
            for count in range(1,Nsqs+2):
              # Find the next point in the direction of the normal
              point1=count*alpha*normal+point0
              # Find position of the mesh of point1
              i3=int((s.__ymax__()-point1[1])/space)
              j3=int((point1[0]- s.__xmin__())/space)
              # Check this point lies within the mesh
              if 0<=i3<=s.__greatesti__() and 0<=j3<=s.__greatestj__():
                #Add the object bounds into the mesh FIXME
                #if abs(i-i3)<=abs(i2-imax) and abs(j-j3)<=abs(j-jmax):
                  s.grid[i3][j3]+=streg
              # Repeat in the negative normal direction
              point1=-count*alpha*normal+point0
              i3=int((s.__ymax__()-point1[1])/space)
              j3=int((point1[0]- s.__xmin__())/space)
              if 0<=i3<=s.__greatesti__() and 0<=j3<=s.__greatestj__():
                #if abs(i-i3)<=abs(i-imax) and abs(j-j3)<=abs(j-jmax):
                  s.grid[i3][j3]+=streg
        i=i2
        j=j2
        # Compute the field strength after loss
        #In db streg=streg-loss
        #phase=rnd.uniform(0,2)
        phase=totdist*2*ma.pi*f/(3*10**8)
        phasechange=np.exp(ma.pi*phase*complex(0,1))
        streg=streg*phasechange/loss
        # Find the distance to the next step
        totdist=totdist+deldist
        # Set the starting point for the next iteration
        point0=alpha*direc+point0
        if lf.length(np.array([point0,ray[0]]))>=maxraylength:
          if abs(lf.length(np.array([point0,ray[0]]))-maxraylength)>0.5*space:
            deldist=lf.length(np.array([point0-alpha*direc,ray[1]]))
            point0=ray[1]
            if abs(deldist)<=5.96e-08:
              return np.array([streg,totdist])
          else:
            return np.array([streg,totdist])
        # Return the strength ready for the next ray
    return np.array([streg,totdist])
  def singleray(s,ray,iterconsts,freq,nrays):
    ''' The field strength at the start of the ray is start assign this
    value to a mesh square and iterate through the ray '''
    streg=iterconsts[0]
    totdist=np.real(iterconsts[1])
    # Get the spacing between co-ordinates
    space=s.__getspacing__()
    # Find the position of the start of the ray
    j=int((ray[0][0]- s.__xmin__())/space)
    i=int((s.__ymax__()-ray[0][1])/space)
    # Find the direction of the ray
    direc=lf.Direction(ray)
    normal=np.array([direc[1],-direc[0]])
    # Find the maximum i and j
    # FIXME find index for the end of the ray.
    jmax=int((direc[0]*space+ray[1][0]- s.__xmin__())/space)
    imax=int((s.__ymax__()-ray[1][1]-direc[1]*space)/space)
    maxraylength=lf.length(np.array([ray[0],ray[1]]))
    point0=ray[0] # Start of the ray
    # Compute the loss using that the length of each step should be the same.
    tmp=np.array([[0,space],[space,0]])
    alpha=(1/(lin.norm(direc)**2))*0.5*(np.absolute(np.dot(tmp,direc)[0])+np.absolute(np.dot(tmp,direc)[1]))
    deldist=lf.length(np.array([(0,0),alpha*direc]))
    # Compute  the number of steps from one end of the ray to the other
    # If ray1=ray0+lam*d then n=lam/space
    n=int(1+lin.norm(ray[1]-ray[0])/deldist)
    if (deldist>np.sqrt(2*(space**2))):
        print('length greater than mesh')
    # Step through the ray decreasing the field strength
    for x in range(0,2*n):
      #if (x>n): print(x,n)
      # Find the number of squares for the cone
      #Nsqs=((totdist*ma.pi)/(nrays*space)*sqrt(2))
      #Nsqs=int(2*totdist*ma.sin(2*ma.pi/nrays)/(space))
      #conedist=totdist*ma.tan(ma.pi/nrays)
      si=ma.sin(2*ma.pi/nrays)
      c=ma.cos(2*ma.pi/nrays)
      conedist=totdist*(2*si*c+si+np.sqrt((c**2+1)*(c**2-0.5)))/(2*c)
      Nsqs=int(1+(conedist/deldist))
      # Find the matrix position of the next point
      i2=int((s.__ymax__()-point0[1])/space)
      j2=int((point0[0]- s.__xmin__())/space)
      # Check the point is in the grid
      if i2>s.__greatesti__() or j2>s.__greatestj__():
        return np.array([streg,totdist])
      elif i2<0 or j2<0:
        return np.array([streg,totdist])
      else :
        if abs(i-i2)>abs(i-imax) or abs(j-j2)>abs(j-jmax):
          return np.array([streg,totdist])
        #if abs(i-i2)==abs(i-imax) and abs(j-j2)==abs(j-jmax):
         # return np.array([streg,totdist])
        loss=(totdist+deldist)/totdist
        if i2 == i and j2 == j and x>0:
          # If a ray is going through the same box again take away the loss
          # but do not add the start again.
          # For db s.grid[i2][j2]-=loss
          s.grid[i2][j2]=s.grid[i2][j2]/loss
        else:
          # If a ray does not go through the same box twice add the signal
          #strength to the box
          s.grid[i2][j2]+=streg

        # Assign the value out to the squares in the cone
        for count in range(1,Nsqs+2):
          point1=count*alpha*normal+point0
          i3=int((s.__ymax__()-point1[1])/space)
          j3=int((point1[0]- s.__xmin__())/space)
          if 0<=i3<=s.__greatesti__() and 0<=j3<=s.__greatestj__():
            #if abs(i2-i3)<abs(i2-imax) and abs(j2-j3)<abs(j2-jmax):
              s.grid[i3][j3]+=streg
          point1=-count*alpha*normal+point0
          i3=int((s.__ymax__()-point1[1])/space)
          j3=int((point1[0]- s.__xmin__())/space)
          if 0<=i3<=s.__greatesti__() and 0<=j3<=s.__greatestj__():
            #if abs(i2-i3)<abs(i2-imax) and abs(j2-j3)<abs(j2-jmax):
              s.grid[i3][j3]+=streg
        i=i2
        j=j2
        # Compute the field strength after loss
        #In db streg=streg-loss
        streg=streg/loss
        # Find the distance to the next step
        totdist=totdist+deldist
        # Set the starting point for the next iteration
        point0=alpha*direc+point0
        if lf.length(np.array([point0,ray[0]]))>=maxraylength:
          if abs(lf.length(np.array([point0,ray[0]]))-maxraylength)>0.5*space:
            deldist=lf.length(np.array([point0-alpha*direc,ray[1]]))
            point0=ray[1]
            if abs(deldist)<=5.96e-08:
              return np.array([streg,totdist])
          else:
            return np.array([streg,totdist])
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
    #diffz=diffz/np.absolute(z1)
    #z1=10*np.log10(np.absolute(s.grid))
    ##print(r.grid)
    #z2=10*np.log10(np.absolute(r.grid))
    np.seterr(divide='ignore')
    mp.imshow(diffz, cmap='viridis', interpolation='nearest',vmin=0,vmax=5) #,extent=extent)
    mp.colorbar()
    return diffz
  def plot(s):
     ''' Plot a heatmap of the strength values '''
     # Remove the points inside the objects from the data set.
     np.seterr(divide='ignore')
     N=s.objectcorners.shape[0]
     z=np.absolute(s.grid)
     for l in range(0,N):
       complist=s.completeinsidelist(l)
       N2=complist.shape[0]
       for k in range(0,N2+1):
         i=complist[k][0]
         j=complist[k][1]
         z[i][j]=empty
     #Convert to db
     z=10*np.ma.log10(z)
     extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
     mp.imshow(z, cmap='viridis', interpolation='nearest',extent=extent,vmin=-110,vmax=30)
     #mp.colorbar()
     return

  def remove_interiors(s,room,origin,outsidepoint1,outsidepoint2):
     n=s.grid.shape[0]
     m=s.grid.shape[1]
     delta=s.__getspacing__()
     for i in range(0,n):
       y=s.__ymax__()-i*delta
       for j in range(0,m):
         x=s.__xmin__()+j*delta
         point1=np.array([x,y])
         line1=np.array([origin,point1])
         line2=np.array([outsidepoint1,point1])
         line3=np.array([outsidepoint2,point1])
         intersection1=room.room_collision_point_with_end(line1,s.__getspacing__())
         intersection2=room.room_collision_point_with_end(line2,s.__getspacing__())
         intersection3=room.room_collision_point_with_end(line3,s.__getspacing__())
         if intersection1==1 and intersection2==1:
           s.grid[i][j]=0.0
         elif intersection2==1 and intersection3==1:
           s.grid[i][j]=0.0
         elif intersection1==1 and intersection3==1:
           s.grid[i][j]=0.0
     return
  def powerplot(s,room,origin,outsidepoint1,outsidepoint2):
     ''' Plot a heatmap of the strength values '''
     np.seterr(divide='ignore')
     N=s.objectcorners.shape[0]
     #print(N)
     s.remove_interiors(room,origin,outsidepoint1,outsidepoint2)
     z=np.absolute(s.grid)
     z=(8.854187817e-12)*np.square(z)
     #Convert to db
     z=10*np.ma.log10(z)
     extent = [s.__xmin__(), s.__xmax__(), s.__ymin__(),s.__ymax__()]
     mp.imshow(z, cmap='YlOrRd', interpolation='nearest',extent=extent,vmin=-110,vmax=30)
     #mp.colorbar()
     return
  def hist(s,i):
     z=s.grid
     # Convert to Power
     z=(8.854187817e-12)*np.square(z)
     # Convert to dB
     z=z[z!=0]
     z=10*np.ma.log10(np.absolute(z))
     mp.figure(i)
     h=np.histogram(z,bins=10,range=(-100,20))
     h2=np.cumsum(h[0])
     #print(max(h2))
     h2=h2*(1.0/max(h2))
     mp.plot(h[1][:-1],h2)
     c=np.array([h[1][:-1],h2])
     mp.ylabel('Cumulative frequency')
     mp.xlabel('Power in dBm')
     mp.figure(i+1)
     mp.hist(z.flatten(),bins=20, range=(-100,20))
     mp.ylabel('# of squares with value in range')
     mp.xlabel('Power in dBm')
     #mp.plot(h[1][:-1],h[0]) Plots the histogram as a line
     return c
  #def histbounded(s,i):
     #z=np.absolute(s.grid)
     #z=10*np.ma.log10(z)
     #mp.figure(i)
     #h=np.histogram(z)
     #h2=np.cumsum(h[0])
     ##print(max(h2))
     #h2=h2*(1.0/max(h2))
     #mp.plot(h[1][:-1],h2)
     #c=np.array([h[1][:-1],h2])
     #mp.ylabel('Cumulative frequency')
     #mp.xlabel('Field strength in dBm')
     #mp.figure(i+1)
     #mp.ylabel('# of squares with value in range')
     #mp.xlabel('Field strength in dBm')
     #mp.hist(z.flatten(),bins='auto')
     ##mp.plot(h[1][:-1],h[0]) Plots the histogram as a line
     #return c
  def teststrength(s):
    ray=np.array([[0.0,0.0],[10.0,10.0]])
    start=100000
    s.singleray(ray,start)
    #print(s.strengthvalues(ray,start))
    return
  def testmesh(s):
    s.constructmesh(0.0,10.0,0.0,10.0,1.0)
    return

