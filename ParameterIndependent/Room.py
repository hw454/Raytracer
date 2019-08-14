#!/usr/bin/env python3
# Hayley Wragg 2017-07-18
''' Code to construct the mesh of the room '''

from math import atan2,hypot,sqrt,copysign
from math import sin,cos,atan2,log
import numpy                as np
import reflection           as ref
import intersection         as ins
import linefunctions        as lf
import HayleysPlotting      as hp
import matplotlib.pyplot    as mp
import math                 as ma
import numpy.linalg         as lin
import random               as rnd
import Rays                 as ry
import time                 as t
from itertools import product
import sys

epsilon=sys.float_info.epsilon

# class obstacle_segment:
  # ' a line segment from p0 to p1 '
  # def __init__(s,p0,p1):
    # assert not (p0==p1).all()
    # s.p=np.vstack(
      # (np.array(p0,dtype=np.float),
       # np.array(p1,dtype=np.float),
    # ))
  # def __getitem__(s,i):
    # return s.p[i]
  # def firstpoint(s):
    # return s.p[0]
  # def secondpoint(s):
    # return s.p[1]
  # def __str__(s):
    # return 'Wall_segment('+str(list(s.p))+')'

## \class room Room.py
# A room is given by a an array of obstacles. Each obst is a triangle
# formed of 3, 3D co-ordinates.
class room:
  ## Constructor
  # @param obst is a Nobx3x[3x1] array, where Nob is the number of
  # obstacles. obst[j] is a 3x[3x1] array which is 3, 3D co-ordinates
  # which form a triangle.
  # This array of triangles forms the obstacles in the room.
  # - s.obst=obst
  # - \f$ s.points[3*j]=obst[j][0] \f$
  # - s.maxlength is a 4x1 array initialised as empty. Once assigned this
  # is the maximum length in theroom and in the x, y, and z axis.
  # - s.bounds is a 3x2 array
  # \f$ s.bounds= [ [minx, miny, minz], [maxx,maxy,maxz]] \f$
  # - s.inside_points is an initial empty array. Points which are known
  # to be inside obstacles are added to this array later.
  # - s.time is an array with the time the room was created.
  # - s.meshwidth is initialised as zero but is stored once asked for
  # using get_meshwidth.
  def __init__(s,obst):
    s.obst=obst
    RoomP=obst[0]
    for j in range(1,len(obst)):
      RoomP=np.concatenate((RoomP,obst[j]),axis=0)
    s.points=RoomP
    # Points is the array of all the co-ordinates which form the surfaces in the room
    s.Nob=len(obst)
    # Nob is the number of surfaces forming obstacles in the room.
    s.maxlength=np.zeros((4,1))
    s.bounds=np.array([np.min(s.points,axis=0),np.max(s.points,axis=0)])
    s.inside_points=np.array([])
    # The inside points line within obstacles and are used to detect if a ray is inside or outside.
    s.time=np.array([t.time()])
    # The time taken for a computation is stored in time.
    s.meshwidth=0.0
  ## Get the 'ith' obstacle
  # @param i the indiex for the term being returned.
  # @return s.points[i]
  def __get_obst__(s,i):
   ''' Returns the ith surface obstacle of the room s'''
   return s.obst[i]
  ## Get the 'ith' inside point
  # @param i the indiex for the term being returned.
  # @return s.inside_points[i]
  def __get_insidepoint__(s,i):
   ''' Returns the ith inside point of s '''
   return s.inside_points[i]
  ## Get the meshwidth.
  # @param Mesh the Nx*Ny*Nz*na*nb array which will be used to store
  # reflection angles and ray distances.
  # If the meshwidth has not yet been assigned then calculate it using
  # the number of terms in the x direction (Nx) in the mesh and the
  # maximum length in the x direction (s.maxlength[1]).
  # \f$ meshwidth=s.maxlength[1]/Nx \f$
  # If the meshwidth has already been found then it is returned without recalculating.
  # @return s.points[i]
  def get_meshwidth(s,Mesh):
    if abs(s.meshwidth)<epsilon:
      return s.maxlength[1]/Mesh.Nx
    else:
      return s.meshwidth
  ## Add a new obst to the room.
  # @param obst0 is added to s.obst
  # @param obst0[0],obst0[1],obst0[2] are added to s.points.
  # @return nothing
  def __set_obst__(s,obst0):
    s.obst+=(obst0,)
    s.points+=obst0
    return
  ##  Add the point p to the list of inside points for the room.
  # @param p=[x,y,z]
  # \if inside_points was empty then assign it to be p.
  # \f$s.insidepoints=[[x,y,z]]\f$ \endif
  # \else stack the new point underneath the previous array.
  # \f$ s.insidepoints=[[x0,y0,z0],[x1,y1,z1],...,[xn,yn,zn],[x,y,z]]\f$,
  # \f$s.insidepoints[j]=[xj,yj,zj]\f$ \endelse
  # @return nothing
  def __set_insidepoint__(s,p):
    if len(s.inside_points)<1:
      s.inside_points=np.array([p])
    else:
      s.inside_points=np.vstack((s.inside_points,p))
    return
  ## The string representation of the room s is the string of the
  # obstacle co-ordinates.
  def __str__(s):
    return 'Room('+str(list(s.obst))+')'
  ## Get the maximum length in the room or axis.
  # @param a the axis or room. a=0 maximum length in room, a=1 for
  # x-axis, a=2 for y-axis a=3 for z-axis.
  # .
  # \par
  # If the maxlength[a] hasn't been found yet find it by comparing the
  # length between points in s.points.
  # @return s.maxlength[a]
  def maxleng(s,a=0):
    # Has the maxlength in the room been found yet? If no compute it.
    if abs(s.maxlength[a])<epsilon:
      leng=0
      if a==0:
        p1=s.points[-1]
        for p2 in s.points:
          leng2=lf.length(np.array([p1,p2]))
          if leng2>leng:
            s.maxlength[a]=leng2
      else:
        s.maxlength[a]=s.bounds[1][a-1]-s.bounds[0][a-1]
      return s.maxlength[a]
    else: return s.maxlength[a]
  ## maxxleng this function is no longer needed but won't be removed
  # until all programs calling it have been corrected.
  # @return s.maxleng(1)
  def maxxleng(s):
    return s.maxleng(1)
  ## maxyleng this function is no longer needed but won't be removed
  # until all programs calling it have been corrected.
  # @return s.maxleng(2)
  def maxyleng(s):
    return s.maxleng(2)
  ## maxzleng this function is no longer needed but won't be removed
  # until all programs calling it have been corrected.
  # @return s.maxlength(3)
  def maxzleng(s):
    return s.maxleng(3)
  ## Find the indexing position in a mesh with width h for point p
  #  lying in the room s.
  # @param p =[x,y,z] the co-ordinate of the point p or an array of
  # points p=[[x0,y0,z0],...,[xn,yn,zn]]
  # @param h is the meshwidth, once assigned this matches s.meshwidth
  # If p is one point, out=(p-[minx,miny,minz])//h,
  # If p is an array of points,
  # out=[(p0-[minx,miny,minz])//h,...,(pn-[minx,miny,minz])//h]
  # @return out
  def position(s,p,h):
    if isinstance(p[0],float): n=1
    elif isinstance(p[0],int): n=1
    else:
      n=len(p)
    if n==1:
      i,j,k=(p-s.bounds[0])//h
      return int(i),int(j),int(k)
    elif n>1:
      positions=np.array((p-np.tile(s.bounds[0],(n,1)))//h,dtype=int)
      return positions
    else:
      raise ValueError("Neither point nor array of points")
  ## Find the co-ordinate of the point at the centre of the element.
  # @param h the meshwdith. Once assigned this matches s.meshwidth
  # @param i the first index or an array corresponding to the first
  # index for multiple points.
  # @param j the second index or an array corresponding to the second
  # index for multiple points.
  # @param k the third index or an array corresponding to the third
  # index for multiple points.
  # .
  # \par
  # \if there is only 1 i, 1 j, and 1 k,
  # \f$ p=[minx,miny,minz] +h*[i+0.5,j+0.5,k+0.5] \f$ \endif
  # \elseif there's arrays for i,j, and k,
  # \f$ p=[[minx,miny,minz] +h*[i0+0.5,j0+0.5,k0+0.5],...,
  # [minx,miny,minz] +h*[in+0.5,jn+0.5,kn+0.5 ]\f$ \endelseif
  # @return p
  def coordinate(s,h,i,j,k):
    if isinstance(i,(float,int,np.complex128,np.int64)): n=1
    else:
      n=len(i)
    if n==1:
      coord=s.bounds[0]+h*np.array([i,j,k])+h*np.array([0.5,0.5,0.5])
      return coord
    elif n>1:
      Addarray=np.tile(s.bounds[0]+h*np.array([0.5,0.5,0.5]),(n,1))
      coord=np.array((h*np.c_[i,j,k]+Addarray),dtype=float)
      return coord
    else:
      raise ValueError("Neither point nor array of points")
 ## Traces ray's uniformly emitted from an origin around a room.
 # @param Tx the co-ordinate of the transmitter location
 # @param Nra Number of rays
 # @param Nre number of reflections
 # @param directions Nra*3 array of the initial direction for each ray.
 # @param Mesh a Nx*Ny*Nz*na*nb array (actually a dictionary of sparse
 # matrices using class DS but built to have similar structure to an array).
 # na=int(Nob*Nre+1), nb=int((Nre)*(Nra)+1)
 # .
 # \par The rays are reflected Nre times with the obstacles s.obst. The
 # points of intersection with the obstacles are stored in raylist. This
 # is done using the mesh_multiref function.
 # \par As each intersection is found the mesh_multiref function
 # forms the line segment between intersection points and the
 # corresponding ray cone. All mesh elements in the ray cone store the
 # reflection angles and the distance along the ray cone from the source
 # to the centre of each mesh element. This is stored in Mesh.
 # See function mesh_multiref for more details on the reflections and
 # storage.
 # \par When complete the time in s.time() is assigned to the time taken
 # to complete the function.
 # @return raylist, Mesh
  def ray_mesh_bounce(s,Tx,Nre,Nra,directions,Mesh):
    start_time    =t.time()         # Start the time counter
    r             =s.maxleng()
    raylist       =np.empty([Nra+1, Nre+1,4])
    directions    =r*directions
    # Iterate through the rays find the ray reflections
    # FIXME rays are independent of each other so this is parallelisable
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =np.append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      Mesh=raystart.mesh_multiref(s,Nre,Mesh,Nra,it)
      raylist[it]=raystart.points[0:-2]
    s.time=t.time()-start_time
    return raylist, Mesh
 ## ray_bounceTraces ray's uniformly emitted from an origin around a room.
 # @param Nra Number of rays
 # @param Nre number of reflections Nre
 # @param directions A Nra*3 array of the initial directions for each ray.
 # .
 # \par The multiref function is used to find the Nre reflections for
 # the Nra rays with the obstacles s.obst.
 # @return An array of the ray points.
 # \f$
 # raylist=[[p00,p01,...,p0Nre],[p10,...,p1Nre],...,[pNra0,...,pNraNre]]
 #  \f$
  def ray_bounce(s,Tx,Nre,Nra,directions):
    start_time    =t.time()         # Start the time counter
    r             =s.maxleng()
    directions    =r*directions
    raylist       =np.empty([Nra+1, Nre+1,4])
    # FIXME the rays are independent of each toher so this is easily parallelisable
    for it in range(0,Nra):
      Dir       =directions[it]
      start     =np.append(Tx,[0])
      raystart  =ry.Ray(start, Dir)
      raystart.multiref(s,Nre)
      raylist[it]=raystart.points[0:-2]
    s.time=t.time()-start_time
    return raylist
  ## Takes in n array of obsts and adds to the room
  # @param obsts=[obst0,obst1,...,obstn] the array of obstacles
  # .
  # \par The obstacles are add to s.obst and points are added to
  # s.points using add_obst.
  # \par The number of obstacles Nob is increased by n.
  # @return nothing
  def roomobstadd(s,obsts):
    for obst1 in obsts[1:]:
      s.add_obst(obst1)
      s.Nob+=1
    return

