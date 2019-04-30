#!/usr/bin/env python3
# Hayley Wragg 2019-29-04
''' Code to construct the ray-tracing objects rays'''
from scipy.sparse import lil_matrix as SM
import numpy as np
import reflection as ref
import intersection as ins
import linefunctions as lf
#import HayleysPlotting as hp
import matplotlib.pyplot as mp
import math as ma
#from math import sin,cos,atan2,hypot,sqrt,copysign
#import roommesh as rmes
import time as t
import random as rnd
from itertools import product
import sys

epsilon=sys.float_info.epsilon

class Ray:
  ''' A ray is a representation of the the trajectory of a reflecting line
  and it's reflections. Ray.points is an array of co-ordinates representing
  the collision points with the last term being the direction the ray ended in.
  And Ray.reflections is an array containing tuples of the angles of incidence
  and the number referring to the position of the obstacle in the obstacle list
  '''
  def __init__(s,origin,direc):
    s.points=np.vstack(
      (np.array(origin,  dtype=np.float),
       np.array(direc,   dtype=np.float)
    ))
    #s.reflections=np.vstack()
  def __str__(s):
    return 'Ray(\n'+str(s.points[1])+')'
  def _get_intersection(s):
    ''' The second to last term in the np array is the starting
    co-ordinate of the travelling ray '''
    return s.ray[-2]
  def _get_direction(s):
    ''' The direction of the travelling ray is the last term in the ray
    array. '''
    return s.ray[-1]
  def _get_travellingray(s):
    '''The ray which is currently travelling. Should return the recent
    origin and direction. '''
    return [s.points[-2][0:3], s.points[-1][0:3]]
  def obst_collision_point(s,surface):
    ''' intersection of the ray with a wall_segment '''
    return ins.intersection(s._get_travellingray(),surface)
  def room_collision_point(s,room):
    ''' The closest intersection out of the possible intersections with
    the wall_segments in room. Returns the intersection point and the
    wall intersected with '''
    if all(p is not None for p in s.points[-1]):
      # Retreive the Maximum length from the Room
      leng=room.maxleng()
      # Initialise the point and wall
      robj=room.obst[0]
      rcp=s.obst_collision_point(robj)
      # Find the intersection with all the walls and check which is the
      #closest. Verify that the intersection is not the current origin.
      Nob=1
      rNob=Nob
      for obj in room.obst:
        cp=s.obst_collision_point(obj)
        if all(c is not None for c in cp):
          if np.allclose(cp, s.points[-2][0:3],atol=epsilon):
            #print("Collision point is the same as the previous")
            pass
            # Do not reassign collision point when it is the previous
            # point, this shouldn't happen because of direction check though
          else:
            #print('cp accepted',cp)
            leng2=s.ray_length(cp)
            if (leng2<leng and leng2>-epsilon) :
              leng=leng2
              rcp=cp
              robj=obj
              rNob=Nob
          Nob+=1
        if any(c is None for c in rcp):
          #print('No collision point found', rcp, cp)
          pass
      return rcp, robj, rNob
    else:
      return np.array([None, None, None]), None, 0
  def ray_length(s,inter):
    '''The length of the ray upto the intersection '''
    o=s.points[-2][0:3]
    ray=np.array([o,inter])
    return lf.length(ray)
  def number_steps(s,meshwidth):
    return int(lf.length(np.vstack((s.points[-3][0:3],s.points[-2][0:3])))/meshwidth)
  def number_cone_steps(s,h,dist,Nra):
     '''find the number of steps taken along one normal in the cone'''
     si=ma.sin(2*ma.pi/Nra)
     c=ma.cos(2*ma.pi/Nra)
     conedist=dist*(2*si*c+si+np.sqrt((c**2+1)*(c**2-0.5)))/(2*c)
     Nc=int(1+(conedist/h))
     return Nc
  def normal_mat(s,Nra,d,dist,h):
     deltheta=(-2+np.sqrt(2.0*(Nra)))*(np.pi/(Nra-2)) # Calculate angle spacing
     Nup=int(dist*ma.sin(deltheta)/h)+1
     anglevec=np.linspace(0.0,2*np.pi,num=int(Nup), endpoint=False)
     Norm=np.zeros((Nra,3),dtype=np.float)
     check=0
     for j in range(0,Nup):
       Norm[j]=np.cross(d,np.array([np.cos(anglevec[j]),np.sin(anglevec[j]),0]))
     return Norm
  def reflect_calc(s,room):
    ''' finds the reflection of the ray inside a room'''
    if any(c is None for c in s.points[-1][0:2]):
      # If there was no previous collision point then there won't
      # be one at the next step.
      #print('no cp at previous stage', s.points[-2:-1])
      s.points=np.vstack((s.points,np.array([None, None, None, None])))
      return 0
    cp,obst,nob=s.room_collision_point(room)
    # Check that a collision does occur
    if any(p is None for p in cp):
      #print('no cp at collision ',len(s.points)-2,' before reflection.')
      #print('Collision point ',cp,' Previous instersection ', s.points[-2:-1])
      # If there is no collision then None's are stored as place holders
      # Replace the last point of the ray instead of keeping the direction term.
      s.points=np.vstack((s.points[0:-1],np.array([None, None, None, None]),np.array([None, None, None, None])))
      return 0
    elif obst is None:
      #print('no ob',s.points[1])
      # Replace the last point of the ray instead of keeping the direction term.
      s.points=np.vstack((s.points[0:-1],np.array([None, None, None, None]),np.array([None, None, None, None])))
      return 0
      #print('ray:',s.points)
      #raise Error('Collision should occur')
    else:
      # Construct the incoming array
      origin=s.points[-2][0:3]
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray=ref.try_reflect_ray(ray,obst) # refray is the intersection point to a reflection point
      # Update intersection point list
      s.points[-1]=np.append(cp, [nob])
      s.points=np.vstack((s.points,np.append(lf.Direction(refray),[0])))
    return 1
  def ref_angle(s,room):
    '''Find the reflection angle of the most recent intersected ray.'''
    nob=s.points[-2][-1]
    direc=s.points[-1][0:3]
    obst=room.obst[int(nob)]
    norm=np.cross(obst[1]-obst[0],obst[2]-obst[0])
    check=(np.inner(direc,norm)/(np.linalg.norm(direc)*np.linalg.norm(norm)))
    theta=ma.acos(check)
    return theta
  def multiref(s,room,m):
    ''' Takes a ray and finds the first five reflections within a room'''
    for i in range(0,m+1):
      end=s.reflect_calc(room)
    return
  def mesh_multiref(s,room,Nre,Mesh,Nra,nra):
    ''' Takes a ray and finds the first Nre reflections within a room.
    As each reflection is found the ray is stepped through and
    information added to the Mesh.
    Inputs: Room- Obstacle co-ordinates, Nre- Number of reflections,
    Mesh- 3D array with each element corresponding to a sparse matrix.
    Nra- Total number of rays.'''
    # The ray distance travelled starts at 0.
    dist=0
    # Vector of the reflection angle entries in relevant positions.
    calcvec=SM((Mesh.shape[0],1),dtype=np.complex128)
    for nre in range(0,Nre+1):
      end=s.reflect_calc(room)
      if end:
          Mesh,dist,calcvec=s.mesh_singleray(room,Mesh,dist,calcvec,Nra,Nre,nra)
      else: pass
    return Mesh
  def mesh_singleray(s,room,Mesh,dist,calcvec,Nra,Nre,nra):
    ''' Iterate between two intersection points and store the ray information in the Mesh '''
    # --- Set initial terms before beginning storage steps -------------
    nre=len(s.points)-2     # The reflection number of the current ray
    h=room.get_meshwidth(Mesh)  # The Meshwidth for a room with Mesh spaces
    # Compute the direction - Since the Ray has reflected but we are
    # storing previous information we want the direction of the ray which
    # hit the object not the outgoing ray.
    direc=lf.Direction(np.array([s.points[-3][0:3],s.points[-2][0:3]]))
    # Before computing the dist travelled through a mesh cube check the direction isn't 0.
    if abs(direc.any()-0.0)>epsilon:
      alpha=h/max(abs(direc))
    else: return Mesh, dist, calcvec
    # Calculate the distance travelled through a mesh cube
    deldist=lf.length(np.array([(0,0,0),alpha*direc]))
    # Find the indexing position of the start of the ray segment and end.
    p0=s.points[-3][0:3]
    i1,j1,k1=room.position(p0,h)    # Start
    endposition=room.position(s.points[-2][0:3],h) # Stopping terms
    # Compute the reflection angle
    theta=s.ref_angle(room)
    # Compute the number of steps that'll be taken along the ray.
    Ns=s.number_steps(deldist)
    # Compute the normals for the cone.
    norm=s.normal_mat(Nra,direc,dist,h) # Matrix of normals to the direc, all of distance 1 equally angle spaced
    Nup=len(norm)                       # The number of normal vectors
    # Add the reflection angle to the vector of  ray history. s.points[-2][-1] is the obstacle number of the last hit.
    calcvec[int(nre*room.Nob+s.points[-2][-1])]=np.exp(1j*theta)
    for m1 in range(Ns):
      stpch=Mesh.stopcheck(i1,j1,k1,endposition,h)
      if m1>0:
        if i2==i1 and j2==j1 and k2==k1:
          pass
        else:
          i1=i2
          j1=j2
          k1=k2
      if stpch:
        Mesh[i1,j1,k1,:,nra*Nre+m1-1]=dist*calcvec
        Nc=s.number_cone_steps(deldist,dist,Nra)
        for m2 in range(Nc):
          p3=np.tile(p0,(Nup,1))+m2*alpha*norm
          conepositions=room.position(p3,h)
          Mesh[conepositions[:][0],conepositions[:][1],conepositions[:][2],:,nra*Nre+m1-1]=dist*calcvec
          p=dist*calcvec
        # Compute the next point along the ray
        p0=p0+alpha*direc
        dist=dist+deldist
        i2,j2,k2=room.position(p0,h)
        #FIXME check if the position is the same as the previous
        if lf.length(np.array([p0,s.points[-2][0:3]]))<h:
          break
    return Mesh,dist,calcvec
  def raytest(s,room,err):
    ''' Checks the reflection for errors'''
    cp,wall=s.room_collision_point(room)
    # Check that a collision does occur
    if cp[0] is None: return
    else:
      # Construct the incoming array
      origin=s._get_origin()
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray,n=ref.try_reflect_ray(ray,wall.p)
      err=ref.errorcheck(err,ray,refray,n)
      # update self...
      s.ray[-1]=cp
      s.ray=np.vstack((s.ray,lf.Direction(refray)))
      #print('ray',ray, 'refray', refray, 'error', err)
    return err
# --- Archive reflect function
# def reflect(s,room):
    # ''' finds the reflection of the ray inside a room'''
    # if any(c is None for c in s.points[-1][0:2]):
      # # If there was no previous collision point then there won't
      # # be one at the next step.
      # #print('no cp at previous stage', s.points[-2:-1])
      # s.points=np.vstack((s.points,np.array([None, None, None, None])))
      # return 0
    # cp,obst,nob=s.room_collision_point(room)
    # # Check that a collision does occur
    # if any(p is None for p in cp):
      # #print('no cp at collision ',len(s.points)-2,' before reflection.')
      # #print('Collision point ',cp,' Previous instersection ', s.points[-2:-1])
      # # If there is no collision then None's are stored as place holders
      # # Replace the last point of the ray instead of keeping the direction term.
      # s.points=np.vstack((s.points[0:-1],np.array([None, None, None, None]),np.array([None, None, None, None])))
      # return 0
    # elif obst is None:
      # #print('no ob',s.points[1])
      # # Replace the last point of the ray instead of keeping the direction term.
      # s.points=np.vstack((s.points[0:-1],np.array([None, None, None, None]),np.array([None, None, None, None])))
      # return 0
      # #print('ray:',s.points)
      # #raise Error('Collision should occur')
    # else:
      # # Construct the incoming array
      # origin=s.points[-2][0:3]
      # ray=np.array([origin,cp])
      # # The reflection function returns a line segment
      # refray=ref.try_reflect_ray(ray,obst) # refray is the intersection point to a reflection point
      # # update self...
      # s.points[-1]=np.append(cp, [nob])
      # s.points=np.vstack((s.points,np.append(lf.Direction(refray),[0])))
    # return 1


