#!/usr/bin/env python3
# Hayley Wragg 2019-29-04
''' Code to construct the ray-tracing objects rays'''
#from scipy.sparse import lil_matrix as SM
import numpy as np
from scipy.sparse import dok_matrix as SM
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
  ''' A ray is a representation of the the trajectory of a reflecting \
  line and its reflections.
  Ray.points is an array of co-ordinates representing
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
      if all(p is None for p in rcp): rNob=0
      else: rNob=Nob
      for obj in room.obst:
        cp=s.obst_collision_point(obj)
        if any(c is None for c in cp):
          # There was no collision point with this obstacle
          Nob+=1
          pass
        elif all(c is not None for c in cp):
          #if np.allclose(cp, s.points[-2][0:3],atol=epsilon):
          # #print("Collision point is the same as the previous")
          #  #pass
          #  ## Do not reassign collision point when it is the previous
          #  ## point, this shouldn't happen because of direction check though
          #else:
          #  ##print('cp accepted',cp)
          leng2=s.ray_length(cp)
          if (leng2<leng and leng2>-epsilon) :
            leng=leng2
            rcp=cp
            robj=obj
            rNob=Nob
          Nob+=1
        else:
          raise Exception("Collision point is a mixture of None's and not None's")
      return rcp, rNob
    else:
      return np.array([None, None, None]), 0
  def ray_length(s,inter):
    '''The length of the ray upto the intersection '''
    o=s.points[-2][0:3]
    ray=np.array([o,inter])
    return lf.length(ray)
  def number_steps(s,meshwidth):
    '''The number of steps along the ray between intersection points'''
    return int(lf.length(np.vstack((s.points[-3][0:3],s.points[-2][0:3])))/meshwidth)
  def number_cone_steps(s,h,dist,Nra):
     '''find the number of steps taken along one normal in the cone'''
     ta=ma.tan(ma.pi/Nra)   # Nra>2 and an integer. Therefore tan(pi/Nra) exists.
     Nc=int(1+(dist*ta)/h) # Compute the distance of the normal vector
                            # for the cone and the number of mesh points
                            # that would fit in that distance.
     return Nc
  def normal_mat(s,Ncones,Nra,d,dist,h):
     ''' Form a matrix of vectors representing the plane which is \
     normal to d

     * Normalise the direction of the ray :math:`d=d/||d||`
     * Calculate angle spacing between rays \
     :math:`deltheta=2\\arcsin(\frac{1}{Ncones})`
     * Calculate the number of normals. \
     :math:`Nnor=1+\frac{(2\\pi)}{deltheta}`
     * Create an array of all the angles.

     .. code::

        anglevec=np.linspace(0.0,2*ma.pi,num=int(Nnor), endpoint=False)

     * The dot product of the direction and the normal needs to be 0.\
     Choose :math:`(1,1,-(dx+dy)/dz)` as the first vector \
     (unless dz==0). Use this to compute another vector in the plane. \
     This forms a co-ordinate axis for the normal vectors.
     ..  code::if :math:`dz!=0`
         p=(1,1,-(dx+dy)/dz)
         N[0]=(1/||p||)*p
         y=N[0] x d
         y=(1/||y||)*y
     If dz==0 then instead choose the vector :math:`(0,0,1)` and \
     repeat the process for the additional axis vector.
     .. code::
        N[0]=(0,0,1)
        y=N[0] x d
        y=(1/||y||)*y

     * Use the ais vectors as multiples of :math:`\\cos(\\theta)` \
     :math:`\\sin(\\theta)` to form equally space vectors.

     .. code::

       N=cos(anglevec)* N[0]+sin(anglevec)*y

     :returns: N

     '''
     d=d/np.linalg.norm(d)                  # Normalise the direction of the ray
     deltheta=2*np.arcsin(1/Ncones)         # Calculate angle spacing so that
                                            # the ends of the cone are less than two meshwidth apart.
     Nnor=int(1+(2*np.pi)/deltheta)         # Calculate the number of normals.
     anglevec=np.linspace(0.0,2*ma.pi,num=int(Nnor), endpoint=False) # Create an array of all the angles
     Norm=np.zeros((Nnor,3),dtype=np.float) # Initialise the matrix of normals
     if abs(d[2]-0)>0:
       ptil=np.array([1,1,-(d[0]+d[1])/d[2]])# This vector will lie in the plane unless d_z=0
       Norm[0]=(1/np.linalg.norm(ptil))*ptil # Normalise the vector
       y=np.cross(Norm[0],d)                 # Compute another vector in the plane for the axis.
       y=(1/np.linalg.norm(y))*y             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
     else:
       Norm[0]=np.array([0,0,1])            # If d_z is 0 then this vector is always in the plane.
       y=np.cross(Norm[0],d)                # Compute another vector in the plane to form co-ordinate axis.
       y=(1/np.linalg.norm(y))*y            # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
     Norm=np.outer(np.cos(anglevec),Norm[0])+np.outer(np.sin(anglevec),y) # Use the outer product to multiple the axis
                                                                          # Norm[0] and y by their corresponding sin(theta),
                                                                          # and cos(theta) parts.
     return Norm
  def reflect_calc(s,room):
    ''' Finds the reflection of the ray inside a room.

    Method:

    * If: the previous collision point was *None* then don't find the \
    next one. Return: 1
    * Else: Compute the next collision point.

      * If: the collision point doesn't exist. Return: 1
      * Else: use the collision point to compute the reflected ray. \
      Return: 0

    :rtype: 0 or 1 indicator of success.

    :returns: 0 if  reflection was computed 1 if not.

    '''
    if any(c is None for c in s.points[-1][0:2]):
      # If there was no previous collision point then there won't
      # be one at the next step.
      s.points=np.vstack((s.points,np.array([None, None, None, None])))
      return 1
    cp, nob=s.room_collision_point(room)
    # Check that a collision does occur
    if any(p is None for p in cp):
      # If there is no collision then None's are stored as place holders
      # Replace the last point of the ray instead of keeping the direction term.
      s.points=np.vstack((s.points[0:-1],np.array([None, None, None, None]),np.array([None, None, None, None])))
      return 1
    else:
      # Construct the incoming array
      origin=s.points[-2][0:3]
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray=ref.try_reflect_ray(ray,room.obst[nob-1]) # refray is the intersection point to a reflection point
      # Update intersection point list
      s.points[-1]=np.append(cp, [nob])
      s.points=np.vstack((s.points,np.append(lf.Direction(refray),[0])))
      return 0
  def ref_angle(s,room):
    '''Find the reflection angle of the most recent intersected ray.

    :param room: :py:mod:`Room`. :py:class:`room` object which \
    contains all the obstacles in the room.

    Use the ray number stored in s.points[-2][-1] to retrieve \
    the obstacle number then retrieve that obstacle from room.

    .. code::

       norm=edge1 x edge2

       c = (ray_direc . norm ) / (||ray_direc|| ||norm||)

       theta=arccos(c)

    :rtype: float

    :return: theta

    '''
    nob=s.points[-2][-1]
    direc=s.points[-1][0:3]
    obst=room.obst[int(nob-1)]
    norm=np.cross(obst[1]-obst[0],obst[2]-obst[0])
    unitnorm=norm/(np.linalg.norm(norm))
    check=(np.linalg.norm(direc)*np.linalg.norm(unitnorm))
    if abs(check-0.0)<=epsilon:
      raise ValueError('direction or normal has no length')
    else:
      nleng=np.linalg.norm(unitnorm)
      dleng=np.linalg.norm(direc)
      cleng=np.linalg.norm(unitnorm-direc)
      frac=(dleng**2+nleng**2-cleng**2)/(2*nleng*dleng)
    theta=ma.acos(frac)
    return theta
  def multiref(s,room,Nre):
    ''' Takes a ray and finds the first five reflections within a room.

    :param room: :py:class:`Room.room` object which \
    contains all the obstacles in the room.

    :param Nre: The number of reflections. Integer value.

    Using the function :py:func:`reflect_calc(room)` find the \
    co-ordinate of the reflected ray. Store this in s.points \
    and return whether the function was successful.

    :rtype: 3x1 numpy array.

    :return: end=1 if unsuccessful, 0 is successful.

    '''
    for i in range(0,Nre+1):
      end=s.reflect_calc(room)
    return
  def mesh_multiref(s,room,Nre,Mesh,Nra,nra):
    ''' Takes a ray and finds the first Nre reflections within a room.
    As each reflection is found the ray is stepped through and
    information added to the Mesh.
    :param room: Obstacle co-ordinates, :py:mod:`Room`. :py:class:`room`.
    :param Nre: Number of reflections, integer.
    :param Mesh: A grid with corresponding sparse matrices, this \
    is a :py:mod:`DictionarySparseMatrix`. :py:class:`DS` object.
    :param Nra: Total number of rays, integer.

    Method:

    * Create a temporary vector vec.
    * For each ray segment use \
    :py:func:`mesh_singleray(room,Mesh,dist,vec,Nra,Nre,nra)` to \
     segment storing r*calcvec in the Mesh. With r being the distance \
     ray travelled to get  centre of the grid point the ray has gone \
     through.

    :returns: Mesh

    '''
    # The ray distance travelled starts at 0.
    dist=0
    # Vector of the reflection angle entries in relevant positions.
    vec=SM((Mesh.shape[0],1),dtype=np.complex128)
    for nre in range(0,Nre+1):
      end=s.reflect_calc(room)
      if abs(end)<epsilon:
          Mesh,dist,vec=s.mesh_singleray(room,Mesh,dist,vec,Nra,Nre,nra)
      else: pass
    del dist,vec
    return Mesh
  def mesh_singleray(s,room,_Mesh,_dist,_calcvec,Nra,Nre,nra):
    ''' Iterate between two intersection points and store the ray \
    information in the Mesh

    :param room: :py:mod:`Room`. :py:class:`room` object which \
    contains the co-ordinates of the obstacles.
    :param Mesh: :py:mod:`DictionarySparseMatrix`. :py:class:`DS` \
    which will store all of the ray information.
    :param dist: A scalar variable which is the distance the ray \
    travelled at the start of the ray segment.
    :param calcvec: A vector containing :math:`e^{i\theta}` terms \
    for reflection angles :math:`\theta`. These terms are stored \
    in row nre*Nob+nob with nre being the current reflection number, \
    Nob the maximum obstacle number and nob the number of the \
    obstacle which was hit with the corresponding angle.
    :param Nra: Total number of rays.
    :param Nre: Maximum number of reflections.
    :param nra: Current ray number.

    Method:

    * Calculate :math:`\\theta` the reflection angle using \
    :func:`reflect_angle(room)`.
    * Find the number of steps :math:`Ns` to the end of the ray segment \
    using :py:func:`number_steps(meshwidth)`.
    * Compute an array of normal vectors representing the ray cone.
    * Check the reflection number:
      * If 0 then the :math:`calcvec[0]` term is 1.
      * ElseIf 1 then set :math:`calcvec[0]=0` and \
      :math:`calcvec[nre*Nob+nob]=e^{i \\theta}`.
      * Else set :math:`calcvec[nre*Nob+nob]=e^{i \\theta}`.

    * Step along the ray, For :math:`m1\\in[0,Ns):`
      * Check if the ray point is outside the domain.
      * Calculate the co-ordinate of the centre.
      * Recalculate distance to be for the centre point \
      :math:`Mesh[i1,j1,k1,:,col]=np.sqrt(np.dot((p0-p2),(p0-p2)))*calcvec`.
      * For each normal:
        * Find the next cone point :math:`p3` from the previous point \
        :math:`p1`, using the distance through a grid cube \
        :math:`\\alpha`. This is given by:math:`p3=p1+m2*\\alpha*norm`.
        * Find the co-ordinate for the centre of the grid point z
        corresponding to the :math:`p3`'s.
        * Find the distance to this centre point.
        * Set the column :math:`nra*Nre+nre` of the mesh term at \
        these grid points to the distance times the vector of \
        reflection angles.

        .. code::

           Mesh[cpos[0][j],cpos[1][j],cpos[2][j],:,col]=r2[j]*calcvec

      * Find the co-ordinate for the next ray point. \
      :math:`p1=p1+alpha*direc`.

    :returns: Mesh, dist, calcvec

    '''
    # --- Set initial terms before beginning storage steps -------------
    nre=len(s.points)-3         # The reflection number of the current ray
    h=room.get_meshwidth(_Mesh)  # The Meshwidth for a room with Mesh spaces
    nob=s.points[-2][-1]        # The obstacle number of the last reflection

    # Compute the direction - Since the Ray has reflected but we are
    # storing previous information we want the direction of the ray which
    # hit the object not the outgoing ray.
    direc=lf.Direction(np.array([s.points[-3][0:3],s.points[-2][0:3]]))
    col=int(Nra*nre+nra)
    if abs(direc.any()-0.0)>epsilon:                       # Before computing the dist travelled through a mesh cube
                                                           # check the direction isn't 0.
      alpha=h/max(abs(direc))                              # Find which boundary of a unit cube gets hit first when
                                                           # direc goes through it.
    else: return _Mesh, _dist, _calcvec                    # If the direction vector is 0 nothing is happening.

    deldist=lf.length(np.array([(0,0,0),alpha*direc]))     # Calculate the distance travelled through a mesh cube
    p0=s.points[-3][0:3]                                   # Set the initial point to the start of the segment.
    p1=p0                                                  # p0 should remain constant and p1 is stepped.
    i1,j1,k1=room.position(p0,h)                           # Find the indices for position p0
    endposition=room.position(s.points[-2][0:3],h)         # The indices of the end of the segment
    theta=s.ref_angle(room)                                # Compute the reflection angle
    Ns=s.number_steps(deldist)                             # Compute the number of steps that'll be taken along the ray.
    segleng=lf.length(np.vstack((s.points[-3][0:3],s.points[-2][0:3]))) # Length of the ray segment
    # Compute a matrix with rows corresponding to normals for the cone.
    Nc=s.number_cone_steps(deldist,_dist+segleng,Nra)
    norm=s.normal_mat(Nc,Nra,direc,_dist,h)                 # Matrix of normals to the direc, all of distance 1 equally
                                                           # angle spaced
    Nnor=len(norm)                                         # The number of normal vectors
    # Add the reflection angle to the vector of  ray history. s.points[-2][-1] is the obstacle number of the last hit.
    if nre==0:                                             # Before reflection use a 1 so that the distance is still stored
      _calcvec[0]=1                                        # The first row corresponds to line of sight terms
    elif nre==1:                                           # The first reflection needs to reset and start storing
                                                           # reflection coefficients.
      _calcvec[0]=0                                        # Reset the first term which was only for line of sight
      _calcvec[int((nre-1)*room.Nob+nob)]=np.exp(1j*theta) # Use a complex exponential to store the reflection
                                                           # angle. This will allow us to multiply by the distance and
                                                           # store both pieces of information in the same place.
    else:
      _calcvec[int((nre-1)*room.Nob+nob)]=np.exp(1j*theta) # After the first reflection all reflection angles
                                 # continue to be stored in calcvec.
    for m1 in range(0,Ns):                             # Step through the ray
      stpch=_Mesh.stopcheck(i1,j1,k1,endposition,h)         # Check if the ray point is outside the domain.
      if m1>0:                                             # After the first step check that each iteration is moving into
                                                           # a new element.
        if i2==i1 and j2==j1 and k2==k1:                   # If the indices are equal pass as no new element.
          pass
        else:
          i1=i2                                            # Reset the check indices for the next check.
          j1=j2
          k1=k2
      if stpch:
        p2=room.coordinate(h,i1,j1,k1)                     # Calculate the co-ordinate of the center
                                                           # of the element the ray hit
        # Recalculate distance to be for the centre point
        _Mesh[i1,j1,k1,:,col]=np.sqrt(np.dot((p0-p2),(p0-p2)))*_calcvec
        #Nc=s.number_cone_steps(deldist,dist,Nra)           # No. of cone steps required for this ray step.
        for m2 in range(Nnor):
          p3=np.tile(p1,(Nnor,1))+m2*alpha*norm             # Step along all the normals from the ray point p1.
          copos=room.position(p3,h)                         # Find the indices corresponding to the cone points.
          start,copos,p3,norm2=_Mesh.stopchecklist(copos,endposition,h,p3,norm) # Check if the point is valid.
          if start==1:
            coords=room.coordinate(h,copos[0],copos[1],copos[2]) # Find the centre of
                                                           # element point for each cone normal.
            p3=np.transpose(p3)                            # Change orientation for stacking co-ordinates.
            norm2=np.transpose(norm2)
            elementdiff=coords-p3                          # Find the distance from the original cone point to the cone
                                                           # element centre.
            dist2 =lf.coordlistdistance(elementdiff) # The distance between the centre of the element and the point the cone entered the element
            normlengths=lf.coordlistdistance(norm2)   # Each norm should have length one but compute them just in case
            normdot=lf.coordlistdot(elementdiff,norm2)# Compute the dot between the vector between the two element points and corresponding normal.
            r1=_dist+np.sqrt((np.square(dist2)+np.divide(np.square(normdot),normlengths)))
            n=len(copos[0])
            Mult=np.zeros((n,3),dtype=np.float)
            Mult=np.outer(r1,direc)
            #for j in range(0,n):
            #  Mult[j]=r1[j]*direc
            alpha2=lf.coordlistdot((Mult+np.tile(p0,(n,1))-p3),norm2)
            r2=np.divide(r1,(np.cos(np.arctan(np.divide(alpha2,r1)))))
            #FIXME try to set them all at once not one by one
            for j in range(0,n):
              _Mesh[copos[0][j],copos[1][j],copos[2][j],:,col]=r2[j]*_calcvec
          else:
            # There are no cone positions
            pass
        # Compute the next point along the ray
      else: break                                         # In this instance stpch==0 and end of ray
      #del Nc
      p1=p0+m1*alpha*direc
      _dist+=deldist
      i2,j2,k2=room.position(p1,h)
      #FIXME don't stop at the end as the cone needs to be filled
      if lf.length(np.array([p1,s.points[-2][0:3]]))<h:
        break
    return _Mesh,_dist,_calcvec
  def raytest(s,room,err):
    ''' Checks the reflection function for errors using the test \
    functions in :py:mod:`reflection`.

    '''
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
