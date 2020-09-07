#!/usr/bin/env python3
# Hayley Wragg 2019-29-04
''' Code to construct the ray-tracing objects rays
'''
#from scipy.sparse import lil_matrix as SM
import numpy as np
from scipy.sparse import dok_matrix as SM
import reflection as ref
import intersection as ins
import linefunctions as lf
import math as ma
import time as t
import random as rnd
from itertools import product
import sys
import DictionarySparseMatrix as DSM
import os
import Room as rom
import ParameterInput as PI
import logging
import pdb
epsilon=sys.float_info.epsilon
xcheck=1
ycheck=0
zcheck=0

dbg=0

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
  def _get_intersection_(s):
    ''' The second to last term in the np array is the starting
    co-ordinate of the travelling ray '''
    return s.ray[-2]
  def _get_direction_(s):
    ''' The direction of the travelling ray is the last term in the ray
    array. '''
    return s.ray[-1]
  def _get_travellingray_(s):
    '''The ray which is currently travelling. Should return the recent
    origin and direction. '''
    return [s.points[-2][0:3], s.points[-1][0:3]]
  def _obst_collision_point_(s,surface):
    ''' intersection of the ray with a wall_segment '''
    return ins.intersection(s._get_travellingray_(),surface)
  def _room_collision_point_(s,room):
    ''' The closest intersection out of the possible intersections with
    the wall_segments in room. Returns the intersection point and the
    wall intersected with '''
    # Find the intersection with all the walls and check which is the
    #closest. Verify that the intersection is not the current origin.
    if all(not ma.isnan(p) for p in s.points[-1]):
      # Retrieve the Maximum length from the Room
      leng2=0
      leng=room.maxleng()+epsilon
      # Initialise the intersection point, obstacle and obstacle number
      robj=room.obst[0]
      rcp=s._obst_collision_point_(robj)
      Nob=1
      # Check the initialised intersection point exists.
      if all(ma.isnan(p) for p in rcp): rNob=0
      else: rNob=Nob
      for obj in room.obst:
        cp=s._obst_collision_point_(obj)
        if any(ma.isnan(c) for c in cp):
          # There was no collision point with this obstacle
          Nob+=1
          continue
        elif all(not ma.isnan(c) for c in cp):
          leng2=s._ray_length_(cp)
          if (leng2<=leng and leng2>-epsilon) :
            leng=leng2
            rcp=cp
            robj=obj
            rNob=Nob
          Nob+=1
          continue
        else:
          raise Exception("Collision point is a mixture of None's and not None's")
      return rcp, rNob
    else:
      logging.info('The previous point was not found, ray history:'+str(s.points))
      return np.array([ma.nan,ma.nan,ma.nan]), 0
  def _ray_length_(s,inter):
    '''The length of the ray upto the intersection.
    :param inter: An array of shape (3,) with the co-ordinate of intersection point.

    * Retrieve the previous point from the second to last term in the ray points.
    * Compute the length between this point and inter.

    :retype float
    :return: length'''
    o=s.points[-2][0:3]
    ray=np.array([o,inter])
    return lf.length(ray)
  def _number_steps_(s,alpha,segleng,dist,delangle,refangle):
    '''The number of steps along the ray between intersection points.
    :meta private:

    :param alpha: float, distance stepped through each cube.
    :param segleng: float, the length of the ray segment.
    :param dist: float the total distance of the ray.
    :param delangle: The angle spacing between rays at the transmitter.
    :param refangle: The angle the ray hits the next obstacle at.

    Calculate the number of steps (ns) using :py:func:`no_steps`

    :rtype: int

    :returns: 1+ns'''
    ns=no_steps(alpha,segleng,dist,delangle,refangle)
    assert ns>(segleng/alpha)
        #errmsg=str('The number of steps along the ray is less than the number needed to reach the intersection')
        #logging.error(errmsg)
        #raise ValueError(errmsg)
    return int(1+ns)
  def _number_cones_(s,h,dist,delangle,refangle):
     '''find the number of steps taken along one normal in the cone.

     :param h:
     :param dist:
     :param delangle:
     :param refangle:'''
     nref=max(s.points.shape[0]-3,0)
     Ncon=no_cones(h,dist,delangle,refangle,nref)
     return Ncon
  def _number_cone_steps_(s,h,dist,delangle):
     '''find the number of steps taken along one normal in the cone.
     :param h: The distance through each mesh element.
     :param dist: The distance the ray has travelled so far.
     :param delangle: The angle spacing between two diagnoally neighbouring rays.

     Uses :py:func:`no_cone_steps` to get Nc
     This is then multiplied by two. This results in neighbouring cones overlapping.
     Duplicated terms are not stored but this is to help prevent errors
     from not counting early ending rays which hit room corners.

     :rtype:int
     :return:Ns'''
     Nc=2*no_cone_steps(h,dist,delangle)
     return Nc
  def _normal_mat_(s,Ncones,Nra,d,dist,h):
     ''' Form a matrix of vectors representing the plane which is \
     normal to d

     * Normalise the direction of the ray :math:`d=d/||d||`
     * Calculate angle spacing between rays :math:`deltheta=2\\arcsin(1/Ncones)`
     * Calculate the number of normals. :math:`Nnor=1+(2\\pi)/deltheta`
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
     d/=np.linalg.norm(d)                  # Normalise the direction of the ray
     anglevec=np.linspace(0.0,2*ma.pi,num=int(Ncones), endpoint=False) # Create an array of all the angles
     Norm=np.zeros((Ncones,3),dtype=np.float) # Initialise the matrix of normals
     if abs(d[2])>0:
       Norm[0]=np.array([1,1,-(d[0]+d[1])/d[2]])# This vector will lie in the plane unless d_z=0
       Norm[0]/=np.linalg.norm(Norm[0]) # Normalise the vector
       y=np.cross(Norm[0],d)            # Compute another vector in the plane for the axis.
       y/=np.linalg.norm(y)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
     else:
       Norm[0]=np.array([0,0,1])        # If d_z is 0 then this vector is always in the plane.
       y=np.cross(Norm[0],d)            # Compute another vector in the plane to form co-ordinate axis.
       y/=np.linalg.norm(y)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
     Norm=np.outer(np.cos(anglevec),Norm[0])+np.outer(np.sin(anglevec),y) # Use the outer product to multiple the axis
                                                                          # Norm[0] and y by their corresponding sin(theta),
                                                                          # and cos(theta) parts.
     return Norm
  def reflect_calc(s,room):
    ''' Finds the reflection of the ray inside a room.

    Method:

    * If: the previous collision point was *None* then don't find the \
    next one. Return: 1
    * Else: Compute the next collision point,

      * If: the collision point doesn't exist. Return: 1
      * Else:  save the collision point in the :py:class:`Ray` points. \
      Return: 0

    :rtype: 0 or 1 indicator of success.

    :returns: 0 if  reflection was computed 1 if not.

    '''
    if any(ma.isnan(c) for c in s.points[-1][0:2]):
      # If there was no previous collision point then there won't
      # be one at the next step.
      s.points=np.vstack((s.points,np.array([ma.nan,ma.nan,ma.nan,ma.nan])))
      return 1
    cp, nob=s._room_collision_point_(room)
    # Check that a collision does occur
    if any(ma.isnan(p) for p in cp):
      # If there is no collision then None's are stored as place holders
      # Replace the last point of the ray instead of keeping the direction term.
      logging.info('Collision point'+str(cp))
      logging.info('No reflection found for ray'+str(s.points))
      s.points=np.vstack((s.points[0:-1],np.array([ma.nan,ma.nan,ma.nan,ma.nan]),np.array([ma.nan,ma.nan,ma.nan,ma.nan])))
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
  def _ref_angle_(s,room,direc,nob):
    '''Find the reflection angle of the most recent intersected ray.
    This is the ray between the two intersection points not the recent intersection point and it's direction.

    :param room: :py:mod:`Room`. :py:class:`room` object which \
    contains all the obstacles in the room.
    :param direc: A numpy array of shape (3,) describing the direction the ray was travelling in.
    :param nob: The obstacle number hit.

    Use the ray number stored in s.points[-2][-1] to retrieve \
    the obstacle number then retrieve that obstacle from room.

    .. code::

       norm=edge1 x edge2

       c = (ray_direc . norm ) / (||ray_direc|| ||norm||)

       theta=arccos(c)

    :rtype: float

    :return: theta

    '''
    obst=room.obst[int(nob-1)]
    norm=np.cross(obst[1]-obst[0],obst[2]-obst[0])
    norm/=(np.linalg.norm(norm))
    check=(np.linalg.norm(direc)*np.linalg.norm(norm))
    if abs(check-0.0)<=epsilon:
      raise ValueError('direction or normal has no length')
    else:
      nleng=np.linalg.norm(norm)
      dleng=np.linalg.norm(direc)
      #cleng=np.linalg.norm(unitnorm-direc)
      frac=np.dot(direc,norm)/(nleng*dleng)
      #(dleng**2+nleng**2-cleng**2)/(2*nleng*dleng)
    theta=ma.acos(frac)
    if theta>np.pi/2:
        theta=np.pi-theta
    return theta
  def multiref(s,room,Nre):
    ''' Takes a ray and finds the first five reflections within a room.

    :param room: :py:class:`Room.room` object which \
    contains all the obstacles in the room.

    :param Nre: The number of reflections. Integer value.

    Using the function :py:func:`reflect_calc(room)` find the \
    co-ordinate of the reflected ray. Store this in s.points \
    and return whether the function was successful.

    :rtype: A numpy array of shape (3,)

    :return: end=1 if unsuccessful, 0 is successful.

    '''
    for i in range(0,Nre+1):
      end=s.reflect_calc(room)
    return
  def mesh_multiref(s,room,Nre,Mesh,Nra,nra,deltheta):
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
    dist=0.0
    # Vector of the reflection angle entries in relevant positions.
    vec=SM((Mesh.shape[0],1),dtype=np.complex128)
    for nre in range(0,Nre+1):
      end=s.reflect_calc(room)
      if abs(end)<epsilon:
          Mesh,dist,vec=s.mesh_singleray(room,Mesh,dist,vec,Nra,Nre,nra,deltheta)
      else: pass
    if not Mesh.check_nonzero_col(Nre,room.Nob):
        errmsg='There is a column in the mesh with too many terms, %d reflections completed'%Nre
        raise ValueError(errmsg)
    return Mesh
  def mesh_power_multiref(s,room,Nre,Mesh,Nra,it,Znobrat,refindex,Antpar,refcoef,deltheta):
    ''' Takes a ray and finds the first Nre reflections within a room.
    As each reflection is found the ray is stepped through and
    information added to the Mesh.
    :param room: Obstacle co-ordinates, :py:mod:`Room`. :py:class:`room`.
    :param Nre: Number of reflections, integer.
    :param Grid: a Nx*Ny*Nz array which will contain power values
    :param Nra: total number of rays.
    :param it: current ray number.
    :param Znobrat: The array with the ratio of the impedance of an \
    obstacle over the impedance of air.
    :param refindex: Array with the refractive indices of an obstacle.
    :param Antpar: array with antenna parameters - scaled wavenumber, wavelength, lengthscale.
    :param Gt: transmitter gains.

    Method:

    * Start with the initial power.
    * For each ray segment use \
    :py:func:`mesh_power_singleray(room,Mesh,dist,vec,Nra,Nre,nra)` to \
     store the power along the ray.

    :returns: grid

    '''
    # The ray distance travelled starts at 0.
    dist=0.0
    # Reflection Coefficient starts as the polarisation and the initial gain
    khat   =Antpar[0]
    L      =Antpar[2]
    for nre in range(0,Nre+1):
      end=s.reflect_calc(room)
      if abs(end)<epsilon:
          Mesh,dist,refcoef=s.mesh_power_singleray(room,Mesh,dist,refcoef,Nra,nre,Nre,it,refindex,Znobrat,khat,L,deltheta)
      else: pass
    del dist,refcoef
    return Mesh
  def mesh_singleray(s,room,Mesh,dist,calcvec,Nra,Nre,nra,deltheta):
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

      * If 0 then the :math:`calcvec[0]` term is exp(1j*pi*0.5).
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
        * Find the co-ordinate for the centre of the grid point z \
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
    ts0=t.time()
    if any(ma.isnan(c) for c in s.points[-1]):
      return Mesh,outdist,calcvec
    # --- Set initial terms before beginning storage steps -------------
    # ----- Get the parameters from the room
    Nob=room.Nob                # Total number of obstacles

    # ----- Get the parameters from the Mesh
    h=room.get_meshwidth(Mesh)  # The Meshwidth for a room with cube mesh elements
    split=Mesh.split

    # ------Get the parameters from the ray
    nre=s.points.shape[0]-3         # The reflection number of the current ray
    nob=int(s.points[-3][-1])       # The obstacle number of the last reflection

    #-------Sanity checks
    if dbg:
      assert dist<room.maxleng()*(nre+1)
      #errmsg='The distance the ray has travelled is longer than possible %f distance, %f total possible'%(dist,room.maxleng()*(nre+1))
      #logging.error(errmsg)
      #raise ValueError(errmsg)
      assert nre==calcvec.getnnz() and nre>0 and nre<Nre
      #errmsg='The reflection number is wrong, reflection number %d, number of  nonzero terms in vec %d'%(nre,calcvec.getnnz())
      #logging.error(errmsg)
      #raise ValueError(errmsg)
      assert nra<Nra and nra>0
      #errmsg='The ray number is invalid, total possible %d, this ray number %nra'%(Nra,nra)
      #raise ValueError(errmsg)
      assert nob<Nob and nob>0
      #errmsg='The obstacle number is invalid, number of obstacles %d, obstacle number %d'%(Nob,nob)
      #logging.error(errmsg)
      #raise ValueError(errmsg)

    # Compute the direction - Since the Ray has reflected but we are
    # storing previous information we want the direction of the ray which
    # hit the object not the outgoing ray.
    direc=lf.Direction(np.array([s.points[-3][0:3],s.points[-2][0:3]]))
    direc/=np.linalg.norm(direc) # Normalise direc

    col=int(Nra*nre+nra) # The column number which the ray information should be stored in.
    row=max(nre,int((nre-1)*Nob+nob)) # The row number which the newest terms should be stored at. Note nob goes from 1-12 not starting at 0.

    # Check that the Ray is travelling in a direction
    if abs(direc.any()-0.0)>epsilon:                       # Before computing the dist travelled through a mesh cube
                                                           # check the direction isn't 0.
      alpha=h/max(abs(direc))                              # Find which boundary of a unit cube gets hit first when
                                                           # direc goes through it.
    else: raise ValueError('The ray is not going anywhere')# If the direction vector is 0 nothing is happening.
    p0=s.points[-3][0:3]       # Set the initial point to the start of the segment.
    p1=p0.copy()               # p0 should remain constant and p1 is stepped. copy() is used since without p0 is changed when p1 is changed.
                               # The need for this copy() has been checked.
    i1,j1,k1=room.position(p1,h)                           # Find the indices for position p
    theta=s._ref_angle_(room,direc,nob)                    # Compute the reflection angle
    segleng=lf.length(np.vstack((p0,s.points[-2][0:3]))) # Length of the ray segment
    # Dist will be increased as the ray is stepped through. But this will go beyond the intersection point.
    # Outdist is the distance the ray has travelled from the source to the intersection point.
    outdist=dist+segleng

    # Compute the number of steps that'll be taken along the ray to ensure the edges of the cone reach the intersection obstacle
    Ns=s._number_steps_(alpha,segleng,outdist,deltheta,theta)

    # Compute the number of cones needed to ensure less than meshwidth gaps at the end of the cone
    Ncon=s._number_cones_(alpha,outdist,deltheta,theta)
    # Compute a matrix with rows corresponding to normals for the cone.

    #-----More sanity checks
    tdb=t.time()
    if dbg:
      assert outdist>0 and outdist>dist and outdist>=segleng
        #raise ValueError('The ray distance has the wrong value')
      assert isinstance(Ns,type(1))or Ns<0
        #raise ValueError('Number of steps should be a non negative integer')
      assert isinstance(Ncon,type(1))or Ncon<0
        #raise ValueError('Number of cone steps should be a non negative integer')
    ts8=t.time()
    logging.info('Time doing initial checks and assignment in singleray %f'%(ts8-ts0))

    #----Store the first ray and ray-cone terms
    if Ncon>0:
      norm=s._normal_mat_(Ncon,Nra,direc,dist,h)             # Matrix of normals to the direc, of distance 1 equally angle spaced
    # Add the reflection angle to the vector of  ray history.
    if nre==0:                                               # Before reflection use a 1 so that the distance is still stored
      calcvec[0]=np.exp(1j*np.pi*0.5)                        # The first row corresponds to line of sight terms
    else:
      calcvec[row]=np.exp(1j*theta) # After the first reflection all reflection angles continue to be stored in calcvec.
    # Initialise the check terms
    rep=0                               # Indicates if the point is the same as the previous
    stpch=Mesh.stopcheck(i1,j1,k1)      # Check if the ray point is outside the domain.
    if stpch:
      p2=room.coordinate(h,i1,j1,k1)    # Calculate the co-ordinate of the center of the element the ray hit
      doubcheck=Mesh.doubles__inMat__(h,calcvec,Nob,(i1,j1,k1),room.Ntri) # Check if the ray has already been stored
      # Recalculate distance to be for the centre point
      distcor=centre_dist(direc,p1,p2,dist,room,col,h,nre,nob)
      if dbg:
        assert not abs(distcor-dist)>np.sqrt(3)*h
      #If the distance is 0 then the point is at the transmitter and the power is not well defined.
      if not doubcheck and abs(distcor)>epsilon:
        # Find the positions of the nonzero terms in calcvec and check the number of terms is valid.
        assert Mesh[i1,j1,k1].shape[0]==calcvec.shape[0]
        Mesh[i1,j1,k1][:,col]=distcor*calcvec
        # #---More Sanity checks
        if dbg:
          assert calcvec.getnnz()==nre+1
          #raise ValueError('incorrect vec length, the number of nonzero terms in the iteration vector should be the number of reflections plus 1')
          assert Mesh.check_nonzero_col(Nre,Nob,nre,np.array([i1,j1,k1,col]))
          #errmsg='The number of nonzero terms at(%d,%d,%d,:,%d) is incorrect'%(i1,j1,k1,col)
          #logging.error(errmsg)
          #raise ValueError(errmsg)
          assert not Mesh[i1,j1,k1][0].getnnz()>(Nre*Nob+1)
          #errormsg='The number of rays %d which have entered the element is more than possible'%Mesh[i1,j1,k1][0].getnnz()
          #logging.error(errormsg)
          #raise ValueError(errormsg)
    # After the point on the ray is stored step along the cone and store the points on there
    if Ncon>0:
      Nc=s._number_cone_steps_(alpha,dist,deltheta)           # No. of cone steps required for this ray step.
      copos2,Mesh,Cones=conestepping(col,dist,0,Nc,p1,norm,alpha,h,nra,Nra,nre,Nre,direc,Mesh,calcvec,room)
    # Compute the next point along the ray
    p1+=alpha*direc/split
    dist+=alpha*np.linalg.norm(direc)/split
    i2,j2,k2=room.position(p1,h)
    #---Iterate along the ray
    for m1 in range(1,split*(Ns+1)):  # Step through the ray
      # check that each iteration is moving into a new element
      if dbg:
        assert abs(i2-i1)<2 and abs(j2-j1)<2 and abs(k2-k1)<2
        #errmsg='The ray has stepped further than one mesh element, first position (%d,%d,%d), second position (%d,%d,%d)'%(i1,j1,k1,i2,j2,k2)
        #logging.error(errmsg)
        #raise ValueError(errmsg)
      if i2==i1 and j2==j1 and k2==k1:    # If the indices are equal pass as no new element.
        rep=1
      else:
        i1=i2                             # Reset the check indices for the next check.
        j1=j2
        k1=k2
        rep=0
        stpch=Mesh.stopcheck(i1,j1,k1)   # stopcheck finds 1 if the term is in the environment and 0 if not
      if stpch:
        p2=room.coordinate(h,i1,j1,k1)    # Calculate the co-ordinate of the center of the element the ray hit
        doubcheck=Mesh.doubles__inMat__(h,calcvec,Nob,(i1,j1,k1),room.Ntri) # Check if the ray has already been stored
        #distcor=centre_dist(direc,p1,p2,dist,room,col,h,nre,nob)
        if rep==0 and not doubcheck:
          # Recalculate distance to be for the centre point
          distcor=centre_dist(direc,p1,p2,dist,room,col,h,nre,nob)
          # If the distance is 0 then the point is at the centre and the power is not well defined.
          if abs(distcor)>epsilon:# and not Mesh.doubles__inMat__(h,calcvec,Nob,(i1,j1,k1),room.Ntri):
            # Find the positions of the nonzero terms in calcvec and check the number of terms is valid.
            Mesh[i1,j1,k1][:,col]=distcor*calcvec
            # #----More sanity checks
            if dbg:
              assert not calcvec.getnnz()!=nre+1
                #errormsg='incorrect number of terms %d in calcvec, after nre=%d, no. of terms should be nre+1'%(calcvec.getnnz(),nre)
                #logging.error(errormsg)
                #raise ValueError(errormsg)
              assert Mesh.check_nonzero_col(Nre,Nob,nre,(i1,j1,k1,col))
                #errmsg='The number of nonzero terms at (%d,%d,%d,:,%d) is incorrect'%(i1,j1,k1,col)
                #raise ValueError(errmsg)
              assert not Mesh[i1,j1,k1][0].getnnz()>(Nre*Nob+1)
                #errmsg='The number of rays which have entered the element is more than possible'
                #raise ValueError(errmsg)
      if Ncon>0:
        # In this instance stpch==0 and end of ray keep storing cone points but not on the ray.
        Nc=s._number_cone_steps_(alpha,dist,deltheta)           # No. of cone steps required for this ray step.
        copos2,Mesh,Cones=conestepping(col,dist,m1,Nc,p1,norm,alpha,h,nra,Nra,nre,Nre,direc,Mesh,calcvec,room,Cones,copos2)
      # Compute the next point along the ray
      p1+=alpha*direc/split
      dist+=alpha/split
      i2,j2,k2=room.position(p1,h)
    return Mesh,outdist,calcvec

  def mesh_power_singleray(s,room,_Grid,dist,RefCoef,Nra,nre,Nre,nra,refindex,Znobrat,khat,L,deltheta):
    ''' Iterate between two intersection points and store the ray \
    information in the Mesh

    :param room: :py:mod:`Room`. :py:class:`room` object which \
    contains the co-ordinates of the obstacles.
    :param Grid: :py:mod:`DictionarySparseMatrix`. :py:class:`DS` \
    which will store the field in the parallel and perdenicular to \
    polarisation components at each :math:`(x,y,z)` position.
    :param dist: A scalar variable which is the distance the ray \
    travelled at the start of the ray segment.
    :param RefCoef: A vector containing the product of the reflection \
    coefficients in the perpendicular and parallel directions to the \
    polarisation.
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

      * If 0 then the :math:`RefCoef` term is 1.
      * Else set :math:`RefCoef`

    * Step along the ray, For :math:`m1\\in[0,Ns):`

      * Check if the ray point is outside the domain.
      * Calculate the co-ordinate of the centre.
      * Recalculate distance to be for the centre point \
      :math:`Mesh[i1,j1,k1,:,col]=np.sqrt(np.dot((p0-p2),(p0-p2)))*calcvec`.
      * For each normal:

        * Find the next cone point :math:`p3` from the previous point \
        :math:`p1`, using the distance through a grid cube \
        :math:`\\alpha`. This is given by:math:`p3=p1+m2*\\alpha*norm`.
        * Find the co-ordinate for the centre of the grid point z \
        corresponding to the :math:`p3`'s.
        * Find the distance to this centre point.
        * Set the column :math:`nra*Nre+nre` of the mesh term at \
        these grid points to the distance times the vector of \
        reflection angles.

        .. code::

           Grid[cpos[0][j],cpos[1][j],cpos[2][j]]+=e^{ikr}(1/r2[j])*RefCoef

      * Find the co-ordinate for the next ray point. \
      :math:`p1=p1+alpha*direc`.

    :returns: Mesh, dist, calcvec

    '''
    # --- Set initial terms before beginning field calculations-------------
    #nre=len(s.points)-3         # The reflection number of the current ray
    h=room.get_meshwidth(_Grid)  # The Meshwidth for a room with Mesh spaces
    nob=int(s.points[-2][-1])       # The obstacle number of the last reflection
    Nx=_Grid.shape[0]
    Ny=_Grid.shape[1]
    Nz=_Grid.shape[2]

    # Compute the direction - Since the Ray has reflected but we are
    # storing previous information we want the direction of the ray which
    # hit the object not the outgoing ray.
    direc=lf.Direction(np.array([s.points[-3][0:3],s.points[-2][0:3]]))
    direc/=np.sqrt(np.dot(direc,direc))
    if abs(direc.any()-0.0)>epsilon:                       # Before computing the dist travelled through a mesh cube
                                                           # check the direction isn't 0.
      alpha=h/max(abs(direc))                              # Find which boundary of a unit cube gets hit first when
                                                           # direc goes through it.
    else: return _Grid, dist, RefCoef                    # If the direction vector is 0 nothing is happening.
    deldist=lf.length(np.array([(0,0,0),alpha*direc]))     # Calculate the distance travelled through a mesh cube
    p0=s.points[-3][0:3]                                   # Set the initial point to the start of the segment.
    if nre==0 and nra==0:
      p1=p0                                                  # p0 should remain constant and p1 is stepped.
    else:
      p1=p0+alpha*direc
      dist+=deldist
    i1,j1,k1=room.position(p1,h)                                                  # p0 should remain constant and p1 is stepped.
    endposition=room.position(s.points[-2][0:3],h)         # The indices of the end of the segment
    theta=s._ref_angle_(room)                                # Compute the reflection angle
    Ns=s._number_steps_(deldist)                             # Compute the number of steps that'll be taken along the ray.
    segleng=lf.length(np.vstack((s.points[-3][0:3],s.points[-2][0:3]))) # Length of the ray segment
    # Compute a matrix with rows corresponding to normals for the cone.
    Ncon=s._number_cones_(deldist,dist+segleng,Nra,deltheta)
    norm=s._normal_mat_(Ncon,Nra,direc,dist,h)                # Matrix of normals to the direc, all of distance 1 equally
                                                           # angle spaced
    Nnor=len(norm)                                         # The number of normal vectors
    # Add the reflection angle to the vector of  ray history. s.points[-2][-1] is the obstacle number of the last hit.
    if nre==0:                                             # Before reflection use a 1 so that the distance is still stored
      pass
    else:
      if Znobrat[nob]==1:
          refper=0
          refpar=0
      else:
        cthi=np.cos(theta)
        ctht=np.cos(np.arcsin(np.sin(theta)/refindex[nob]))
        refper=(Znobrat[nob]*cthi-ctht)/(Znobrat[nob]*cthi+ctht)
        refpar=(cthi-Znobrat[nob]*ctht)/(Znobrat[nob]*ctht+cthi)
      RefCoef=np.matmul(np.array([[refpar,0],[0,refper]]),RefCoef)
    for m1 in range(0,Ns+1):                             # Step through the ray
      if m1==0:
          stpch=DSM.stopcheck(i1,j1,k1,Nx,Ny,Nz) # Check if the ray point is outside the domain.
      if m1>0:                                             # After the first step check that each iteration is moving into
                                                           # a new element.
        if i2==i1 and j2==j1 and k2==k1:                   # If the indices are equal pass as no new element.
          stpch=0
        else:
          i1=i2                                           # Reset the check indices for the next check.
          j1=j2
          k1=k2
          stpch=DSM.stopcheck(i1,j1,k1,Nx,Ny,Nz)
      if stpch:
        p2=room.coordinate(h,i1,j1,k1)                     # Calculate the co-ordinate of the center
                                                           # of the element the ray hit
        # Recalculate distance to be for the centre point
        alcor=np.dot((p2-p1),direc)
        rtil=np.sqrt((dist+alcor)**2+np.dot(p2-p1-alcor*direc,p2-p1-alcor*direc))
        if rtil==0:
          _Grid[i1,j1,k1,0]=0#RefCoef[0]*DSM.FieldEquation(rtil,khat,L,lam)
          _Grid[i1,j1,k1,1]=0#RefCoef[1]*DSM.FieldEquation(rtil,khat,L,lam)
        else:
          _Grid[i1,j1,k1,0]+=RefCoef[0]*DSM.FieldEquation(rtil,khat,L,lam)
          _Grid[i1,j1,k1,1]+=RefCoef[1]*DSM.FieldEquation(rtil,khat,L,lam)
        Nc=s._number_cone_steps_(deldist,dist,Nra,deltheta)           # No. of cone steps required for this ray step.
        for m2 in range(1,Nc):
          p3=np.tile(p1,(Nnor,1))+0.25*m2*alpha*norm        # Step along all the normals from the ray point p1.
          copos=room.position(p3,h)                         # Find the indices corresponding to the cone points.
          #print("before",copos,m2,alpha,nre,nra)
          start,copos,p3,norm2=DSM.stopchecklist(copos,p3,norm,Nx,Ny,Nz) # Check if the point is valid.
          #print("after",copos)
          if start==1:
            Nnorcor=len(norm2[0])
            coords=room.coordinate(h,copos[0],copos[1],copos[2]) # Find the centre of element point for each cone normal.
            r2=centre_dist(direc,p3,coords,dist,room,col,h,nre,nob)
            n=len(copos[0])
            #FIXME try to set them all at once not one by one
            x,y,z=i1,j1,k1
            for j in range(0,n):
              if x==copos[0][j] and y==copos[1][j] and z==copos[2][j]:
                pass
              else:
                _Grid[copos[0][j],copos[1][j],copos[2][j],0]+=(1.0/(r2[j]))*np.exp(1j*khat*r2[j]*(L**2))*RefCoef[0]
                _Grid[copos[0][j],copos[1][j],copos[2][j],1]+=(1.0/(r2[j]))*np.exp(1j*khat*r2[j]*(L**2))*RefCoef[1]
                x=copos[0][j]
                y=copos[1][j]
                z=copos[2][j]
          else:
            # There are no cone positions
            pass
        # Compute the next point along the ray
      else:
          #print(i1,j1,k1)
          pass  # In this instance stpch==0 and end of ray
      p1+=0.25*alpha*direc
      dist+=0.25*deldist
      i2,j2,k2=room.position(p1,h)
      #FIXME don't stop at the end as the cone needs to be filled
      #if lf.length(np.array([p1-alpha*direc,s.points[-2][0:3]]))<h/4:
      #  break
      #print(_Grid)
    return _Grid,dist,RefCoef
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
    return

def centre_dist(direc,p1,p2,dist,room,col,h,nre,nob):
    ''' Correct the distance stored in the element to the distance the ray would travel to
    the point at the centre of the element.
    :param direc: A numpy array of shape (3,) which is the direction the ray is going in.
    :param p1: A numpy array of shape (3,) which is the co-ordinate of the current ray point.
    :param p2: A numpy array of shape (3,) which is the co-ordinate at the centre of the mesh element.\
    or a numpy array of shape (Nnor,3) which is an array of co-ordinates for the centres of mesh elements on the cones.
    :param dist: The distance the ray travelled from the transmitter to p1.

    :math:`alpha=(p2-p1).(direc)/(||direc||)'
    :math:`l1=(dist+alpha)^2`
    :math:`l2=||p2-p1-alpha*direc||^2
    :math:`alcor=\sqrt(l1^2+l2^2)

    :rtype: float or a numpy array of shape (Nnor,3).
    :returns: alcor
    '''
    direc/=np.linalg.norm(direc)
    if DSM.singletype(p2[0]):
      diff=p2-p1
      alcor=np.dot(diff,direc)
      alcor/=np.linalg.norm(direc)
      l1=(alcor*np.linalg.norm(direc)+dist)**2
      l2=np.linalg.norm(diff-alcor*direc)**2
      distcor=np.sqrt(l1+l2)
      x,y,z=room.position(p2,h)
    else:
      Nnorcor=p2.shape[0]
      p1cone=np.tile(p1,(Nnorcor,1))               # The ray point is tiled to perform arithmetic which each cone vector.
      diffvec=np.subtract(p2,p1cone)               # The vector from the ray point to the cone-point.
      direcstack=np.tile(direc,(Nnorcor,1))        # The ray direction tiled to perform arithmetic with each cone vector.
      alcorrvec=lf.coordlistdot(diffvec,direcstack)# Compute the dot between the vector between the two element points and corresponding normal.
      #alcorrvec/=np.linalg.norm(direc)
      l1vec  =lf.coordlistdot(direcstack,direcstack)
      l1vec  =np.sqrt(l1vec)*alcorrvec+dist*np.ones(Nnorcor)
      l1vec  =np.power(l1vec,2)
      l2vec  =diffvec-direcstack*alcorrvec[:,np.newaxis]
      l2vec  =lf.coordlistdot(l2vec,l2vec)
      distcor=np.sqrt(l1vec+l2vec)                 # r2 is the corrected distance.
                                                   # This is a vector with terms corresponding to the corrected distance for each cone point.

    return distcor
def singleray_test():
    '''Test the stepping through a single ray. '''
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=np.array([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')[0:3]
    Nre=int(Nre)
    # Take Tx to be 0,0,0
    Tx=     np.load('Parameters/Origin.npy')/L
    delangle      =np.load('Parameters/delangle.npy')
    ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
    Oblist        =OuterBoundary/L #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

    # Room contains all the obstacles and walls.
    Room=rom.room(Oblist)
    Nob=Room.Nob
    i=0
    directionname=str('Parameters/Directions'+str(int(i))+'.npy')
    data_matrix   =np.load(directionname)         # Matrix of ray directions
    direc=data_matrix[0]
    ry=Ray(np.append(Tx,[0]),direc)
    nra=0
    Nx=int(Room.maxxleng()/(h))
    Ny=int(Room.maxyleng()/(h))
    Nz=int(Room.maxzleng()/(h))

    Mesh=DSM.DS(Nx,Ny,Nz,Nob*Nre+1,Nra[i]*(Nre+1))
    end=ry.reflect_calc(Room)
    disttrue=np.linalg.norm(ry.points[0][0:3]-ry.points[1][0:3])
    dist=0
    calcvec=SM((Mesh.shape[0],1),dtype=np.complex128)
    Mesh,dist,calcvec=ry.mesh_singleray(Room,Mesh,dist,calcvec,Nra[i],Nre,nra,delangle[i])
    return Mesh

def power_singleray_test(Mesh):

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  i=0
  Nre=int(Nre)
  Nob            =np.load('Parameters/Nob.npy')

  G_z=np.zeros((1,1))

  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  Grid=np.zeros((Nx,Ny,Nz),dtype=float)
  PI.ObstacleCoefficients(0)
  ##----Retrieve the antenna parameters--------------------------------------
  gainname=str('Parameters/Tx'+str(Nra[i])+'Gains'+str(0)+'.npy')
  Gt            = np.load(gainname)
  freq          = np.load('Parameters/frequency'+str(0)+'.npy')
  Freespace     = np.load('Parameters/Freespace'+str(0)+'.npy')
  Pol           = np.load('Parameters/Pol'+str(0)+'.npy')

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat'+str(0)+'.npy')
  refindex     =np.load('Parameters/refindex'+str(0)+'.npy')
  # Make the refindex, impedance and gains vectors the right length to
  # match the matrices.
  Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nob*Nre+1 Repeat Nob
  Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a zero for placement in the LOS row
  refindex=np.tile(refindex,(Nre,1))
  refindex=np.insert(refindex,0,1.0+0.0j)
  Gt=np.tile(Gt,(Nre+1,1))

  # Calculate the necessry parameters for the power calculation.
  c             =Freespace[3]
  khat          =freq*L/c
  lam           =(2*np.pi*c)/freq
  Antpar        =np.array([khat,lam,L])
  ind=Mesh.nonzero()
  n=ind.shape[0]
  Grid,ind=DSM.power_compute(Mesh,Grid,Znobrat,refindex,Antpar,Gt,Pol,Nra[i],Nre,Nx)
  #for j in range(n):
 #   print('after power',ind[0,j],ind[1,j],ind[2,j])
  G_z[0]=np.count_nonzero((Grid==0))
  ub=np.amax(Grid)
  lb=np.amin(Grid)
  for j in range(0,Nz):
    mp.figure(j)
    mp.imshow(Grid[:,:,j], cmap='viridis', vmax=ub,vmin=lb)
  mp.show()
  return

def duplicatecheck(copos,copos2,p3,norm,Mesh):
  if isinstance(copos[0],(float,int,np.int64, np.complex128 )):
    copos2=copos
    return copos,p3,norm,copos2
  #----Find the iteration numbers
  Ncon=copos.shape[0]
  if len(copos.shape)==1:
    Ncon1=1
  Ncon2=copos2.shape[0]
  if len(copos2.shape)==1:
    Ncon2=1
  #----Initialise output variables
  p3out=np.array([])
  normout=np.array([])
  altcopos=np.array([])
  rep=0

  for j in range(0,Ncon):
    if Ncon==1:
      if Mesh.stopcheck(copos[0],copos[1],copos[2]):
        double=0
      else:
        break
      if Ncon2==1:
        if copos[0]==copos2[0] and copos[1]==copos2[1] and copos[2]==copos2[2]:
          double=1
        else:
          altcopos=np.vstack((altcopos,np.array([copos[0],copos[1],copos[2]])))
          p3out=np.vstack((p3out,np.array([p3[0],p3[1],p3[2]])))
          normout=np.vstack((normout,np.array([norm[0],norm[1],norm[2]])))
      else:
        for k in range(0,Ncon2):
          if copos[0]==copos2[k][0] and copos[1]==copos2[k][1] and copos[2]==copos2[k][2]:
            double=1
        if double!=1:
          altcopos=np.array([copos[0],copos[1],copos[2]])
          p3out=np.array([p3[0],p3[1],p3[2]])
          normout=np.array([norm[0],norm[1],norm[2]])
          rep=1
    else:
      if Mesh.stopcheck(copos[j][0],copos[j][1],copos[j][2]):
        double=0
      else:
        continue
      if Ncon2==1:
        if copos[j][0]!=copos2[0] or copos[j][1]!=copos2[1] or copos[j][2]!=copos2[2]:
          altcopos=np.array([copos[j][0],copos[j][1],copos[j][2]])
          p3out=np.array([p3[j][0],p3[j][1],p3[j][2]])
          normout=np.array([norm[j,0],norm[j,1],norm[j,2]])
      else:
        for k in range(0,Ncon2):
          if copos[j][0]==copos2[k][0] and copos[j][1]==copos2[k][1] and copos[j][2]==copos2[k][2]:
            double=1
        if double!=1 and rep==0:
          altcopos=np.array([copos[j][0],copos[j][1],copos[j][2]])
          p3out=np.array([p3[j][0],p3[j][1],p3[j][2]])
          normout=np.array([norm[j,0],norm[j,1],norm[j,2]])
          rep=1
        elif double!=1 and rep==1:
          altcopos=np.vstack((altcopos,np.array([copos[j][0],copos[j][1],copos[j][2]])))
          p3out=np.vstack((p3out,np.array([p3[j][0],p3[j][1],p3[j][2]])))
          normout=np.vstack((normout,np.array([norm[j,0],norm[j,1],norm[j,2]])))
  copos2=copos
  return altcopos,p3out,normout,copos2

def conestepping(col,dist,m1,Nc,p1,norm,alpha,h,nra,Nra,nre,Nre,direc,Mesh,calcvec,room,Cones=np.array([]),copos2=np.array([])):
  '''Step through the lines which form a cone on the ray and store the ray information. Check if the ray is already stored in the mesh and that the number of non-zero terms in a column is correct after storing.
  '''
  Nob=room.Nob
  Ncon=norm.shape[0]
  split=Mesh.split
  for c in calcvec.nonzero()[0]:
    if calcvec.getnnz()==nre+1:
      nob=DSM.nob_fromrow(c,Nob)
  assert alpha/split<h                                 # The step size should never be bigger than the meshwidth
   # errmsg='The step size %f along the ray should not be bigger than the mesh width %f'%(alpha/split,h)
  #  logging.error(errmsg)
   # raise ValueError(errmsg)
  for m2 in range(0,split*(Nc)):
    p3=np.tile(p1,(Ncon,1))+(m2+1)*alpha*norm/split   # Step along all the normals from the ray point p1.
    copos=room.position(p3,h)                         # Find the indices corresponding to the cone points.
    #if m2==split*Nc-1 and m1==0:
     # Cones=np.c_[p1[0],p1[1],p1[2]]
      #Cones=np.r_['0',Cones,copos]
    #elif m2==split*Nc-1:
     # Cones=np.r_['0',Cones,np.c_[p1[0],p1[1],p1[2]]]
    #Cones=np.r_['0',Cones,copos]
      # This line saves the points of the cone, only needed for testing.
      # if not os.path.exists('./Mesh'):
      #  os.makedirs('./Mesh')
      # np.save('./Mesh/SingleCone'+str(int(Nra))+'Ray'+str(int(nra))+'Refs'+str(int(Nre))+'m.npy',Cones)
    tc0=t.time()
    if m2==0 and m1==0:
      copos2=np.zeros(copos.shape)
      altcopos,p3out,normout,copos2=removedoubles(copos,p3,norm,Mesh)    # If there are any repeated cone indices in copos then these are removed.
      check,altcopos,p3out,normout=Mesh.stopchecklist(altcopos,p3,norm)  # The indices which do not correspond to positions in the environment are removed.
      copos2=altcopos
    else:
      altcopos,p3out,normout,copos =removedoubles(copos,p3,norm,Mesh)         # If there are any cone indices in list twice these are removed.
      altcopos,p3out,normout,copos2=duplicatecheck(copos,copos2,p3,norm,Mesh) # If there are any cone indices which were in the previous step these are removed.
    tc1=t.time()
    logging.info('time checking for doubles and duplicates in cone list %f'%(tc1-tc0))
    if altcopos.shape[0]>0: # Check that there are some cone positions to store.
      # Check whether there is more than one vector in the list of cones.
      if isinstance(normout[0],(float,int,np.int64, np.complex128 )):
        Nnorcor=1
        coords=room.coordinate(h,altcopos[0],altcopos[1],altcopos[2]) # Find the centre element point for each cone normal.
      else:
        Nnorcor=normout.shape[0]
        coords=room.coordinate(h,altcopos[:,0],altcopos[:,1],altcopos[:,2])
      r2=centre_dist(direc,p1,coords,dist,room,col,h,nre,nob)
      if Nnorcor==1:
        x,y,z=altcopos
        if abs(r2)>epsilon and not Mesh.doubles__inMat__(h,calcvec,Nob,(x,y,z),room.Ntri):
          errmsg1='The number of nonzero terms at '
          errmsg2='(%d,%d,%d) col %d is not %d'%(x,y,z,col,nre+1)
          errmsg=errmsg1+errmsg2
          Mesh[x,y,z][:,col]=r2*calcvec[:,0]
          assert Mesh.check_nonzero_col(Nre,Nob,nre,(altcopos[0],altcopos[1],altcopos[2],col))
            #Check that the number of terms stored
            #logging.error(errmsg)
            #raise ValueError(errmsg)
        continue
      for j in range(0,Nnorcor):
        x,y,z=altcopos[j]
        if not Mesh.doubles__inMat__(h,calcvec,Nob,(x,y,z),room.Ntri)and abs(r2[j])>epsilon:
          # If the ray is already accounted for in this mesh square then step to the next point.
          errmsg='Number of cone positions (%d,%d,%d) not the same as number of distances, %d'%(altcopos[j,0],altcopos[j,1],altcopos[j,2],r2[j])
          assert altcopos.shape[0]==r2.shape[0]
            #raise ValueError(errmsg)
          assert Mesh[x,y,z].shape[0]==calcvec.shape[0]
          Mesh[x,y,z][:,col]=r2[j]*calcvec[:,0]
          errmsg1='The number of nonzero terms at '
          errmsg2='(%d,%d,%d) col %d is not %d'%(altcopos[j][0],altcopos[j][1],altcopos[j][2],col,nre+1)
          errmsg=errmsg1+errmsg2
          assert Mesh.check_nonzero_col(Nre,Nob,nre,np.array([altcopos[j,0],altcopos[j,1],altcopos[j,2],col]))
            # Check that the correct number of terms have been stored in the mesh column
            #logging.error(errmsg)
            #raise ValueError(errmsg)
    tc2=t.time()
    logging.info('Time storing cone positions %f'%(tc2-tc1))
  return copos2,Mesh,Cones

def removedoubles(copos,p3,norm,Mesh):
  ''' This function checks whether any of the co-ordinates in the array copos are repeated.

  :param copos: array of cone position indices
  :param p3:    The array of cone points co-ordinates corresponding to copos indices.
  :param norm:  The cone directions corresponding to copos.
  :param Mesh:  The DSM which is storing the ray information.

  * Iterate through the list of cone positions copos. If another point
  in the list is the same then that point is not stored in altcopos.
  * If the point is stored in altcopos then the corresponding norm and p3 are also stored.

  :rtype: numpy array,numpy array, numpy array
  :returns: copos, p3, norm, copos2
  '''
  if isinstance(copos[0],(float,int,np.int64, np.complex128 )):
    return copos,p3,norm,copos
  #----Find the iteration lengths
  Ncon=copos.shape[0]
  #----Initialise output variables
  rep=0
  for j in range(0,Ncon):
    if Mesh.stopcheck(copos[j][0],copos[j][1],copos[j][2]):
      double=0
      for k in range(0,Ncon):
        if j!=k:
          if copos[j,0]==copos[k,0] and copos[j,1]==copos[k,1] and copos[j,2]==copos[k,2]:
            double=1
      if double==0 and rep==0:
       altcopos=copos[j]
       p3out=p3[j]
       normout=norm[j]
       rep=1
      elif double==0 and rep!=0:
       #print('infunc',altcopos)
       altcopos=np.vstack((altcopos,np.array([copos[j,0],copos[j,1],copos[j,2]])))
       p3out=np.vstack((p3out,np.array([p3[j,0],p3[j,1],p3[j,2]])))
       normout=np.vstack((normout,np.array([norm[j,0],norm[j,1],norm[j,2]])))
      else:
       if rep==0:
         altcopos=np.array([copos[j,0],copos[j,1],copos[j,2]])
         p3out=np.array([p3[j,0],p3[j,1],p3[j,2]])
         normout=np.array([norm[j,0],norm[j,1],norm[j,2]])
         rep=1
       else:
         if isinstance(altcopos[0],(float,int,np.int64, np.complex128 )):
           N=1
         else:
           N=altcopos.shape[0]
         already=0
         for l in range(0,N):
           if N==1:
             if copos[j,0]==altcopos[0] and copos[j,1]==altcopos[1] and copos[j,2]==altcopos[2]:
              already=1
           else:
             if copos[j,0]==altcopos[l,0] and copos[j,1]==altcopos[l,1] and copos[j,2]==altcopos[l,2]:
              already=1
         if already==0:
           altcopos=np.vstack((altcopos,np.array([copos[j,0],copos[j,1],copos[j,2]])))
           p3out=np.vstack((p3out,np.array([p3[j,0],p3[j,1],p3[j,2]])))
           normout=np.vstack((normout,np.array([norm[j,0],norm[j,1],norm[j,2]])))
  if rep==0:
    altcopos=copos
    p3out=p3
    normout=norm
  return altcopos,p3out,normout,copos

def removedoubletest():
    arr=np.zeros((20,3))
    p3=np.zeros((20,3))
    norm=np.zeros((20,3))
    Mesh=DSM.DS(10,10,10,1,1)
    arrout,p3out,normout,copos2=removedoubles(arr,p3,norm,Mesh)
    return


def no_cones(h,dist,delangle,refangle,nref):
     '''find the number of steps taken along one normal in the cone'''
     refangle=np.pi*0.5-refangle
     delth=angle_space(delangle,nref)
     beta=beta_leng(dist,delth,refangle)
     if beta<(h/4):
       Ncon=0
     else:
       Ncon=int(1+np.pi/np.arcsin(h/(4*beta))) # Compute the distance of the normal vector
                            # for the cone and the number of mesh points
                            # that would fit in that distance.
     return Ncon

def no_steps(alpha,segleng,dist,delangle,refangle=0.0):
  '''The number of steps along the ray between intersection points'''
  refangle=np.pi*0.5-refangle
  rhat=extra_r(dist,delangle,refangle)
  ns=(segleng+rhat)/alpha
  return int(1+ns)

def angle_space(delangle,nref=0):
  if abs(np.sqrt(2)*ma.sin(delangle/2)-1)<epsilon:
    # There are no diagonal spacings to correct for
    return delangle
  else:
    return 2*np.arcsin(np.sqrt(2)*ma.sin(delangle/2))

def beta_leng(dist,delth,refangle):
    # Nra>2 and an integer. Therefore tan(theta) exists.
    ta=np.tan(delth/2)
    rhat=extra_r(dist,delth,refangle)
    beta=(rhat+dist)*ta
    return beta

def extra_r(dist,delth,refangle=0):
    if refangle>np.pi/4:
      refangle=np.pi/2-refangle
    ta=np.tan(delth/2)
    t2=np.tan(refangle+delth/2)
    top2=0.5*ta*(ma.sin(2*refangle)+(1-np.cos(2*refangle))*t2)
    rhat=dist*top2
    return rhat
def no_cone_steps(h,dist,delangle):
     '''find the number of steps taken along one normal in the cone.
     :param h: The mesh width
     :param dist: The distance the ray has travelled so far.
     :param delangle: The angle spacing between diagonally neighbouring rays from the source.

     :math:`delth=2*\arcsin(\sqrt(2)*\sin(delangle/2))`
     :math:`t=\tan(delth/2)`
     :math:`Nc=int(1+(dist*t/h))`

     :rtype: integer
     :return: Nc'''
     delth=2*np.arcsin(np.sqrt(2)*ma.sin(delangle/2))
     t=ma.tan(delth/2)    # Nra>2 and an integer. Therefore tan(theta) exists.
     Nc=int(1+(dist*t/h)) # Compute the distance of the normal vector for
                          # the cone and the number of mesh points that would
                          # fit in that distance.
     return Nc

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  Mesh=singleray_test()
  power_singleray_test(Mesh)
  exit()
