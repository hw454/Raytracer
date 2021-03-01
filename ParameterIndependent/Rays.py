#!/usr/bin/env python3
# Hayley Wragg 2019-29-04
''' Code to construct the ray-tracing objects rays
'''
#from scipy.sparse import lil_matrix as SM
import numpy as np
from numpy.linalg import norm as leng
from numpy import sqrt,arcsin,arccos,cos,sin,tan,pi,outer,cross,exp,load,vstack,angle,arange
from numpy import linspace, zeros, ones, empty, append
from numpy import array as numparr
from scipy.sparse import dok_matrix as SM
import reflection as ref
import intersection as ins
import linefunctions as lf
import math as ma
from math import isnan
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

dbg=0
if dbg:
  logon=1
else:
  logon=load('Parameters/logon.npy')
xcheck=2
ycheck=9
zcheck=5

class Ray:
  ''' A ray is a representation of the the trajectory of a reflecting \
  line and its reflections.
  Ray.points is an array of co-ordinates representing
  the collision points with the last term being the direction the ray ended in.
  And Ray.reflections is an array containing tuples of the angles of incidence
  and the number referring to the position of the obstacle in the obstacle list

  '''
  def __init__(s,origin,direc):
    s.points=vstack(
      (numparr(origin,  dtype=np.float),
       numparr(direc,   dtype=np.float)
    ))
    s.maxleng=0.0
    s.angspace=0.0
    s.meshwidth=0.0
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
  def _obst_collision_point_(s,room,nob):
    ''' intersection of the ray with a wall_segment '''
    return ins.intersection(s._get_travellingray_(),room,nob)
  def _room_collision_point_(s,room):
    ''' The closest intersection out of the possible intersections with
    the wall_segments in room. Returns the intersection point and the
    wall intersected with '''
    # Find the intersection with all the walls and check which is the
    #closest. Verify that the intersection is not the current origin.
    if all(not isnan(p) for p in s.points[-1]):
      # Retrieve the Maximum length from the Room
      leng2=0
      leng=room.maxleng()+epsilon
      # Initialise the intersection point, obstacle and obstacle number
      robj=room.obst[0]
      nob=1
      # Check the initialised intersection point exists.
      nsur=room.surf_from_ob(nob)
      if nsur!=s.points[-2,3]:
        rcp=s._obst_collision_point_(room,nob)
      else:
        rcp=numparr([ma.nan,ma.nan,ma.nan])
      if all(isnan(p) for p in rcp):
        rNob=0
        rNsur=-1
        intercount=0
      else:
        rNob=nob
        rNsur=nsur
        intercount=1
      for nob in range(2,room.Nob+1):
        obj=room.obst[nob-1]
        nsur=room.surf_from_ob(nob)
        if intercount<room.MaxInter and nsur!=s.points[-2,3] and nsur!=rNsur:
          cp=s._obst_collision_point_(room,nob)
          #logging.info('Intercount %d'%intercount)
          #logging.info('Ob number %d, Surface number %d'%(nob,nsur))
          #logging.info('Collision point')
          #logging.info(cp)
          if all(not isnan(c) for c in cp):
            intercount+=1
            leng2=s._ray_length_(cp)
            if -epsilon<leng2<=leng:
              leng=leng2
              rcp=cp
              robj=obj
              rNob=nob
              rNsur=nsur
            continue
        #else:
          #logging.info('Intercount %d'%intercount)
          #logging.info('Ob number %d, Surface number %d'%(nob,nsur))
          #logging.info('No ref calc condition')
      if dbg:
        if all(not isnan(c) for c in cp):
          if not ins.InsideCheck(rcp,room.obst[rNob-1]):
            print(rcp)
            print(room.obst[rNob-1])
            print(rNob)
          assert(ins.InsideCheck(rcp,room.obst[rNob-1]))
      return rcp, rNsur
    else:
      #logging.info('The previous point was not found'+str(s.points))
      return numparr([ma.nan,ma.nan,ma.nan]), 0
  def _ray_length_(s,inter):
    '''The length of the ray upto the intersection.
    :param inter: An array of shape (3,) with the co-ordinate of intersection point.

    * Retrieve the previous point from the second to last term in the ray points.
    * Compute the length between this point and inter.

    :retype float
    :return: length'''
    o=s.points[-2][0:3]
    return leng(o-inter)
  def _number_steps_(s,alpha,segleng,dist,delangle,refangle,maxsteps):
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
    refangle=pi*0.5-refangle
    rhat=s.extra_r(dist,delangle,maxsteps,refangle)
    ns=(segleng+rhat)/alpha
    if ns<(segleng/alpha):
        errmsg='The number of steps along the ray is less than the number needed to reach the intersection'
        raise ValueError(errmsg)
    return min(int(1+ns),2*maxsteps)
  def _number_cones_(s,h,dist,delangle,refangle,maxsteps=0):
     '''find the number of normals needed to form the cone.

     :param h: meshwidth
     :param delangle: The angle spacing between rays
     :param dist: The distance the ray has travelled
     :param refangle: The angle the ray reflected with.
     :param maxsteps: The maximum number of steps a ray can have.

     Ncon=Number of steps taken along a cone to get to the edge or the boundary of the environment.

     :math:delth=:py:func:'angle_space'(delangle,nref)
     :math:\beta=:py:func:'beta_leng'(dist,delth,refangle)
     :math:Ncon=1+\\frac{\pi}{\\arcsin(\\frac{h}{4\beta})}

     :rtype: integer
     :returns: Ncon'''
     nref=max(s.points.shape[0]-3,0)
     refangle=pi*0.5-refangle
     delth=s.angle_space(delangle)
     beta=s.beta_leng(dist,delth,refangle,maxsteps)
     if beta<(h/4):
       Ncon=0
     else:
       Ncon=int(1+pi/arcsin(h/(4*beta))) # Compute the distance of the normal vector
                            # for the cone and the number of mesh points
                            # that would fit in that distance.
     return Ncon
  def _number_cone_steps_(s,h,dist,delangle,maxsteps=0):
     '''find the number of steps taken along one normal in the cone.
     :param h: The mesh width
     :param dist: The distance the ray has travelled so far.
     :param delangle: The angle spacing between diagonally neighbouring rays from the source.

     :math:`delth=2*\arcsin(\sqrt(2)*\sin(delangle/2))`
     :math:`t=\tan(delth/2)`
     :math:`Nc=int(1+(dist*t/h))`

     :rtype: integer
     :return: Nc'''
     delth=2.0*arcsin(sqrt(2.0)*sin(delangle*0.5))
     t=tan(delth*0.5)    # Nra>2 and an integer. Therefore tan(theta) exists.
     Nc=int(1.0+(dist*t/h)) # Compute the distance of the normal vector for
                            # the cone and the number of mesh points that would
                            # fit in that distance.
     return min(3*Nc,maxsteps)
  def ray_maxsteps(s,room):
    ''' Return the maximum number of steps a ray can have. If the term hasn't be set yet find it
    using the maximum length in the room and the mesh width. Once the term is found it doesn't change so it only needs to be returned.
    '''
    if abs(s.maxleng)<epsilon:
      s.maxsteps=int(1+room.maxleng()//room.get_meshwidth(s))
    return s.maxsteps

  def extra_r(s,dist,delth,maxsteps,refangle=0.0):
    ''' When a ray intersects a boundary it's cone may not have intersected yet.
    This function finds the distance the ray needs to continue to cover this space.
    This is bounded by the maximum width the cone can have in the environment.

    :param dist: Distance the ray has travelled so far.
    :param delth: Angle spacing between rays.
    :param refangle: The angle to the normal the ray hit the obstacle with.

    :math: r=\\frac{\\tan(\\frac{delth}{2})*\\tan(refangle+\\frac{delth}{2})*(\\sin(2*refangle)+(1.0-\\cos(2*refangle)))}{2}

    :rtype: float
    :returns: max(r,maxleng/h)
    '''
    if refangle>pi*0.25:
      refangle=0.5*pi-refangle
    ta=tan(delth*0.5)
    t2=tan(refangle+delth*0.5)
    top2=0.5*ta*(sin(2.0*refangle)+(1.0-cos(2.0*refangle))*t2)
    return min(dist*top2,maxsteps)
  def centre_dist(s,p1,p2,dist,room):
    ''' Correct the distance stored in the element to the distance the ray would travel to
    the point at the centre of the element.
    :param s: Current ray
    :param p1: A numpy array of shape (3,) which is the co-ordinate of the current ray point.
    :param p2: A numpy array of shape (3,) which is the co-ordinate at the centre of the mesh element.\
    or a numpy array of shape (Nnor,3) which is an array of co-ordinates for the centres of mesh elements on the cones.
    :param dist: The distance the ray travelled from the transmitter to p1.
    :param room: Object type room from :py:mod:'Room'
    :param nre: Current reflection number
    :param col: column number in mesh
    :param nob: obstacle number

    :math:`alpha=(p2-p1).(direc)/(||direc||)'
    :math:`l1=(dist+alpha)^2`
    :math:`l2=||p2-p1-alpha*direc||^2
    :math:`alcor=\sqrt(l1^2+l2^2)

    :rtype: float or a numpy array of shape (Nnor,3).
    :returns: alcor
    '''
    direc=s.points[-1][0:3]
    h=room.get_meshwidth(s)
    if DSM.singletype(p2[0]):
      diff=p2-p1
      alcor=diff@direc
      alcor/=direc@direc
      l1=(alcor*leng(direc)+dist)**2
      l2=leng(diff-alcor*direc)**2
      distcor=sqrt(l1+l2)
    else:
      alcorrvec=numparr([direc@(p-p1) for p in p2])
      l1vec  =numparr([(leng(direc)*al +dist)**2 for al in alcorrvec])
      l2vec  =numparr([((p-p1)-direc*(direc@(p-p1)))@((p-p1)-direc*(direc@(p-p1))) for p in p2])
      distcor=numparr([sqrt(l1+l2) for l1,l2 in zip(l1vec,l2vec)])               # r2 is the corrected distance.
                                                   # This is a vector with terms corresponding to the corrected distance for each cone point.
    return distcor
  def angle_correct(s,dist,direc,n,x,p0):
    ''' When the ray information is stored the point which the storage corresponds to is the
    centre of an element which may be different from the point on the ray. The angle which this
    new ray hits an obstacle is different to the angle being stored. This function finds the angle
    corresponding to a a ray which hits the point x.
    :param dist:
    :param direc:
    :param n:
    :param x:
    :param p0:
    '''
    Txhat=p0-dist*direc/leng(direc)
    #print(Txhat)
    if leng(n)==0 or leng(x-Txhat)==0:
      pdb.set_trace()
    phi=arccos(((Txhat-x)@n)/(leng(n)*leng(Txhat-x)))
    #if dbg:
    if not 0<=phi<=pi*0.5:
      phi=pi-phi
      #pdb.set_trace()
    #if abs(phi-pi/2)<epsilon:
    #  pdb.set_trace()
    return phi
  def angle_space(s,delangle):
    ''' Using the horizontal anglespacing between rays find the diagonal angle spacing between rays.
    '''
    if abs(s.angspace)<epsilon:
      si=sin(delangle*0.5)
      if abs(sqrt(2.0)*si-1.0)<epsilon:
        # There are no diagonal spacings to correct for
        s.angspace=delangle
      else:
        s.angspace=2.0*arcsin(sqrt(2.0)*si)
    return s.angspace
  def beta_leng(s,dist,delth,refangle,maxsteps=0):
    # Nra>2 and an integer. Therefore tan(theta) exists.
    ta=tan(0.5*delth)
    rhat=s.extra_r(dist,delth,maxsteps,refangle)
    return (rhat+dist)*ta
  def conestepping(s,iterterms,rayterms,programterms,p1,roomnorm,norm,direc,Mesh,calcvec,room,copos2=numparr([])):
    '''Step through the lines which form a cone on the ray and store the ray information. Check if the ray is already stored in the
    mesh and that the number of non-zero terms in a column is correct after storing.
    :param rayterms:     nra-ray number,nre-reflection number,Nc-Number of steps to take along the cone,col-column number,alpha-distance stepped through a voxel.
    :param iterterms:    olddist-distance ray travelled to previous intersection, dist-distance ray travelled to the p1.
    :param programterms: Nra- total number of rays,Nre- total number of reflections.
    :param p1:           The point on the ray the cones step away from. A numpy array with shape (1,3).
    :param roomnorm:     Norm vector to most recently intersected surface.
    :param norm:         Array with shape (Ncon,3) with each row corresponding to a vector normal to the ray.
    :param direc:        The direction the ray is travelling in.
    :param Mesh:         DSM containing the ray tracer information so far.
    :param calcvec:      Vector containing the information about this ray which will be stored.
    :param room:         room object which obstacle co-ordiantes, norms and number of triangles per surface.

    Steps through the points on the cone vectors and check if the points have previousl been stored at. Removes the doubles then
    stores calcvec at the points. Continued until the end of the cone is reached.'''
    # Retrieve the program parameters from the objects containing them
    Nc,col=rayterms[2:4].astype(int)       # number of steps along the cone vectors to take, column in matrix to store calcvec
    alpha=rayterms[4]                      # Distance stepped through a voxel at each step
    dist=iterterms[1]                      # Distance ray travelled up to intersection and to current ray point.
    m1=iterterms[2].astype(int)            # Step along the ray
    anglechange=programterms[2].astype(int)# Total number of rays and reflections, angle correction switch and steps through each voxel.
    h=room.get_meshwidth(Mesh)                 # Voxel width
    cal=calcvec[:,0].toarray()
    rind=cal.nonzero()[0]
    row=rind[-1]                           # Row number corresponding to recent entry in calcvec
    conesplit=2
    conesplitinv=0.5
    # Iterate along the normal vectors which form the cones. At each step check for double counting and duplicate points.
    ste=alpha*conesplitinv
    coords=zeros((0,3))
    for m2 in linspace(ste,ste*(conesplit*(Nc)+1),num=conesplit*Nc):
      p3=numparr([p1+m2*n for n in norm])                 # Step along all the normals from the ray point p1.
      copos=room.position(p3,h)                         # Find the indices corresponding to the cone points.
      if abs(m2-ste)<epsilon and m1==0:
        altcopos,p3out,normout,copos2=removedoubles(copos,p3,norm,Mesh)     # If there are any repeated cone indices in copos then these are removed.
      else:
        altcopos,p3out,normout,copos =removedoubles(copos,p3,norm,Mesh)             # If there are any cone indices in list twice these are removed.
        altcopos,p3out,normout,copos2=duplicatecheck(altcopos,copos2,p3out,normout) # If there are any cone indices which were in the previous step these are removed.
      if altcopos.shape[0]>1 and len(altcopos.shape)>1: # Check that there are some cone positions to store.
        # Check whether there is more than one vector in the list of cones.
        coords=room.coordinate(h,altcopos[:,0],altcopos[:,1],altcopos[:,2])
      elif altcopos.shape[0]==1 and len(altcopos.shape)>1:
        # There is only one position to store so slicing is not used.
        coords=room.coordinate(h,altcopos[0,0],altcopos[0,1],altcopos[0,2])
      else: continue  # There are no new positions to store so step again
      # Correct the distance travelled to the distance corresponding to the co-ordinate at the centre of all the voxels which will be stored at.
      r2=s.centre_dist(p1,coords,dist,room)
      for j,p in enumerate(altcopos):
        x,y,z=p
        if dbg:
            if x==xcheck and y==ycheck and z==zcheck:
              logging.info(str(Mesh.doubles__inMat__(calcvec,(x,y,z),rind))+str(room.check_innerpoint(coords[j])))
              logging.info('calcrows'+str(rind))
        if not Mesh.doubles__inMat__(cal,(x,y,z),rind) and  not room.check_innerpoint(coords[j]) and abs(r2[j])>epsilon:
          for r in rind:
            if r==row and row>0 and anglechange:
              phi=s.angle_correct(dist,direc,roomnorm,coords[j],p1)  # If angle correction is turned on then correct the reflection angle to be for the point being stored at
              Mesh[x,y,z][row,col]=r2[j]*exp(1j*phi)            # Overwrite the previously stored angle with the corrected term.
            else:
             Mesh[x,y,z][r,col]=r2[j]*cal[r,0] # Store the ray information in the corresponding column in the voxel
    return copos2,Mesh

  def _normal_mat_(s,Ncones,d):
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
     d/=leng(d)                  # Normalise the direction of the ray
     anglevec=linspace(0.0,2*pi,num=int(Ncones), endpoint=False) # Create an array of all the angles
     Norm=zeros((Ncones,3),dtype=np.float) # Initialise the matrix of normals
     if abs(d[2])>0:
       Norm[0]=numparr([1,1,-(d[0]+d[1])/d[2]])# This vector will lie in the plane unless d_z=0
       Norm[0]/=leng(Norm[0]) # Normalise the vector
       y=cross(Norm[0],d)            # Compute another vector in the plane for the axis.
       y/=leng(y)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
     else:
       Norm[0]=numparr([0,0,1])        # If d_z is 0 then this vector is always in the plane.
       y=cross(Norm[0],d)            # Compute another vector in the plane to form co-ordinate axis.
       y/=leng(y)             # Normalise y. y and Norm[0] are now the co-ordinate axis in the plane.
     Norm=numparr([cos(a)*Norm[0]+sin(a)*y for a in anglevec])
     #outer(cos(anglevec),Norm[0])+outer(sin(anglevec),y) # Use the outer product to multiple the axis
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
    if any(isnan(c) for c in s.points[-1][0:2]):
      # If there was no previous collision point then there won't
      # be one at the next step.
      s.points=vstack((s.points,numparr([ma.nan,ma.nan,ma.nan,ma.nan])))
      return 1
    cp, nsur=s._room_collision_point_(room)
    # Check that a collision does occur
    if any(isnan(p) for p in cp):
      # If there is no collision then None's are stored as place holders
      # Replace the last point of the ray instead of keeping the direction term.
      if logon:
        logging.info('Collision point'+str(cp))
        logging.info('No reflection found for ray'+str(s.points))
      s.points=vstack((s.points[0:-1],numparr([ma.nan,ma.nan,ma.nan,ma.nan]),numparr([ma.nan,ma.nan,ma.nan,ma.nan])))
      return 1
    else:
      # Construct the incoming array
      #logging.info('Collision point'+str(cp))

      origin=s.points[-2][0:3]
      ray=numparr([origin,cp])
      # The reflection function returns a line segment
      nob=room.nob_from_sur(nsur) # Since the normal vector is the same for different triangles in the same plane taking the lowest obstacle number is enough.
      if dbg:
        if nsur==s.points[-2][3]:
          logging.error('Surface number same as previous')
          logging.info('Sur number %d'%nob)
          logging.info('Prev sur %d'%s.points[-2][3])
          logging.info(s.points)
        assert nsur!=s.points[-2][3]
      refray=ref.room_reflect_ray(ray,room,nob-1) # refray is the intersection point to a reflection point
      # Update intersection point list
      s.points[-1]=np.append(cp, [nsur])
      s.points=vstack((s.points,np.append(lf.Direction(refray),[0])))
      return 0
  def refcoefonray(s,room,Znob,refindex):
    ''' Compute the reflection coefficient from the last intersection.
    Retrieve the obstacle number from the points list. Then retrieve the corresponding impedance and refractive index.
    output the reflection coefficient and the reflection angle.
    :param room: The room class object containing the obstacle co-ordinates, the normal vectors and number of triangles per surface.
    :param Znob: The impedance of each surface.
    :param refindex: The refractive index of each surface.
    :rtype: numpy array with shape (2,1) and float
    :returns: Refcoef, theta
    '''
    nob=int(s.points[-2,3])
    nsur=room.nsurffromnob(nob)
    Z=Znob[nsur+1]
    r=1.0/refindex[nsur+1]
    direc=s.points[1][0:3]
    theta=s._ref_angle_(room,direc,nob)
    cthi=cos(theta)
    ctht=cos(arcsin(sin(theta)*r))
    ctht=sqrt(1-(sin(theta)*r)**2)
    S1=cthi*Z
    S2=ctht*Z
    refcoef=ones((2,1))
    if abs(cthi)<epsilon and abs(ctht)<epsilon:
      # Ref coef is very close to 1
      if abs(Z)>epsilon:
        frac=(Z-1)/(Z+1)
        refcoef[0]=frac
        refcoef[1]=frac
      else:
        refcoef[0]=(S1-ctht)/(S1+ctht)
        refcoef[1]=(cthi-S2)/(cthi+S2)
    return refcoef,theta
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
    nob=int(nob)
    obst=room.obst[nob-1]
    norm=room.norms[nob-1]
    nleng=leng(norm)
    dleng=leng(direc)
    check=(nleng*dleng)
    if abs(check)<=epsilon:
      raise ValueError('direction or normal has no length')
    else:
      checkinv=1.0/check
      #cleng=np.linalg.norm(unitnorm-direc)
      frac=direc@norm*checkinv
      #(dleng**2+nleng**2-cleng**2)/(2*nleng*dleng)
    theta=arccos(frac)
    if theta>pi*0.5:
        theta=pi-theta
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
  def mesh_multiref(s,room,Mesh,nra,programterms):
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
    Nre=int(programterms[1])
    for nre in range(0,Nre+1):
      end=s.reflect_calc(room)
      if not end:
        Mesh,dist,vec=s.mesh_singleray(room,Mesh,dist,vec,programterms,nra)
      else:
        nanmat=empty((Nre-nre,4))
        nanmat[:]=np.nan
        s.points=vstack((s.points,nanmat))
        return Mesh
      if dbg:
        if not Mesh.check_nonzero_col(Nre,room.Nsur):
          errmsg='There is a column in the mesh with too many terms, %d reflections completed'%Nre
          raise ValueError(errmsg)
    return Mesh
  def mesh_power_multiref(s,room,Mesh,it,Znobrat,refindex,Antpar,Pol,programterms,loghandle=str()):
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
    :param Pol: polarisation of the antenna
    :param deltheta: anglespacing between rays
    :param loghandle: handle for logging errors and runtime information

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
    refcoef=1.0*Pol
    Nre=int(programterms[1])
    for nre in range(0,Nre+1):
      end=s.reflect_calc(room)
      if any(isnan(c) for c in s.points[-1]):
        nanmat=empty((Nre-nre,4))
        nanmat[:]=np.nan
        s.points=vstack((s.points,nanmat))
        return Mesh
      refcoefterm,theta=s.refcoefonray(room,Znobrat,refindex)
      refcoef[0]*=refcoefterm[0]
      refcoef[1]*=refcoefterm[1]
      if abs(end)<epsilon:
        rayterms=numparr([nre,it,theta])
        Mesh,dist,refcoef=s.mesh_power_singleray(room,Mesh,dist,refcoef,rayterms,Antpar,programterms,loghandle)
      else:
        nanmat=zeros((Nre-nre,4))
        nanmat[:]=np.nan
        s.points=vstack((s.points,nanmat))
        return Mesh
    return Mesh
  def mesh_singleray(s,room,Mesh,dist,calcvec,programterms,nra):
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
    in row nre*Nsur+nsur with nre being the current reflection number, \
    Nsur the maximum surface number and nsur the number of the \
    surface which was hit with the corresponding angle.
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
      * Else set :math:`calcvec[nre*Nsur+nsur]=e^{i \\theta}`.

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
    if any(isnan(c) for c in s.points[-1]):
      return Mesh,outdist,calcvec
    # --- Set initial terms before beginning storage steps -------------
    Nra,Nre,anglechange,split,_,_=programterms.astype(int)
    splitinv,deltheta=programterms[-2:]
    # ----- Get the parameters from the room
    Nsur=room.Nsur                # Total number of obstacles

    # ----- Get the parameters from the Mesh
    h=room.get_meshwidth(Mesh)  # The Meshwidth for a room with cube mesh elements

    # ------Get the parameters from the ray
    nre=s.points.shape[0]-3         # The reflection number of the current ray
    nsur=int(s.points[-3][-1])       # The obstacle number of the last reflection
    nob=room.nob_from_sur(nsur)

    #-------Sanity checks
    if dbg:
        if dist>room.maxleng()*(nre+1):
          errmsg='The distance the ray has travelled is longer than possible'
          raise ValueError(errmsg)
        if nre>Nre:
          errmsg='The reflection number is greater than the total number of reflections'
          raise ValueError(errmsg)
        elif nre<0:
          errmsg='reflection number can not be negative'
        elif nre!=calcvec.getnnz():
          errmsg='The reflection number is wrong'
          raise ValueError(errmsg)
        if nra>Nra:
          errmsg='The ray number is greater than the total number of rays'
          raise ValueError(errmsg)
        elif nra<0:
          errmsg='ray number can not be negative'
        if nsur>Nsur:
          errmsg='The obstacle number is greater than the total number of obstacles'
          raise ValueError(errmsg)
        elif nob<0:
          errmsg='obstacle number can not be negative'

    # Compute the direction - Since the Ray has reflected but we are
    # storing previous information we want the direction of the ray which

    direc=lf.Direction(numparr([s.points[-3][0:3],s.points[-2][0:3]]))
    direc/=leng(direc) # Normalise direc

    col=int(Nra*nre+nra) # The column number which the ray information should be stored in.
    row=max(nre,int((nre-1)*Nsur+nsur)) # The row number which the newest terms should be stored at. Note nob goes from 1-12 not starting at 0.

    # Check that the Ray is travelling in a direction
    alpha=h/max(abs(direc))
    if dbg:
      if not abs(direc.any())>epsilon:                  # Before computing the dist travelled through a mesh cube
                                                       # check the direction isn't 0.
        raise ValueError('The ray is not going anywhere')# If the direction vector is 0 nothing is happening.
    p1=s.points[-3][0:3].copy()               # p0 should remain constant and p1 is stepped. copy() is used since without p0 is changed when p1 is changed.
                               # The need for this copy() has been checked.
    i1,j1,k1=room.position(p1,h)                           # Find the indices for position p
    theta=s._ref_angle_(room,direc,nob)                    # Compute the reflection angle
    segleng=leng(s.points[-3][0:3]-s.points[-2][0:3])           # Length of the ray segment
    # Dist will be increased as the ray is stepped through. But this will go beyond the intersection point.
    # Outdist is the distance the ray has travelled from the source to the intersection point.
    olddist=dist
    outdist=dist+segleng

    # The normal which defines the plane the obstacle hit was in.
    #Tri=room.obst[nob-1]
    roomnorm=room.norms[nob-1]

    # Compute the number of steps that'll be taken along the ray to ensure the edges of the cone reach the intersection obstacle
    Ms=s.ray_maxsteps(room)
    Ns=s._number_steps_(alpha,segleng,outdist,deltheta,theta,Ms)

    # Compute the number of cones needed to ensure less than meshwidth gaps at the end of the cone
    Ncon=s._number_cones_(alpha,outdist,deltheta,theta,Ms)

    # Compute the number of cones steps needed to reach the end of the cone.
    Nc=s._number_cone_steps_(h,dist,deltheta,Ms)

    # Compute a matrix with rows corresponding to normals for the cone.
    rayterms=numparr([nra,nre,Nc,col,alpha])
    iterterms=numparr([olddist,dist,0])
    #-----More sanity checks
    if dbg:
      if outdist<0 or outdist<dist or outdist<segleng:
        raise ValueError('The ray distance has the wrong value')
      if not isinstance(Ns,type(1))or Ns<0:
        raise ValueError('Number of steps should be a non negative integer')
      if not isinstance(Ncon,type(1))or Ncon<0:
        raise ValueError('Number of cone steps should be a non negative integer')

    #----Store the first ray and ray-cone terms
    if Ncon>0:
      norm=s._normal_mat_(Ncon,direc)             # Matrix of normals to the direc, of distance 1 equally angle spaced
    # Add the reflection angle to the vector of  ray history.
    if nre==0:                                               # Before reflection use a 1 so that the distance is still stored
      calcvec[0]=np.exp(1j*pi*0.5)                        # The first row corresponds to line of sight terms
    else:
      calcvec[row]=np.exp(1j*theta) # After the first reflection all reflection angles continue to be stored in calcvec.
    # Initialise the check terms
    cal=calcvec.toarray()
    rind=calcvec.nonzero()[0]
    rep=0                               # Indicates if the point is the same as the previous
    stpch=Mesh.stopcheck(i1,j1,k1)      # Check if the ray point is outside the domain.
    if stpch:
      p2           =room.coordinate(h,i1,j1,k1)                                # Calculate the co-ordinate of the center of the element the ray hit
      doubcheck    =Mesh.doubles__inMat__(calcvec,(i1,j1,k1),rind) # Check if the ray has already been stored
      #interiorcheck=room.check_innerpoint(p2)                                  # Check if the point is inside obstacles
      # Recalculate distance to be for the centre point
      distcor=s.centre_dist(p1,p2,olddist,room)
      if abs(distcor-dist)>sqrt(3)*h:
        raise ValueError('The corrected distance on the ray is more than a mesh width from the ray point distance')
      # If the distance is 0 then the point is at the centre and the power is not well defined.
      if not doubcheck  and abs(distcor)>epsilon:#and not interiorcheck
        # Find the positions of the nonzero terms in calcvec and check the number of terms is valid.
        #calind=calcvec.nonzero()
        #assert Mesh[i1,j1,k1].shape[0]==calcvec.shape[0]
        for r in rind:
          if r==row and row>0 and anglechange:
            phi=s.angle_correct(dist,direc,roomnorm,p2,p1)
            Mesh[i1,j1,k1][row,col]=distcor*np.exp(1j*phi)
          else:
            Mesh[i1,j1,k1][r,col]=distcor*cal[r,0]
        #---More Sanity checks
        if dbg:
          if calcvec.getnnz()!=nre+1:
            raise ValueError('incorrect vec length, the number of nonzero terms in the iteration vector should be the number of reflections plus 1')
          errmsg='The number of nonzero terms at (%d,%d,%d,%d) is incorrect'%(i1,j1,k1,col)
          if not Mesh.check_nonzero_col(Nre,Nsur,nre,numparr([i1,j1,k1,col])):
            raise ValueError(errmsg)
          if Mesh[i1,j1,k1][0].getnnz()>(Nsur*((Nsur-1)**Nre-1)/(Nsur-2)+1):
            logging.info('Too many rays in mesh element')
            logging.info(Mesh[i1,j1,k1])
            raise ValueError('The number of rays which have entered the element is %d more than possible'%Mesh[i1,j1,k1][0].getnnz())
    # After the point on the ray is stored step along the cone and store the points on there
    if Ncon>0:
      copos2,Mesh=s.conestepping(iterterms,rayterms,programterms,p1,roomnorm,norm,direc,Mesh,calcvec,room)
    # Compute the next point along the ray
    p1  +=alpha*direc*splitinv
    dist+=alpha*leng(direc)*splitinv
    i2,j2,k2=room.position(p1,h)
    #---Iterate along the ray
    for m1 in range(1,split*(Ns+1)):  # Step through the ray
      # check that each iteration is moving into a new element
      iterterms[2]=m1
      iterterms[1]=dist
      if dbg:
        if abs(i2-i1)>1 or abs(j2-j1)>1 or abs(k2-k1)>1:
          errmsg='The ray has stepped further than one mesh element'
          raise ValueError(errmsg)
      if i2!=i1 or j2!=j1 or k2!=k1:    # If the indices are equal pass as no new element.
        i1=i2                             # Reset the check indices for the next check.
        j1=j2
        k1=k2
        stpch=Mesh.stopcheck(i1,j1,k1)   # stopcheck finds 1 if the term is in the environment and 0 if not
        if stpch:
          p2=room.coordinate(h,i1,j1,k1) # Calculate the co-ordinate of the center of the element the ray hit
          doubcheck=Mesh.doubles__inMat__(calcvec,(i1,j1,k1),rind) # Check if the ray has already been stored
          #interiorcheck=room.check_innerpoint(p2) # Check the point is not inside an obstacle
          distcor=s.centre_dist(p1,p2,dist,room) # Correct the distance travelled to the distance for the point as the centre of the element.
          if dbg:
            if i1==xcheck and j1==ycheck and k1==zcheck:
              logging.info('Double status'+str(Mesh.doubles__inMat__(calcvec,(i1,j1,k1),rind)))
              logging.info('Meshwidth %f, Obstacle number %d'%(h,Nsur))
              logging.info('Calcvec'+str(calcvec))
              logging.info('Room triangles'+str(room.Ntri))
              logging.warning('distance%f for ray with distance %f at position (%d,%d,%d,%d,%d) reflection number %d'%(distcor,dist,i1,j1,k1,row,col,nre))
          if abs(distcor)>epsilon and not doubcheck: # and not interiorcheck:
            # If the distance is 0 then the point is at the centre and the power is not well defined.
            # Find the positions of the nonzero terms in calcvec and check the number of terms is valid.
            for r in rind:
              if r==row and row>0 and anglechange:
                phi=s.angle_correct(dist,direc,roomnorm,p2,p1)
                Mesh[i1,j1,k1][row,col]=distcor*np.exp(1j*phi)
              else:
                Mesh[i1,j1,k1][r,col]=distcor*cal[r,0]
            if dbg:
              #----More sanity checks
              if calcvec.getnnz()!=nre+1:
                raise ValueError('incorrect vec length')
              if not Mesh.check_nonzero_col(Nre,Nsur,nre,(i1,j1,k1,col)):
                errmsg='The number of nonzero terms at (%d,%d,%d,%d) is incorrect'%(i1,j1,k1,col)
                raise ValueError(errmsg)
              if Mesh[i1,j1,k1][0].getnnz()>(Nsur*((Nsur-1)**Nre-1)/(Nsur-2)+1):
                logging.info('position (%d,%d,%d)'%(i1,j1,k1))
                logging.info(str(Mesh[i1,j1,k1]))
                errmsg='The number of rays which have entered the element is more than possible'
                raise ValueError(errmsg)
      if Ncon>0:
        # In this instance stpch==0 and end of ray has been reached keep storing cone points but not the points on the ray.
        Nc=s._number_cone_steps_(alpha,dist,deltheta,Ms)           # No. of cone steps required for this ray step.
        rayterms[2]=Nc
        copos2,Mesh=s.conestepping(iterterms,rayterms,programterms,p1,roomnorm,norm,direc,Mesh,calcvec,room,copos2)
      # Compute the next point along the ray
      p1+=alpha*direc*splitinv
      dist+=alpha*leng(direc)*splitinv
      iterterms[1]=dist
      i2,j2,k2=room.position(p1,h)
    return Mesh,outdist,calcvec

  def mesh_power_singleray(s,room,_Grid,dist,RefCoef,rayterms,Antpar,programterms,loghandle=str()):
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
    nsur=int(s.points[-2][-1])   # The obstacle number of the last reflection
    nob=sum(room.Ntri[0:nsur])
    Nx=_Grid.shape[0]
    Ny=_Grid.shape[1]
    Nz=_Grid.shape[2]

    Nra,Nre,anglechange,split=programterms[0:4].astype(int)
    splitinv,deltheta=programterms[-2:]
    khat,_,L=Antpar[0:3]
    nre,nra=rayterms[0:2].astype(int)
    theta=rayterms[2]

    # Compute the direction - Since the Ray has reflected but we are
    # storing previous information we want the direction of the ray which
    # hit the object not the outgoing ray.
    direc=lf.Direction(numparr([s.points[-3][0:3],s.points[-2][0:3]]))
    direc/=leng(direc)
    if abs(direc.any()-0.0)>epsilon:                       # Before computing the dist travelled through a mesh cube
                                                           # check the direction isn't 0.
      alpha=h/max(abs(direc))                              # Find which boundary of a unit cube gets hit first when
                                                           # direc goes through it.
    else: return _Grid, dist, RefCoef                    # If the direction vector is 0 nothing is happening.
    deldist=lf.length(numparr([(0,0,0),alpha*direc]))     # Calculate the distance travelled through a mesh cube
    p0=s.points[-3][0:3]                                   # Set the initial point to the start of the segment.
    if nre==0 and nra==0:
      p1=p0                                                  # p0 should remain constant and p1 is stepped.
    else:
      p1=p0+alpha*direc
      dist+=deldist
    i1,j1,k1=room.position(p1,h)                                                  # p0 should remain constant and p1 is stepped.
    endposition=room.position(s.points[-2][0:3],h)         # The indices of the end of the segment
    segleng=lf.length(vstack((s.points[-3][0:3],s.points[-2][0:3]))) # Length of the ray segment
    Ms=s.ray_maxsteps(room)
    Ns=s._number_steps_(deldist,segleng+dist,nra,deltheta,theta,Ms)                             # Compute the number of steps that'll be taken along the ray.
    # Compute a matrix with rows corresponding to normals for the cone.
    Ncon=s._number_cones_(deldist,dist+segleng,deltheta,theta,Ms)
    if Ncon>0:
      norm=s._normal_mat_(Ncon,Nra,direc,dist,h)             # Matrix of normals to the direc, of distance 1 equally angle spaced
      Nnor=len(norm)                                         # The number of normal vectors#
    else:
      Nnor=0
    # Initialise the check terms
    rep=0                               # Indicates if the point is the same as the previous
    stpch=DSM.stopcheck(i1,j1,k1,Nx,Ny,Nz)      # Check if the ray point is outside the domain.
    i2=-1
    j2=-1
    k2=-1
    for m1 in range(0,split*(Ns+1)):                             # Step through the ray
      if dbg:
        if abs(i2-i1)>1 or abs(j2-j1)>1 or abs(k2-k1)>1:
          errmsg='The ray has stepped further than one mesh element'
          raise ValueError(errmsg)
      if i2!=i1 or j2!=j1 or k2!=k1:    # If the indices are equal pass as no new element.
        i1=i2                             # Reset the check indices for the next check.
        j1=j2
        k1=k2
        stpch=DSM.stopcheck(i1,j1,k1,Nx,Ny,Nz)   # stopcheck finds 1 if the term is in the environment and 0 if not
      if stpch:
        p2=room.coordinate(h,i1,j1,k1) # Calculate the co-ordinate of the center of the element the ray hit
        #interiorcheck=room.check_innerpoint(p2)
        distcor=s.centre_dist(p1,p2,dist,room)
        if i1==xcheck and j1==ycheck and k1==zcheck and dbg:
          if nob==0 or nob==1:
            logging.warning('distance%f for ray with distance %f at position (%d,%d,%d,%d,%d) reflection number %d'%(distcor,dist,i1,j1,k1,nre))
        if abs(distcor)>epsilon: # not interiorcheck and
          if abs(distcor-dist)>sqrt(3)*h:
            raise ValueError('The corrected distance on the ray is more than a mesh width from the ray point distance')
            # If the distance is 0 then the point is at the centre and the power is not well defined.
          if distcor==0:
            _Grid[i1,j1,k1,0]=np.exp(1j*khat*distcor*(L**2))*RefCoef[0]
            _Grid[i1,j1,k1,1]=np.exp(1j*khat*distcor*(L**2))*RefCoef[1]
          else:
            _Grid[i1,j1,k1,0]+=(1.0/(distcor))*np.exp(1j*khat*distcor*(L**2))*RefCoef[0]
            _Grid[i1,j1,k1,1]+=(1.0/(distcor))*np.exp(1j*khat*distcor*(L**2))*RefCoef[1]
          Nc=s._number_cone_steps_(deldist,dist,deltheta,Ms)       # No. of cone steps required for this ray step.
          for m2 in range(0,split*(Nc)):
            p3=numparr([p1+(m2+1)*alpha*splitinv*n for n in norm])
            #np.tile(p1,(Ncon,1))+(m2+1)*alpha*norm*splitinv        # Step along all the normals from the ray point p1.
            copos=room.position(p3,h)                              # Find the indices corresponding to the cone points.
            if m2==0 and m1==0:
              copos2=zeros(copos.shape)
              check,altcopos,p3out,normout =DSM.stopchecklist(altcopos,p3,norm,Nx,Ny,Nz)  # The indices which do not correspond to positions in the environment are removed.
              copos2=altcopos
            else:
              altcopos,p3out,normout,copos2=duplicatecheck(copos,copos2,p3,norm,Nx,Ny,Nz) # If there are any cone indices which were in the previous step these are removed.
            if altcopos.shape[0]>0: # Check that there are some cone positions to store.
              # Check whether there is more than one vector in the list of cones.
              if isinstance(normout[0],(float,int,np.int64, np.complex128 )):
                Nnorcor=1
                coords=room.coordinate(h,altcopos[0],altcopos[1],altcopos[2]) # Find the centre element point for each cone normal.
              else:
                Nnorcor=normout.shape[0]
                coords=room.coordinate(h,altcopos[:,0],altcopos[:,1],altcopos[:,2])
              r2=s.centre_dist(p1,coords,dist,room)
              if Nnorcor==1:
                x,y,z=altcopos
                intercheck=room.check_innerpoint(coords)
                if not intercheck and abs(r2)>epsilon:
                  _Grid[x,y,z,0]+=(1.0/(r2))*np.exp(1j*khat*r2*(L**2))*RefCoef[0]
                  _Grid[x,y,z,1]+=(1.0/(r2))*np.exp(1j*khat*r2*(L**2))*RefCoef[1]
                continue
              else:
                for j in range(0,Nnorcor):
                  x,y,z=altcopos[j]
                  intercheck=room.check_innerpoint(coords[j])
                  if not intercheck and abs(r2[j])>epsilon:
                    # If the ray is already accounted for in this mesh square then step to the next point.
                    errmsg='Number of cone positions (%d,%d,%d) not the same as number of distances, %d'%(altcopos[j,0],altcopos[j,1],altcopos[j,2],r2[j])
                    if altcopos.shape[0]!=r2.shape[0]:
                      raise ValueError(errmsg)
                    _Grid[x,y,z,0]+=(1.0/(r2[j]))*np.exp(1j*khat*r2[j]*(L**2))*RefCoef[0]
                    _Grid[x,y,z,1]+=(1.0/(r2[j]))*np.exp(1j*khat*r2[j]*(L**2))*RefCoef[1]
      else:
        if Ncon>0:
          # In this instance stpch==0 and end of ray keep storing cone points but not on the ray.
          Nc=s._number_cone_steps_(alpha,dist,deltheta,Ms)           # No. of cone steps required for this ray step.
          for m2 in range(0,split*(Nc)):
            p3=numparr([p1+(m2+1)*alpha*splitinv*n for n in norm])
            #np.tile(p1,(Ncon,1))+(m2+1)*alpha*norm/split        # Step along all the normals from the ray point p1.
            copos=room.position(p3,h)                              # Find the indices corresponding to the cone points.
            if m2==0 and m1==0:
              copos2=zeros(copos.shape)
              check,altcopos,p3out,normout =DSM.stopchecklist(copos,p3,h,Nx,Ny,Nz)  # The indices which do not correspond to positions in the environment are removed.
              copos2=copos
            else:
              altcopos,p3out,normout,copos2=duplicatecheck(copos,copos2,p3,norm,Nx,Ny,Nz) # If there are any cone indices which were in the previous step these are removed.
            if altcopos.shape[0]>0: # Check that there are some cone positions to store.
              # Check whether there is more than one vector in the list of cones.
              if isinstance(normout[0],(float,int,np.int64, np.complex128 )):
                Nnorcor=1
                coords=room.coordinate(h,altcopos[0],altcopos[1],altcopos[2]) # Find the centre element point for each cone normal.
              else:
                Nnorcor=normout.shape[0]
                coords=room.coordinate(h,altcopos[:,0],altcopos[:,1],altcopos[:,2])
              r2=s.centre_dist(p1,coords,dist,room)
              if Nnorcor==1:
                x,y,z=altcopos
                intercheck=room.check_innerpoint(coords)
                if not intercheck and abs(r2)>epsilon:
                  _Grid[x,y,z,0]+=(1.0/(r2))*np.exp(1j*khat*r2*(L**2))*RefCoef[0]
                  _Grid[x,y,z,1]+=(1.0/(r2))*np.exp(1j*khat*r2*(L**2))*RefCoef[1]
                continue
              else:
                for j in range(0,Nnorcor):
                  x,y,z=altcopos[j]
                  intercheck=room.check_innerpoint(coords[j])
                  if not intercheck and abs(r2[j])>epsilon:
                    # If the ray is already accounted for in this mesh square then step to the next point.
                    errmsg='Number of cone positions (%d,%d,%d) not the same as number of distances, %d'%(altcopos[j,0],altcopos[j,1],altcopos[j,2],r2[j])
                    if altcopos.shape[0]!=r2.shape[0]:
                      raise ValueError(errmsg)
                    _Grid[x,y,z,0]+=(1.0/(r2[j]))*np.exp(1j*khat*r2[j]*(L**2))*RefCoef[0]
                    _Grid[x,y,z,1]+=(1.0/(r2[j]))*np.exp(1j*khat*r2[j]*(L**2))*RefCoef[1]
      # Compute the next point along the ray
      p1+=alpha*direc*splitinv
      dist+=alpha*splitinv
      i2,j2,k2=room.position(p1,h)
    return _Grid,dist,RefCoef
  def raytest(s,room,err):
    ''' Checks the test functions run '''
    try: singleray_test()
    except:
      raise Error('Single ray function fails')
    try: centre_dist_test()
    except:
      raise Error('Centre dist function fails')
    try: dup_check_test()
    except:
      raise Error('Duplicate check function fails')
    try: removedouble_test()
    except:
      raise Error('Duplicate check function fails')
    return

def centre_dist(direc,p1,p2,dist,room,h,nre=0,col=0,nob=12):
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
    direc/=leng(direc)
    if DSM.singletype(p2[0]):
      diff=p2-p1
      alcor=diff@direc
      alcor/=direc@direc
      l1=(alcor*leng(direc)+dist)**2
      l2=leng(diff-alcor*direc)**2
      distcor=sqrt(l1+l2)
      x,y,z=room.position(p2,h)
      if dbg:
        if x==xcheck and y==ycheck and z==zcheck:
         if nob==0 or  nob==1:#and col==27 :
            logging.info('Position (%f,%f,%f,%f)'%(p2[0],p2[1],p2[2],col))
            logging.info('Index position (%d,%d,%d,%d,%d)'%(x,y,z,0,col))
            logging.info('Ray starting position (%f,%f,%f)'%(p1[0],p1[1],p1[2]))
            # logging.info('The length of the direction vector %f'%np.linalg.norm(direc))
            logging.info('Ray direction (%f,%f,%f)'%(direc[0],direc[1],direc[2]))
            logging.info('Distance before %f'%dist)
            # logging.info('Alpha correction term %f'%alcor)
            # logging.info('l1 correction term %f'%l1)
            # logging.info('l2 correction term %f'%l2)
            logging.info('The corrected distance %f'%distcor)
            # logging.info('Diff (%f,%f,%f)'%(diff[0],diff[1],diff[1]))
            logging.info('Reflection number %d'%nre)
            logging.info('Obstacle number %d'%nob)
    else:
      Nnorcor=p2.shape[0]
      p1cone=np.tile(p1,(Nnorcor,1))               # The ray point is tiled to perform arithmetic which each cone vector.
      diffvec=np.subtract(p2,p1cone)               # The vector from the ray point to the cone-point.
      direcstack=np.tile(direc,(Nnorcor,1))        # The ray direction tiled to perform arithmetic with each cone vector.
      alcorrvec=lf.coordlistdot(diffvec,direcstack)# Compute the dot between the vector between the two element points and corresponding normal.
      #alcorrvec/=np.linalg.norm(direc)
      l1vec  =lf.coordlistdot(direcstack,direcstack)
      l1vec  =sqrt(l1vec)*alcorrvec+dist*ones(Nnorcor)
      l1vec  =numparr([l**2 for l in l1vec])
      l2vec  =diffvec-direcstack*alcorrvec[:,np.newaxis]
      l2vec  =lf.coordlistdot(l2vec,l2vec)
      distcor=sqrt(l1vec+l2vec)                 # r2 is the corrected distance.
                                                   # This is a vector with terms corresponding to the corrected distance for each cone point.
      if dbg:
          for j in range(Nnorcor):
            x,y,z=room.position(p2[j],h)
            if x==xcheck and y==ycheck and z==zcheck:
             if nob==0 or nob==1:#and col==27:
              #pdb.set_trace()
              logging.info('Position (%f,%f,%f,%f),Distance change after correction%f'%(p2[j,0],p2[j,1],p2[j,2],col,distcor[j]-dist))
              logging.info('Index position (%d,%d,%d,%d,%d)'%(x,y,z,0,col))
              logging.info('Ray starting position (%f,%f,%f)'%(p1[0],p1[1],p1[2]))
              # logging.info('The length of the direction vector %f'%np.linalg.norm(direc))
              logging.info('Ray direction (%f,%f,%f)'%(direc[0],direc[1],direc[2]))
              logging.info('Distance before %f'%dist)
              # logging.info('Alpha correction term %f'%alcorrvec[j])
              # logging.info('l1 correction term %f'%l1vec[j])
              # logging.info('l2 correction term %f'%l2vec[j])
              logging.info('The corrected distance %f'%distcor[j])
              # logging.info('Diff (%f,%f,%f)'%(diffvec[j,0],diffvec[j,1],diffvec[j,1]))
              logging.info('Reflection number %d'%nre)
              logging.info('Obstacle number %d'%nob)
    return distcor
def singleray_test():
    '''Test the stepping through a single ray. '''
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=numparr([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')[0:3]
    Nre=int(Nre)
    # Take Tx to be 0,0,0
    Tx=     np.load('Parameters/Origin.npy')
    delangle      =np.load('Parameters/delangle.npy')
    ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
    Oblist        =OuterBoundary #np.concatenate((Oblist,OuterBoundary),axis=0)# Oblist is the list of all the obstacles in the domain

    # Room contains all the obstacles and walls.
    Room=rom.room(Oblist)
    Nob=Room.Nob
    Nsur=Room.Nsur
    i=0
    directionname='Parameters/Directions%03d.npy'%Nra[i]
    data_matrix   =np.load(directionname)         # Matrix of ray directions
    j=np.random.randint(0,Nra[i])
    direc=data_matrix[j]
    ry=Ray(np.append(Tx,[0]),direc)
    nra=0
    Nx=int(Room.maxxleng()/(h))
    Ny=int(Room.maxyleng()/(h))
    Nz=int(Room.maxzleng()/(h))

    Mesh=DSM.DS(Nx,Ny,Nz,Nsur*Nre+1,Nra[i]*(Nre+1))
    end=ry.reflect_calc(Room)
    disttrue=leng(ry.points[0][0:3]-ry.points[1][0:3])
    dist=0
    calcvec=SM((Mesh.shape[0],1),dtype=np.complex128)
    split=3
    splitinv=1.0/split
    anglechange=1
    programterms=numparr([Nra[i],Nre,anglechange,split,splitinv,delangle[i]])
    Mesh,dist,calcvec=ry.mesh_singleray(Room,Mesh,dist,calcvec,programterms,nra)
    return 1


def centre_dist_test():
    '''Test the stepping through a single ray. '''
    Nra         =np.load('Parameters/Nra.npy')
    if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=numparr([Nra])
      nra=1
    else:
      nra=len(Nra)
    Nre,h ,L =np.load('Parameters/Raytracing.npy')[0:3]
    Nre=int(Nre)
    # Take Tx to be 0,0,0
    Tx=     np.load('Parameters/Origin.npy')
    delangle      =np.load('Parameters/delangle.npy')
    ##----Retrieve the environment--------------------------------------
    Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
    OuterBoundary =np.load('Parameters/OuterBoundary.npy')      # The Obstacles forming the outer boundary of the room
    # Room contains all the obstacles and walls.
    Room=rom.room(Oblist)
    Nob=Room.Nob
    Nsur=Room.Nsur
    i=0
    directionname=str('Parameters/Directions'+str(int(i))+'.npy')
    data_matrix   =np.load(directionname)         # Matrix of ray directions
    direc=data_matrix[0]
    ry=Ray(np.append(Tx,[0]),direc)
    nra=0
    Nx=int(Room.maxleng(1)/(h))
    Ny=int(Room.maxleng(2)/(h))
    Nz=int(Room.maxleng(3)/(h))
    split=3
    splitinv=1.0/split
    anglechange=1
    programterms=numparr([Nra[i],Nre,anglechange,split,splitinv,delangle[i]])
    Avenum=5
    time=0
    for j in range(Avenum):
        direc=np.random.rand(1,4)
        ry=Ray(np.append(Tx,[0]),direc)
        Mesh=DSM.DS(Nx,Ny,Nz,Nsur*Nre+1,Nra[i]*(Nre+1))
        end=ry.reflect_calc(Room)
        disttrue=np.linalg.norm(ry.points[0][0:3]-ry.points[1][0:3])
        dist=0
        calcvec=SM((Mesh.shape[0],1),dtype=np.complex128)
        t0=t.time()
        Mesh2,dist2,calcvec2=ry.mesh_singleray(Room,Mesh,dist,calcvec,programterms,nra)
        t1=t.time()
        time+=t1-t0
        print('Single ray time', t1-t0)
    print('average time',time/Avenum)
    return 1

def dup_check_test():
  Nra         =np.load('Parameters/Nra.npy')
  if isinstance(Nra, (float,int,np.int32,np.int64, np.complex128 )):
      Nra=numparr([Nra])
      nra=1
  else:
      nra=len(Nra)
  Nx=10
  Ny=10
  Nz=10
  newtime=0
  oldtime=0
  h=1.0/Nx
  dist=sqrt(3)
  delangle=max(np.load('Parameters/delangle.npy'))
  refangle=pi*0.5
  nref=10
  scal=no_cones(h,dist,delangle,refangle,nref)
  Avenum=2
  Avetot=0
  Nre,h ,L =np.load('Parameters/Raytracing.npy')[0:3]
  Nre=int(Nre)
  # Take Tx to be 0,0,0
  Tx=     np.load('Parameters/Origin.npy')
  delangle      =np.load('Parameters/delangle.npy')
  ##----Retrieve the environment--------------------------------------
  Oblist        =np.load('Parameters/Obstacles.npy')          # The obstacles which are within the outerboundary
  # Room contains all the obstacles and walls.
  Room=rom.room(Oblist)
  Nob=Room.Nob
  Nsur=Room.Nsur
  i=0
  for j in range(Avenum):
    for k in range(2,scal):
      for l in range(1,k):
        new=k
        old=l
        copos2=np.random.randint(1,10,size=(old,3))
        copos=np.random.randint(1,10,size=(new,3))
        p3=np.random.rand(new,3)
        norm=np.random.rand(new,3)
        #print('before')
        #print(copos,copos2,p3,norm)
        row=numparr([5,3,6])
        #print(np.allclose(row,copos))
        t0=t.time()
        altcoposout,p3out,normout,copos2out=duplicatecheck(copos,copos2,p3,norm)
        t1=t.time()
        #altcoposout,p3put,normout,copos2out=duplicatecheck_old(copos,copos2,p3,norm,Nx,Ny,Nz)
        #t2=t.time()
        #print('after')
        #print(altcoposout,copos2out,p3out,normout)
        newtime+=t1-t0
        #oldtime+=t2-t1
        Avetot+=1
  print('new method time',newtime/(Avetot))
  #print('old method time',oldtime/(Avetot))
    # The 'new' method is slower so the old method is kept
  return 1


def power_singleray_test(Mesh):

  ##----Retrieve the Raytracing Parameters-----------------------------
  Nre,h,L    =np.load('Parameters/Raytracing.npy')[0:3]
  Nra        =np.load('Parameters/Nra.npy')
  i=0
  Nre=int(Nre)
  Nob            =np.load('Parameters/Nob.npy')
  Nsur           =np.load('Parameters/Nsur.npy')

  G_z=zeros((1,1))

  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  Grid=zeros((Nx,Ny,Nz),dtype=float)
  PI.ObstacleCoefficients(0)
  ##----Retrieve the antenna parameters--------------------------------------
  gainname      ='Parameters/Tx%03dGains%d.npy'%(Nra[i],0)
  Gt            = np.load(gainname)
  freq          = np.load('Parameters/frequency%d.npy'%0)
  Freespace     = np.load('Parameters/Freespace.npy'%0)
  Pol           = np.load('Parameters/Pol.npy'%0)

  ##----Retrieve the Obstacle Parameters--------------------------------------
  Znobrat      =np.load('Parameters/Znobrat.npy'%0)
  refindex     =np.load('Parameters/refindex.npy'%0)
  # Make the refindex, impedance and gains vectors the right length to
  # match the matrices.
  Znobrat=np.tile(Znobrat,(Nre,1))          # The number of rows is Nsur*Nre+1 Repeat Nob
  Znobrat=np.insert(Znobrat,0,1.0+0.0j)     # Use a zero for placement in the LOS row
  refindex=np.tile(refindex,(Nre,1))
  refindex=np.insert(refindex,0,1.0+0.0j)
  Gt=np.tile(Gt,(Nre+1,1))

  # Calculate the necessry parameters for the power calculation.
  c             =Freespace[3]
  khat          =freq*L/c
  lam           =(2*pi*c)/freq
  Antpar        =numparr([khat,lam,L])
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

def duplicatecheck(altcopos,copos2,p3,norm):
  '''
  :param copos2: The previous set of cone positions to compare to
  :param altcopos: The current set of cone positions to be output after reductions
  :param p3: The points corresponding to copos2
  :param norm: The normal vectors corresponding to p3
  :param Nx: number of steps in x-axis
  :param Ny: number of steps in y-axis
  :param Nz: number of steps in the z-axis'''
  if len(altcopos.shape)>1 and altcopos.shape[0]>0:
    if isinstance(altcopos[0],(float,int,np.int64, np.complex128 )):
      return altcopos,p3,norm,altcopos.copy()
    if copos2.shape[0]>0 and len(copos2.shape)>1:
      ind=[ind for ind,row in enumerate(altcopos) if all(not (row == x).all() for x in copos2)]#not all(row==row2) for row2 in copos2]
      if len(ind)<1:
        temp=zeros((0,altcopos.shape[1]))
        return temp.copy(),temp.copy(),temp.copy(),altcopos.copy()
      else:
        return altcopos[ind],p3[ind],norm[ind],altcopos[ind].copy()
    else: return altcopos,p3,norm,altcopos.copy()
  else:
    return altcopos,p3,norm,altcopos.copy()


def duplicatecheck_old(copos,copos2,p3,norm,Nx,Ny,Nz):
  '''
  :param copos: The previous set of cone positions to compare to
  :param copos2: The current set of cone positions to be output after reductions
  :param p3: The points corresponding to copos2
  :param norm: The normal vectors corresponding to p3
  :param Nx: number of steps in x-axis
  :param Ny: number of steps in y-axis
  :param Nz: number of steps in the z-axis'''
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
  p3out=numparr([])
  normout=numparr([])
  altcopos=numparr([])
  rep=0
  if Ncon==1 and Ncon2==1:
    if DSM.stopcheck(copos[0],copos[1],copos[2],Nx,Ny,Nz):
      double=0
      if copos[0]==copos2[0] and copos[1]==copos2[1] and copos[2]==copos2[2]:
        double=1
      else:
         altcopos=vstack((altcopos,numparr([copos[0],copos[1],copos[2]])))
         p3out=vstack((p3out,numparr([p3[0],p3[1],p3[2]])))
         normout=vstack((normout,numparr([norm[0],norm[1],norm[2]])))
  elif Ncon==1 and Ncon2!=1:
    if DSM.stopcheck(copos[0],copos[1],copos[2],Nx,Ny,Nz):
      double=0
      for k in range(0,Ncon2):
        if copos[0]==copos2[k][0] and copos[1]==copos2[k][1] and copos[2]==copos2[k][2]:
          double=1
          break
      if double!=1:
        altcopos=copos
        p3out=p3
        normout=norm
        rep=1
  elif Ncon!=1 and Ncon2==1:
     for j in range(0,Ncon2):
       if DSM.stopcheck(copos[j][0],copos[j][1],copos[j][2],Nx,Ny,Nz):
         double=0
         if copos2[0]==copos[j][0] and copos2[1]==copos[j][1] and copos2[2]==copos[j][2]:
           double=1
         if double!=1 and rep==0:
           altcopos=numparr([copos[j][0],copos[j][1],copos[j][2]])
           p3out=numparr([p3[j][0],p3[j][1],p3[j][2]])
           normout=numparr([norm[j,0],norm[j,1],norm[j,2]])
           rep=1
         elif double!=1 and rep==1:
           altcopos=vstack((altcopos,numparr([copos[j][0],copos[j][1],copos[j][2]])))
           p3out=vstack((p3out,numparr([p3[j][0],p3[j][1],p3[j][2]])))
           normout=vstack((normout,numparr([norm[j,0],norm[j,1],norm[j,2]])))
  else:
    for j in range(0,Ncon):
      if DSM.stopcheck(copos[j][0],copos[j][1],copos[j][2],Nx,Ny,Nz):
        double=0
        for k in range(0,Ncon2):
          #print(Ncon,Ncon2,j,k)
          if copos[j][0]==copos2[k][0] and copos[j][1]==copos2[k][1] and copos[j][2]==copos2[k][2]:
            double=1
            break
        if double!=1 and rep==0:
          altcopos=numparr([copos[j][0],copos[j][1],copos[j][2]])
          p3out=numparr([p3[j][0],p3[j][1],p3[j][2]])
          normout=numparr([norm[j,0],norm[j,1],norm[j,2]])
          rep=1
        elif double!=1 and rep==1:
          altcopos=vstack((altcopos,numparr([copos[j][0],copos[j][1],copos[j][2]])))
          p3out=vstack((p3out,numparr([p3[j][0],p3[j][1],p3[j][2]])))
          normout=vstack((normout,numparr([norm[j,0],norm[j,1],norm[j,2]])))
  copos2=copos
  return altcopos,p3out,normout,copos2

def conestepping(iterterms,rayterms,programterms,p1,roomnorm,norm,direc,Mesh,calcvec,room,copos2=numparr([])):
  '''Step through the lines which form a cone on the ray and store the ray information. Check if the ray is already stored in the
  mesh and that the number of non-zero terms in a column is correct after storing.
  :param rayterms:     nra-ray number,nre-reflection number,Nc-Number of steps to take along the cone,col-column number,alpha-distance stepped through a voxel.
  :param iterterms:    olddist-distance ray travelled to previous intersection, dist-distance ray travelled to the p1.
  :param programterms: Nra- total number of rays,Nre- total number of reflections.
  :param p1:           The point on the ray the cones step away from. A numpy array with shape (1,3).
  :param roomnorm:     Norm vector to most recently intersected surface.
  :param norm:         Array with shape (Nc,3) with each row corresponding to a vector normal to the ray.
  :param direc:        The direction the ray is travelling in.
  :param Mesh:         DSM containing the ray tracer information so far.
  :param calcvec:      Vector containing the information about this ray which will be stored.
  :param room:         room object which obstacle co-ordiantes, norms and number of triangles per surface.

  Steps through the points on the cone vectors and check if the points have previousl been stored at. Removes the doubles then
  stores calcvec at the points. Continued until the end of the cone is reached.'''
  # Retrieve the program paramters from the objects containing them
  nra,nre,Nc,col=rayterms[0:4].astype(int)  # Ray number, reflection number, number of steps along the cone vectors to take, column in matrix to store calcvec
  alpha=rayterms[4]                         # Distance stepped through a voxel at each step
  olddist,dist=iterterms[0,1]               # Distance ray travelled up to intersection and to current ray point.
  m1=iterterms[2].astype(int)                      # Step along the ray
  Nra,Nre,anglechange,_,_,_=programterms.astype(int)     # Total number of rays and reflections, angle correction switch and steps through each voxel.
  Nsur=room.Nsur                            # Total number of surfaces
  rind=cal.nonzero()[0] # Nonzero rows of the ray storage vector
  row=rind[-1]              # Row number corresponding to recent entry in calcvec
  theta=np.angle(cal[row,0])            # Most recent reflection angle
  nsur=DSM.nob_fromrow(row,Nsur)            # The surface number from the last reflection.
  conesplit=2
  conesplitinv=0.5
  if dbg:
    assert calcvec.getnnz()==nre+1
    if alpha*conesplitinv>h:                       # The step size should never be bigger than the meshwidth
      errmsg='The step size %f along the ray should not be bigger than the mesh width %f'%(alpha/split,h)
      raise ValueError(errmsg)
  # Iterate along the normal vectors which form the cones. At each step check for double counting and duplicate points.
  for m2 in range(0,conesplit*(Nc)):
    p3=np.tile(p1,(Ncon,1))+(m2+1)*alpha*norm*conesplitinv   # Step along all the normals from the ray point p1.
    copos=room.position(p3,h)                         # Find the indices corresponding to the cone points.
    if m2==0 and m1==0:
      altcopos,p3out,normout,copos2=removedoubles(copos,p3,norm,Mesh)     # If there are any repeated cone indices in copos then these are removed.
      #check,altcopos,p3out,normout =Mesh.stopchecklist(altcopos,p3,norm)  # The indices which do not correspond to positions in the environment are removed.
      copos2=altcopos
    else:
      altcopos,p3out,normout,copos =removedoubles(copos,p3,norm,Mesh)             # If there are any cone indices in list twice these are removed.
      altcopos,p3out,normout,copos2=duplicatecheck(copos,copos2,p3,norm) # If there are any cone indices which were in the previous step these are removed.
    if altcopos.shape[0]>0: # Check that there are some cone positions to store.
      # Check whether there is more than one vector in the list of cones.
      if isinstance(normout[0],(float,int,np.int64, np.complex128 )):
        Nnorcor=1
        coords=room.coordinate(h,altcopos[0],altcopos[1],altcopos[2]) # Find the centre element point for each cone normal.
      else:
        Nnorcor=normout.shape[0]
        coords=room.coordinate(h,altcopos[:,0],altcopos[:,1],altcopos[:,2])
      # Correct the distance travelled to the distance corresponding to the co-ordinate at the centre of all the voxels which will be stored at.
      r2=s.centre_dist(p1,coords,dist,room,nre,col)
      if Nnorcor==1:    # If there is only one term then there is no indexing for r2
        x,y,z=altcopos  # The position for the cone term
        intercheck=room.check_innerpoint(coords)                          # Check the term is not inside an obstacle
        doubcheck=Mesh.doubles__inMat__(calcvec,(x,y,z),rind) # Check this propagation level has not already been stored
        if not doubcheck and not intercheck and abs(r2)>epsilon:
          for r in rind:
            if r==row and row>0 and anglechange:
              phi=s.angle_correct(dist,direc,roomnorm,coords,p1)              # If angle correction is turned on then correct the most recent reflection angle to correspond to the center point being stored for
              Mesh[x,y,z][row,col]=r2*np.exp(1j*phi)                        # Correct the last reflection term
            else:
              Mesh[x,y,z][r,col]=r2*calcve[r,0]                      # Store the ray information at the mesh element
          if dbg:
            if x==xcheck and y==ycheck and z==zcheck :
                logging.info('Double status'+str(Mesh.doubles__inMat__(calcvec,(x,y,z),rind)))
                logging.info('Meshwidth %f, Obstacle number %d'%(h,nob))
                logging.info('Calcvec'+str(calcvec))
                logging.info('Room triangles'+str(room.Ntri))
                logging.warning('distance%f for ray with distance %f at position (%d,%d,%d,%d,%d) reflection number %d'%(r2,dist,x,y,x,0,col,nre))
                logging.info('Mesh stored'+str(Mesh[x,y,z][:,col]))
            if not Mesh.check_nonzero_col(Nre,Nsur,nre,(altcopos[0],altcopos[1],altcopos[2],col)):
              # Check the number of terms stored
              errmsg1='The number of nonzero terms at '
              errmsg2='(%d,%d,%d) col %d is not %d'%(x,y,z,col,nre+1)
              errmsg=errmsg1+errmsg2
              raise ValueError(errmsg)
          continue
      else: # More than one cone term to be stored
        for j in range(0,Nnorcor):                    # Go through each of the cone positions one by one
          x,y,z=altcopos[j]                           # Position at the centre of the element to be stored.
          intercheck=room.check_innerpoint(coords[j]) # Check the position is not inside an obstacle
          doubcheck= Mesh.doubles__inMat__(calcvec,(x,y,z),rind) # Check if this popagation level has already been stored
          if dbg:
            if x==xcheck and y==ycheck and z==zcheck:
              logging.info('Double status'+str(doubcheck))
              logging.info('Meshwidth %f, Surface number %d'%(h,nsur))
              logging.info('Calcvec'+str(calcvec))
              logging.info('Room triangles'+str(room.Ntri))
              logging.info('Index position (%d,%d,%d,%d,%d)'%(x,y,z,0,col))
              logging.warning('distance%f for ray with distance %f at position (%d,%d,%d,%d,%d) reflection number %d'%(r2[j],dist,x,y,z,0,col,nre))
              logging.info('vec stored at (%d,%d,%d,%d,%d)'%(x,y,z,0,col))
              logging.info('Number of nonzero %d'%calcvec.getnnz())
              logging.info('stored vec is'+str(Mesh[x,y,z][:,col]))
          if not doubcheck and not intercheck and abs(r2[j])>epsilon:
            # If the ray is already accounted for in this mesh square then step to the next point.
            if dbg:
              if altcopos.shape[0]!=r2.shape[0]:
                errmsg='Number of cone positions (%d,%d,%d) not the same as number of distances, %d'%(altcopos[j,0],altcopos[j,1],altcopos[j,2],r2[j])
                raise ValueError(errmsg)
              assert Mesh[x,y,z].shape[0]==calcvec.shape[0]
            for r in rind:
              if r==row and row>0 and anglechange:
                phi=s.angle_correct(dist,direc,roomnorm,coords[j],p1)  # If angle correction is turned on then correct the reflection angle to be for the point being stored at
                Mesh[x,y,z][row,col]=r2[j]*np.exp(1j*phi)            # Overwrite the previously stored angle with the corrected term.
              else:
                Mesh[x,y,z][r,col]=r2*cal[r,0]                       # Store the ray information at the mesh element
            if dbg:
              if not Mesh.check_nonzero_col(Nre,Nsur,nre,numparr([altcopos[j,0],altcopos[j,1],altcopos[j,2],col])):
                # Check that the correct number of terms have been stored in the mesh column
                errmsg1='The number of nonzero terms at '
                errmsg2='(%d,%d,%d) col %d is not %d'%(altcopos[j][0],altcopos[j][1],altcopos[j][2],col,nre+1)
                errmsg=errmsg1+errmsg2
                raise ValueError(errmsg)
  return copos2,Mesh

def removedoubles_old(copos,p3,norm,Mesh):
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
            break
      if double==0 and rep==0:
       altcopos=copos[j]
       p3out=p3[j]
       normout=norm[j]
       rep=1
      elif double==0 and rep!=0:
       #print('infunc',altcopos)
       altcopos=vstack((altcopos,numparr([copos[j,0],copos[j,1],copos[j,2]])))
       p3out=vstack((p3out,numparr([p3[j,0],p3[j,1],p3[j,2]])))
       normout=vstack((normout,numparr([norm[j,0],norm[j,1],norm[j,2]])))
      else:
       if rep==0:
         altcopos=numparr([copos[j,0],copos[j,1],copos[j,2]])
         p3out=numparr([p3[j,0],p3[j,1],p3[j,2]])
         normout=numparr([norm[j,0],norm[j,1],norm[j,2]])
         rep=1
       else:
         if isinstance(altcopos[0],(float,int,np.int64, np.complex128 )):
           N=1
         else:
           N=altcopos.shape[0]
         already=0
         if N==1:
             if copos[j,0]==altcopos[0] and copos[j,1]==altcopos[1] and copos[j,2]==altcopos[2]:
              already=1
         else:
           for l in range(0,N):
             if copos[j,0]==altcopos[l,0] and copos[j,1]==altcopos[l,1] and copos[j,2]==altcopos[l,2]:
              already=1
         if already==0:
           altcopos=vstack((altcopos,numparr([copos[j,0],copos[j,1],copos[j,2]])))
           p3out=vstack((p3out,numparr([p3[j,0],p3[j,1],p3[j,2]])))
           normout=vstack((normout,numparr([norm[j,0],norm[j,1],norm[j,2]])))
  if rep==0:
    altcopos=copos
    p3out=p3
    normout=norm
  return altcopos,p3out,normout,copos

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
  altcopos,ind=np.unique(copos,axis=0,return_index=True)
  p3=p3[ind]
  norm=norm[ind]
  ind=[ind for ind,row in enumerate(altcopos) if Mesh.stopcheck(row[0],row[1],row[2]) ] # Check the positions are in the room
  if len(ind)<1:
    temp=zeros((0,altcopos.shape[1]))
    return temp.copy(),temp.copy(),temp.copy(),temp.copy()
  else:
    return altcopos[ind],p3[ind],norm[ind],altcopos[ind].copy()

def removedouble_test():
  ##----Retrieve the Mesh--------------------------------------
  meshname='testDS'
  Mesh= DSM.load_dict(meshname)
  newtime=0
  Avenum=100
  Avetot=0
  Nsur=np.load('Parameters/Nsur.npy')
  Nre,_,_,_    =np.load('Parameters/Raytracing.npy')
  rn=Mesh.shape[0]
  cn=Mesh.shape[1]
  Nx=Mesh.Nx
  Ny=Mesh.Ny
  Nz=Mesh.Nz
  h=1.0/max(Nx,Ny,Nz)
  dist=sqrt(3)
  delangle=max(np.load('Parameters/delangle.npy'))
  refangle=pi*0.5
  nref=10
  scal=no_cones(h,dist,delangle,refangle,nref)
  for j in range(Avenum):
    for k in range(1,scal):
      new=k
      arr=np.random.randint(1,10,size=(new,3))
      p3=np.random.rand(new,3)
      norm=np.random.rand(new,3)
      t0=t.time()
      arrout,p3out,normout,copos2=removedoubles(arr,p3,norm,Mesh)
      t1=t.time()
      #print('after')
      #print(altcoposout,copos2out,p3out,normout)
      newtime+=t2-t1
      Avetot+=1
  print('new method time',newtime/(Avetot))
  return 1


def no_cones(h,dist,delangle,refangle,nref,maxleng=0):
     '''find the number of normals needed to form the cone.

     :param h: meshwidth
     :param delangle: The angle spacing between rays
     :param dist: The distance the ray has travelled
     :param refangle: The angle the ray reflected with.
     :param nref: The reflection number.
     :param maxleng: The maximum length contained in the room.

     Ncon=Number of steps taken along a cone to get to the edge or the boundary of the environment.

     :math:delth=:py:func:'angle_space'(delangle,nref)
     :math:\beta=:py:func:'beta_leng'(dist,delth,refangle)
     :math:Ncon=1+\\frac{\pi}{\\arcsin(\\frac{h}{4\beta})}

     :rtype: integer
     :returns: max(Ncon,maxleng//h)'''
     refangle=pi*0.5-refangle
     delth=angle_space(delangle)
     beta=beta_leng(dist,delth,refangle)
     if beta<(h/4):
       Ncon=0
     else:
       Ncon=int(1+pi/arcsin(h/(4*beta))) # Compute the distance of the normal vector
                            # for the cone and the number of mesh points
                            # that would fit in that distance.
     return Ncon

def no_steps(alpha,segleng,dist,delangle,refangle=0.0):
  '''The number of steps along the ray between intersection points'''
  refangle=pi*0.5-refangle
  rhat=extra_r(dist,delangle,refangle)
  ns=(segleng+rhat)/alpha
  return int(1+ns)

def angle_correct(dist,direc,n,x,p0):
  ''' When the ray information is stored the point which the storage corresponds to is the
  centre of an element which may be different from the point on the ray. The angle which this
  new ray hits an obstacle is different to the angle being stored. This function finds the angle
  corresponding to a a ray which hits the point x.
  :param dist:
  :param direc:
  :param n:
  :param x:
  :param p0:
  '''
  Txhat=p0-dist*direc/leng(direc)
  #print(Txhat)
  if np.linalg.norm(n)==0 or leng(x-Txhat)==0:
    pdb.set_trace()
  phi=arccos(((Txhat-x)@n)/(leng(n)*leng(Txhat-x)))
  #if dbg:
  if not 0<=phi<=pi*0.5:
    phi=pi-phi
    #pdb.set_trace()
  #if abs(phi-pi/2)<epsilon:
  #  pdb.set_trace()
  return phi

def angle_space(delangle,nref=0):
  s=sin(delangle*0.5)
  if abs(sqrt(2.0)*s-1.0)<epsilon:
    # There are no diagonal spacings to correct for
    return delangle
  else:
    return 2.0*arcsin(sqrt(2.0)*s)

def beta_leng(dist,delth,refangle):
    # Nra>2 and an integer. Therefore tan(theta) exists.
    ta=tan(0.5*delth)
    rhat=extra_r(dist,delth,refangle)
    return (rhat+dist)*ta

def extra_r(dist,delth,refangle=0.0):
    if refangle>pi*0.25:
      refangle=0.5*pi-refangle
    ta=tan(delth*0.5)
    t2=tan(refangle+delth*0.5)
    top2=0.5*ta*(sin(2.0*refangle)+(1.0-cos(2.0*refangle))*t2)
    return dist*top2
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
     delth=2.0*arcsin(sqrt(2.0)*sin(delangle*0.5))
     t=ma.tan(delth*0.5)    # Nra>2 and an integer. Therefore tan(theta) exists.
     Nc=int(1.0+(dist*t/h)) # Compute the distance of the normal vector for
                          # the cone and the number of mesh points that would
                          # fit in that distance.
     return 2*Nc

if __name__=='__main__':
  print('Running  on python version')
  print(sys.version)
  Mesh=centre_dist_test()
  #power_singleray_test(Mesh)
  singleray_test()
  dup_check_test()
  removedoubletest()

  exit()
