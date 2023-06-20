#!/usr/bin/env python3
# Hayley Wragg 2019-02-28
''' Code for a ray class '''

class Ray:
  ''' represents a ray by a collection of line segments followed by
      an origin and direction.  A Ray will grow when reflected replacing
       the previous direction by the recently found intersection and
       inserting the new direction onto the end of the ray.
  '''
  def __init__(s,start,origin,direction):
    s.ray=np.vstack(
      (np.array(origin,   dtype=np.float),
       np.array(direction,dtype=np.float),
    ))
  def __str__(s):
    return 'Ray(\n'+str(s.ray)+')'
  def _get_origin(s):
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
    return [s._get_origin(), s._get_direction()]
  def wall_collision_point(s,wall_segment):
    ''' intersection of the ray with a wall_segment '''
    return ins.intersection(s._get_travellingray(),wall_segment)
  def room_collision_point(s,room):
    ''' The closest intersection out of the possible intersections with
    the wall_segments in room. Returns the intersection point and the
    wall intersected with '''
    # Retreive the Maximum length from the Room
    leng=room.maxleng()
    # Initialise the point and wall
    rwall=room.walls[0]
    rcp=s.wall_collision_point(rwall)
    # Find the intersection with all the walls and check which is the
    #closest. Verify that the intersection is not the current origin.
    for wall in room.walls[1:]:
      cp=s.wall_collision_point(wall)
      if (cp[0] is not None and (cp!=s.ray[-2]).all()):
        leng2=s.ray_length(cp)
        if (leng2<leng) :
          leng=leng2
          rcp=cp
          rwall=wall
    return rcp, rwall
  def ray_length(s,inter):
    '''The length of the ray upto the intersection '''
    o=s._get_origin()
    ray=np.array([o,inter])
    return lf.length(ray)
  def reflect(s,room):
    ''' finds the reflection of the ray inside a room'''
    cp,wall=s.room_collision_point(room)
    # Check that a collision does occur
    if cp[0] is None: return
    else:
      # Construct the incoming array
      origin=s._get_origin()
      ray=np.array([origin,cp])
      # The reflection function returns a line segment
      refray,n=ref.try_reflect_ray(ray,wall.p)
      # update self...
      s.ray[-1]=cp
      s.ray=np.vstack((s.ray,lf.Direction(refray)))
    return
  def multiref(s,room,m):
    ''' Takes a ray and finds the first five reflections within a room'''
    for i in range(1,m+1):
      s.reflect(room)
    # print('iteraction', i, 'ray', s.ray)
    return s.ray[0:-1]
  def Plotray(s,room):
    ''' Plots the ray from point to point and the final travelling ray
    the maximum length in the room '''
    rayline1=s.ray[0]
    wid=7
    for rayline2 in s.ray[0:-1]:
      wid=wid*0.5
      hp.Plotray(np.array([rayline1, rayline2]),'BlueViolet',wid)
      rayline1=rayline2
    #hp.Plotline(s.ray[-2:],room.maxleng(),'b')
    return
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
      print('ray',ray, 'refray', refray, 'error', err)
    return err
