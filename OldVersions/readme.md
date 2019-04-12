
INTENSIONS HWRAGG 19-05-2017:
FILES
Objects.py       - This should contain classes which structure the rays.
                   Each ray is stored as co-ordinates and a direction.
                 - The ray class should store the previous trajectory
                    of the ray as well as it's current propagation.
                 - It should also define operations on the objects to
                   enable them to reflect.
                 - There should be a furniture class to hold the
                   coefficients and functions on the objects.
Room.py          - This should set the initial conditions of the room.
                   It should also contain the initial position and
                   direction of the source router.
Ray_tracer.py    - This should be the main program to send the
                   reflections.
ray_tracer_test.py- Using the error check in reflection tests 1
                   reflection for uniform propagated rays in the room
                   at the testing origin.
Intersection.py  - Finds the intersection of a ray and a line segment.
reflection.py    - Code to Reflect a line in an edge without using
                   Shapely. Takes in a line as origin and direction and
                   reflects with an edge represement by two pairs of
                   co-ordinates.
HayleysPlotting.py- Plotting functions for rays, edges and points.
linefunctions.py  - Some basic line functions.
testprog.py       - Program to test the reflection and intersection code.

NEED: The signal loss along the path needs to be calculated at each
reflection and  stored relevant to the co-ordinate somehow. This can
then be used to create a heat map.
NEED: 22-05-2017 Take the reflection and intersection functions into the
ray classes. This has been started but not finished. 10-10-2017 DONE
NEED: 26-05-2017 -The intersection needs to determine whether the point
is within the edge bounds. I think this could go in the edge class under
a term bound. Then this term can be used to determine diffraction later
on. 10-07-2017 DONE
- There also needs to be an iteration over the edges in
 the environment to determine all intersections and find the closest.
 10-07-2017 DONE
NEED: 10-07-2017 - In the objects module there needs to be changes to
the ray class. - The reflection function needs to add to the array not
replace terms. 13-10-2017 DONE

 BUILDING HWRAGG 13-07-2017:
  HayleysPlotting.py      - HWRAGG's code for plotting lines, edges and
                           points.
                         - The function Plotledge takes in pairs of
                           co-ordinates and a colour and plots the line
                           between the points in the colour.
                         - The function Plotline takes a line with a
                           start and direction, and a length and a
                           colour. It plots a line of the length from
                           the start point in the direction in the given
                            colour.
                         - Plotpoint plots a point using it's
                           co-ordinates as an x. This does not take in a
                           colour.
 reflection.py           - HWRAGG's Reflection
                           function without shapely. This is a module
                           containing the function and a test and an
                           error check for the test.
                         - The code now works and the test function
                           checks 9 different rays and edges
                           combinations, to make sure the angle to the
                           normal is the same for the input and
                           reflection.
                         - The function still needs to be edited so that
                           the length of the reflection should be given
                           by a constant that depends on the size of the
                           environment to enable it to be used in a room
                           setting.
                         - Changes have been made to prevent changes in
                          the edges.
 intersection.py         - Contains the intersection function and it's
                           test.
                         - Function now tests that the intersection lies
                           within the bounds of the edges. Currently
                           doesn't store a diffraction reference term so
                           this needs to added.
 testprog.py             - A test program which calls the functions in
                           reflection.py and intersection.py and tests
                           them.
 linefunctions.py        - A module containing some functions to compute
                           basic properties of a line. Including the
                           direction length and and dot product.
 objects.py              - Classes for the ray and wall-segment and room.
                         - Now amends rays as they travel to store
                         trajectory.
 ray_tracer.py           -Program for reflecting a ray around a room.

  BUILDING HWRAGG 10-07-2017:
 HayleysPlotting.py      - HWRAGG's code for plotting lines, edges and
                           points.
                         - The function Plotledge takes in pairs of
                           co-ordinates and a colour and plots the line
                           between the points in the colour.
                         - The function Plotline takes a line with a
                           start and direction, and a length and a
                           colour. It plots a line of the length from
                           the start point in the direction in the given
                            colour.
                         - Plotpoint plots a point using it's
                           co-ordinates as an x. This does not take in a
                           colour.
 reflection.py           - HWRAGG's Reflection
                           function without shapely. This is a module
                           containing the function and a test and an
                           error check for the test.
                         - The code now works and the test function
                           checks 9 different rays and edges
                           combinations, to make sure the angle to the
                           normal is the same for the input and
                           reflection.
                         - The function still needs to be edited so that
                           the length of the reflection should be given
                           by a constant that depends on the size of the
                           environment to enable it to be used in a room
                           setting.
 intersection.py         - Contains the intersection function and it's
                           test.
                         - Function now tests that the intersection lies
                           within the bounds of the edges. Currently
                           doesn't store a diffraction reference term so
                           this needs to added.
 testprog.py             - A test program which calls the functions in
                           reflection.py and intersection.py and tests
                           them.
 linefunctions.py        - A module containing some functions to compute
                           basic properties of a line. Including the
                           direction length and and dot product.
 objects.py              - Classes for the ray and wall-segment.
                         - Currently contains program testing the
                         classes. This should be moved to testprog.py
  BUILDING HWRAGG 22-05-2017:
 FILES:
 HayleysPlotting.py      - HWRAGG's code for plotting lines, edges and
                           points.
                         - The function Plotledge takes in pairs of
                           co-ordinates and a colour and plots the line
                           between the points in the colour.
                         - The function Plotline takes a line with a
                           start and direction, and a length and a
                           colour. It plots a line of the length from
                           the start point in the direction in the given
                            colour.
                         - Plotpoint plots a point using it's
                           co-ordinates as an x. This does not take in a
                           colour.
 reflection.py           - HWRAGG's Reflection
                           function without shapely. This is a module
                           containing the function and a test and an
                           error check for the test.
                         - The code now works and the test function
                           checks 9 different rays and edges
                           combinations, to make sure the angle to the
                           normal is the same for the input and
                           reflection.
                         - The function still needs to be edited so that
                           the length of the reflection should be given
                           by a constant that depends on the size of the
                           environment to enable it to be used in a room
                           setting.
 intersection.py         - Contains the intersection function and it's
                           test.
 testprog.py             - A test program which calls the functions in
                           reflection.py and intersection.py and tests
                           them.
  BUILDING HWRAGG 19-05-2017:
 FILES:
 reflection.py           - HWRAGG's attempt to write a reflection
                           function without shapely. This is a module
                           containing the function and a test.
                         - The function needs to be edited so that the
                         length of the reflection should be given by a
                         constant that depends on the size
  #of the environment
 intersection.py         - Contains the intersection function and it's
                           test.
 testprog.py             - A test program which calls the functions in
                           reflection.py and intersection.py and tests
                           them.
 BUILDING HWRAGG 18-05-2017:
 FILES:
 ray_tracing_ideas_00.py - KBRIGGS initial ideas for rewriting the ray
                           tracer.
 ray_tracing_ideas_01.py - HWRAGGS first edit and attempt to fix KBRIGGS
                           code.
 ray_tracing_ideas_02.py - HWRAGG's current attempt to write a new
                           ray-tracer
 reflection.py           - HWRAGG's attempt to write a reflection
                           function without shapely.
 intersection.py         - Contains the intersection function and it's
                           test.





