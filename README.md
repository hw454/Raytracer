# Generalised Ray launching
**Notice:** This repository developed over my PhD I am making this public but no longer work in this area. The core idea of this package is to store the history of the rays travelling (most raytracers will calculate power as they go along). Storing the ray history improves error from double counting. 

The theory underlying this package is covered in my thesis: https://researchportal.bath.ac.uk/en/studentTheses/indoor-propagation-of-radio-frequency-small-cells

This repository contains all the Raytracer versions.

The latest version is in the folder `/ParameterIndependent/`. 
Clone this folder to run the raylauncher.

## ParameterIndependent:
The current Raytracer uses a method for iterating along rays and storing information then outputs a mesh. This mesh is then input into another function to compute power.
To run the program 
  * run `RayTracerMainProgram.py`. This will run the main ray tracer and save the output. It will then compute the power using the inputs from "ParameterInput.py" and show heatmap slices corresponding to this and save this figures too.

To change any parameters edit `ParameterInput.py`
  * To edit the ray tracer parameters *(number of rays, number of grid points, number of reflections, length scale)* edit `DeclareParameters()`.
  * To edit the geometry edit `DeclareParameters()`
  * To edit obstacle coefficients edit `ObstacleCoefficients()`
  * To edit antenna gains or wavelength edit `ObstacleCoefficients()`

Status: Currently the program is computing power values but this has not be completely tested. 
