This repository contains all the Raytracer versions.

ParameterIndependent:
The current Raytracer uses a method for iterating along rays and storing information then outputs a mesh. This mesh is then input into another function to compute power.
To run the program run "RayTracerMainProgram.py".
To change any parameters edit "ParameterInput.py"
- To edit the ray tracer parameters (number of rays, number of grid points, number of reflections, length scale) edit "DeclareParameters()".
- To edit the geometry edit "DeclareParameters()"
- To edit obstacle coefficients edit "ObstacleCoefficients()"
- To edit antenna gains or wavelength edit "ObstacleCoefficients()"

Status: Currently the program is computing the same value at all grid points. This is not the intention of the code. I have created a "__self_eq__()" for the DS object in "DictionarySparseMatrix.py" this checks if all matrices in each grid point are equal. Using this function I have determined that the problem exists at the output of the Mesh from "MeshProgram()" in "RayTracerMainProgram.py". I have not yet found the cause within the program of why these are all the same.
