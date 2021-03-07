#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

# Start with no internal Obstacles and Perfect Reflections on.
# Start with Corner Plates
python ParameterLoad.py
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
python ParameterLoadObstacle.py         # Add Obstacle
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
python ParameterLoadObstacle.py         # Add Obstacle
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
# Repeat but turn Perfect Reflection Off
python ParameterLoadPerfOff.py
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
python ParameterLoadObstacle.py         # Add Obstacle
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
python ParameterLoadObstacle.py         # Add Obstacle
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
# Single plates
python ParameterLoadPerfOnSinglePlate.py
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
# Repeat but turn Perfect Reflection Off
python ParameterLoadPerfOff.py
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
./Raytracer_transmittermover.sh
python RayTracerPlottingFunctions.py
python OptimisationMethod.py
