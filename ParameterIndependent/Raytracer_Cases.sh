#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

# Start with no internal Obstacles and Perfect Reflections on.
# Parallel plates
python ParameterLoad.py
python RayTracerMainProgram.py "${job}"
python ParameterLoadObstacle.py         # Add Obstacle
python RayTracerMainProgram.py "${job}"
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
python RayTracerMainProgram.py "${job}"
python ParameterLoadObstacle.py         # Add Obstacle
python RayTracerMainProgram.py "${job}"
# Repeat but turn Perfect Reflection Off
python ParameterLoadPerfOff.py
python RayTracerMainProgram.py "${job}"
python ParameterLoadObstacle.py         # Add Obstacle
python RayTracerMainProgram.py "${job}"
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
python RayTracerMainProgram.py "${job}"
python ParameterLoadObstacle.py         # Add Obstacle
python RayTracerMainProgram.py "${job}"
# Single plates
python ParameterLoadPerfOnSinglePlate.py
python RayTracerMainProgram.py "${job}"
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
python RayTracerMainProgram.py "${job}"
# Repeat but turn Perfect Reflection Off
python ParameterLoadPerfOff.py
python RayTracerMainProgram.py "${job}"
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
python RayTracerMainProgram.py "${job}"
# Repeat but move transmitter
# Start with no Obstacles and Perfect Reflections on.
python ParameterLoadPerfOnMoveTx.py
python RayTracerMainProgram.py "${job}"
python ParameterLoadObstacle.py         # Add Obstacle
python RayTracerMainProgram.py "${job}"
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
python RayTracerMainProgram.py "${job}"
python ParameterLoadObstacle.py         # Add Obstacle
python RayTracerMainProgram.py "${job}"
# Repeat but turn Perfect Reflection Off
python ParameterLoadPerfOff.py
python RayTracerMainProgram.py "${job}"
python ParameterLoadObstacle.py         # Add Obstacle
python RayTracerMainProgram.py "${job}"
python ParameterLoadRefsNoOb.py         # Remove Obstacle, Increase Reflections
python RayTracerMainProgram.py "${job}"
python ParameterLoadObstacle.py         # Add Obstacle
python RayTracerMainProgram.py "${job}"
# Single plates
python ParameterLoadPerfOnSinglePlate.py
python RayTracerMainProgram.py "${job}"
python ParameterLoadRefsNoOb.py         #  Increase Reflections
python RayTracerMainProgram.py "${job}"
# Repeat but turn Perfect Reflection Off
python ParameterLoadPerfOff.py
python RayTracerMainProgram.py "${job}"
python ParameterLoadRefsNoOb.py         #  Increase Reflections
python RayTracerMainProgram.py "${job}"
