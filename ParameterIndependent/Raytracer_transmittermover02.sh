#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

python ParameterLoad.py
for ((job=0; job<10; job++)); do
  echo "job=${job}"
  time python RayTracerMainProgram.py "${job}" 
done
