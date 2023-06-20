#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

python ParameterLoad.py
for ((job=150; job<200; job++)); do
  echo "job=${job}"
  time python RayTracerMainProgram.py "${job}"
done
