#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

./ParameterLoad.py
for ((job=0; job<251; job++)); do
  echo "job=${job}"
  time python3 RayTracerMainProgram.py "${job}"
done
