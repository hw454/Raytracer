#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

python ParameterLoad.py
for ((job=50; job<126; job++)); do
  echo "job=${job}"
  git pull git@github.com:/hw454/Raytracer.git
  time python RayTracerMainProgram.py "${job}"
  git pull git@github.com:/hw454/Raytracer.git
  git add Mesh Quality
  git commit -m "Mesh files to share with server"
  git push git@github.com:/hw454/Raytracer.git
done

