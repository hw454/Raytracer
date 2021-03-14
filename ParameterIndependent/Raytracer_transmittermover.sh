#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

python ParameterLoad.py
for ((job0; job<126; job++)); do
  echo "job=${job}"
  git pull git@github.com:/hw454/Raytracer.git
  time python RayTracerMainProgram.py "${job}"
  git add Mesh Quality
  git commit -m "Mesh files to share with server"
  git push git@github.com:/hw454/Raytracer.git
done
git pull git@github.com:/hw454/Raytracer.git
python RayTracerPlottingFunctions
git add OptimisationResults Quality GeneralMethodPowerFigures
git commit -m "Share optimal transmitter locations"
git push git@github.com:/hw454/Raytracer.git
git pull git@github.com:/hw454/Raytracer.git
python OptimisationMethod.py
git add Quality OptimisationResults
git commit -m "Share optimisation results"
git psuh git@github.com:/hw454/Raytracer.git