#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

python ParameterLoad.py
<<<<<<< HEAD
for ((job=0; job<125; job++)); do
||||||| merged common ancestors
for ((job=0; job<126; job++)); do
=======
for ((job0; job<126; job++)); do
>>>>>>> 31da2501208146bab07037c64507338bac6bd44c
  echo "job=${job}"
  git pull git@github.com:/hw454/Raytracer.git
  time python RayTracerMainProgram.py "${job}"
<<<<<<< HEAD
  git pull git@github.com:/hw454/Raytracer.git
  git add Mesh Quality OptimisationResults
  git commit -m "Mesh files to share with server"
  git push git@github.com:/hw454/Raytracer.git
||||||| merged common ancestors
=======
  git add Mesh Quality
  git commit -m "Mesh files to share with server"
  git push git@github.com:/hw454/Raytracer.git
>>>>>>> 31da2501208146bab07037c64507338bac6bd44c
done
<<<<<<< HEAD
git pull git@github.com:/hw454/Raytracer.git
time python RayTracerPlottingFunctions.py
git pull git@github.com:/hw454/Raytracer.git
git add Quality OptimisationResults
git commit -m "Mesh files to share with server"
git push git@github.com:/hw454/Raytracer.git
git pull git@github.com:/hw454/Raytracer.git
time python OptimisationMethod.py
git pull git@github.com:/hw454/Raytracer.git
git add Quality OptimisationResults
git commit -m "Mesh files to share with server"
git push git@github.com:/hw454/Raytracer.git

||||||| merged common ancestors
=======
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
>>>>>>> 31da2501208146bab07037c64507338bac6bd44c
