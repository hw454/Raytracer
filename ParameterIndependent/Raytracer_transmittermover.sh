#!/bin/bash
# Keith Briggs 2020-11-02 & Hayley Wragg
# bash Raytracer_transmittermover.sh

python ParameterLoad.py
<<<<<<< HEAD
for ((job=50; job<126; job++)); do
=======
for ((job=0; job<125; job++)); do
>>>>>>> 822d3c13752e69ba619a25e51c4b8e505e08e709
  echo "job=${job}"
  git pull git@github.com:/hw454/Raytracer.git
  time python RayTracerMainProgram.py "${job}"
  git pull git@github.com:/hw454/Raytracer.git
<<<<<<< HEAD
  git add Mesh Quality
  git commit -m "Mesh files to share with server"
  git push git@github.com:/hw454/Raytracer.git
done
=======
  git add Mesh Quality OptimisationResults
  git commit -m "Mesh files to share with server"
  git push git@github.com:/hw454/Raytracer.git
done
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
>>>>>>> 822d3c13752e69ba619a25e51c4b8e505e08e709

