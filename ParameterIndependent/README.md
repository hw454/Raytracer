This is the latest ray-launcher in my project. Previous versions all took a standard approach by computing fields on segments and
adding values to grid cells.
This version stores information to compute the power later, this makes the output more generalised and should enable sophisticated
optimisation methods.
To run the launcher run
"RayTracerMainProgram.py" this will save the following:
- Mesh/ -RayMeshPoints'+(Nra)+'Refs'+(Nre)+'m.npy' replace '+(Nra)+' with the total number of rays and '+(Nre)+' with the number
        of reflections.
	This file is the array containing the intersection points of the rays in the form.
	[[x0,y0,z0,nob0],...,[xNre,yNre,zNre,nobNre],...,[x(Nre*Nra),y(Nre*Nra),z(Nre*Nra),nob(Nre*Nra)]].
	This is the file you need to plot the lines showing ray trajectories.
        -DSM'+(Nra)+'Refs'+(Nre)+'m.npy' replace '+(Nra)+' with the total number of rays and '+(Nre)+' with the number
        of reflections.
	This is a sparse mesh containing the lengths of rays and their reflection angles in positions corresponding to ray
	number, reflection number and obstacle number.
        -Power_Grid'+(Nra)+'Refs'+(Nre)+'m.npy replace '+(Nra)+' with the total number of rays and '+(Nre)+' with the number
        of reflections.
	This is a 3D grid with a value in each position corresponding to the power.
	This is specific to the inputs in ParameterInput.py when the program was run.
- GeneralMethodPowerFigures/
        -PowerSlice'+(i)+'Nra'+(Nra)+'n'+(n)+'Nref'+Nre+'.eps replace '+(i)+' with the z term you want, '+(Nra)+' with the total
	number of rays '+(n)+' with the total number of z terms and '+(Nre)+'with the number of reflections.

Status:
This version computes the DSM for the Raylauncher but functions applied to the DSM are still being tested.
The power can be computed and looks appropriate but this is not tested.