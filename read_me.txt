Hello thank you for reading

In case release mode doesn't work for some reason
Please do:

In Configuration Properties: c/c++ -> General -> Additional Include DIrectories => add ImGUi, Include : Linker-> Input-> Additional Dependencies => add opengl32.lib, glfw3.lib : Linker -> general-> Additional LIbrary Directories => add libs/

There might be a Firewall issue maybe? one of my teammates Bill had issue trying run the code.

If OpenMp requires to be turned on for some reason. Solution explorer->properties->C/C++->Language and turn on OpenMp Support to Yes

For trying other obj and its respective mtl files please make sure its in /Wind_sim-masters and not in old_obj_files. The eye will have to adjusted for it though there are many obj that exist in there, but require a lot of tuning.

Started project based off lab1 and so Sphere-stacks exists. I just used the main to start off and deleted everything on it so I didnt need import imGui and other GLFW.

All functions are commented and some are commented out because its old code that is still useful to check on and test against optimized code to prove optimization.


General Summary of code above render loop


Creates the model for arrow

Gets all obj object's verticies and normals.

variables to use in generate arrow function (use this if you want play around
cube_surround_other_cube(arrows, yLayers, 220, numArrowsPerLayer); 
(explained in code how to use)

mouse and key calls for camera



General summary of code in render loop for SPH and intersection

Set mass
find neighboring particles (finds density and pressure also)

for loop {
find all forces and place in vector of vec3 call force for later
}

do some parallel threads for interesction checking
get vector of struct that hold info for reflect from the parallel thread in collision
Do reflect and move the velocity and position if needed based on excess energy

render the model

color the model

update the particle velocity and position in the vector of arrows, particles.

translate and rotate based off arrows.direction, (glm::normalize(velocitty), and arrows.position and then render the arrow. 

start over

Thank you for reading.
