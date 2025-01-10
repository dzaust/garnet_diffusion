# About the Project
This program is designed to address the transient diffusion of species within container minerals.  
The code has been adapted from the Step-26 tutorial program of deal.II, which was originally designed to solve the heat conduction equation.

# Steps for usage
The compiled executable files (MacOS) for two-dimensional and three-dimensional diffusion can be found in the bin folder.  
All that needs to be done is to create a mesh file and configure the parameters for diffusion in the diffusion_equation.prm file.  
The user does not need to modify the source code unless the existing functionality does not meet their requirements.

## Prerequisites
Users are required to install the deal.II library on either macOS or Linux.  
For detailed instructions, please refer to the official documentation available in the deal.II README at https://www.dealii.org/current/readme.html.
## Step 1: Create the Mesh File
Create a mesh file with a .msh extension, for example, using Gmsh. Place the file into the mesh folder. 

The example mesh files in the folder include:  
•	sphere.msh: a mesh for a sphere used for accuracy analysis in the text.  
•	2d.msh and 3d.msh: meshes used in Figure 5.
## Step 2: Set the Parameters
Edit the diffusion_equation.prm file to configure the necessary parameters.  
The current setup is designed for isothermal diffusion with a duration of 10 million years (see Figure S1b).  

Notation:  
•	For a two-dimensional setup, use the following variable names:  

    set Variable names = x, y, t
•	For a three-dimensional setup, use the following variable names:  

    set Variable names = x, y, z, t
## Step 3: Execute the Program
To run the program, please execute the following commands in the terminal:  
•	For a three-dimensional simulation, use the command:  

    ./bin/garnet_diffusion_3d
•	For a two-dimensional simulation, use the command:  

    ./bin/garnet_diffusion_2d
## Step 4: Postprocessing
The results will be written to a VTK file, which can be processed using ParaView.

# Compile the Program (CMake Required)
If any functionality in the source code has been added or modified, the program must be recompiled.  
In the terminal, execute the following commands:

    cd path/to/the/code
    rm -r build
    cmake -DCMAKE_BUILD_TYPE:STRING=Release -B./build
    cmake --build ./build
Alternatively, if you are using Visual Studio Code, you can simply press the build button provided by the CMake Tools extension.  

Upon successful compilation, the executable file (garnet_diffusion) will be located within the build directory.
