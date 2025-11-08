### Yu Shuai's source code for MAE557 Assignment 3

### 1. Prerequisities (tested on Adroit)

language: c++ (c++17)
compiler: g++ (GCC) 11.5.0 from GNU
library: eigen3 3.4.0

### 2. Compiling and running

in Adroit terminal, type the following for compilation:

g++ -std=c++17 -I/usr/include/eigen3 solver.cpp -o your_name

then use the slurm batch script to execute the program with customized parameters for simulation

sbatch your_name.sh (in our example, it's 2d.sh)

### 3. Postprocessing

After running the codes, there will be folders generated named such as:

output_Ma_###_Nx_###_dt_###

These folders include .txt files for density, velocity components and temperature that can be used for postprocessing.

