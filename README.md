# Discovering Symbolic Models from Deep Learning with Inductive Biases - Final Project

## Description

This project contains the code for the reproduction of the paper "Discovering Symbolic Models from Deep Learning with Inductive Biases" by Cranmer et al. (2020). The code is implemented inPython and uses the PyTorch Geometric library for the implmentation of single message passing Graph Network architecture, and the PySR library for symbolic regression. The main goal of this reproduction work is to demonstrate the intersection of symbolic regression and deep learning to extract the physical intuition behind black-box models. In the pursuit of this, a latent  

A Slurm bash script is included for easy submission and execution of the project on the CSD3 computer cluster.

## Installation
The first step is to login to CSD3, and clone the remote GitLab repository on your account. From the root directory, run the following:

```bash
make clean
```
This will remove any build function that might be present. Next, we want to make sure the correct modules, especially MPI, are loaded in the current HPC environment before we compile any code. Run the following command to load the OpenMPI:

```bash
module load openmpi
```
To ensure the environment is using OpenMPI instead of Intel's MPI, we want to check `mpicxx`:

```bash
which mpicxx
```

This should give a path that has OpenMPI in it instead of Intel's MPI. If this is not the case, repeat the above steps. Next, we want to build the project using CMake. The command to build the project is:

```bash
make build
```
From this step, there are a couple of options to compile different build files. The options are given below

## Release Mode

```bash
make release
```
This mode is generally preferred due to -O3 optimization level. It should be used as the default method for compiling the project unless debugging or testing is required.

## Debug Mode

```bash
make debug
```
This mode is used for debugging, profiling, or any memory analysis of the project. It uses the debug flag when compiled.

## Docs Mode
```bash
make docs
```
This is for generating the documentation of the project. After it is run, a directory named `docs` is created in the root level directory. Inside this directory, navigate to html folder and open the index.html file for the documentation.

## Test Mode
```bash
make run_tests
```
This mode is for running the `GoogleTest` framework that contains 8 testing functions for the grid class methods.

## Clean
```bash
make clean
```
This command should be used anytime a new change is made to the code or before any compilation followed by a previous one.


## Data Generation

The simulated data for the project is generated using the `src/simulate_systems.py` script. The results of this script is saves inside the `data` directory. The repository already contains previously simulated data for all the systems that were used for the study. However, one can re-run the simulations by running the following command from the root directory:

```bash
python src/simulate_systems.py
```

The `sim` variable inside this script determines the type of the simulation, and the `N` variable determines the number of simulations for each type. The simulations are saved in the `.npy` format.

## Training of the Graph Network

## Edge Model Analysis

## Symbolic Regression




## Submit a Job
An example submission script is provided in the project named `slurm_submit.sh`. This script can be used to submit a job to run the executable `game` with the command:

```bash
sbatch slurm_submit.sh
```
Note that this script should be adjusted for desired number of ranks and number of threads per rank. The full path to application executable should also be adjusted with correct account name and build file used. For example for a code compiled with release mode the correct path is:

`application="/home/CRSID/build/release/src/game"`

The run options should be adjusted according to the three arguments passed with the executable. The first argument is either an integer for the size of the square grid or the name of the `.txt` file used as an input grid. If a text file is used as the input, it should be placed in the `grids` directory before. The second argument is the number of time steps for the grid updates. The third argument is either `"visualize"` to print the final grind on the terminal or `"save"` which will save the final grid inside the `grids` directory with name `final_grid.txt`. Example options are:

```
glider_81x81.txt 100 visualize
```
```
10000 100 save
```




## Analysis Jobs
The other two scripts `slurm_submit.mpiranks.sh` and `slurm_submit_mpivsopenmp.sh` are for parallelization analyses of the project that runs `run_ranks.py` and `mpi_openmp.py` python scripts respectively. `run_ranks.py` script internally calls the executable with different number of MPI processes and saves the execution time for each in a `.txt` file. This `.txt` file can then be moved to `graphs` directory where a python script `plot.graphs.py` can be run to generate the plot used in the report. 
`mpi_openmp.py` script internally calls the executable with different combinations of MPI ranks and OpenMP threads and saves the execution times in a `.txt` file. The table in the report was made using this file.



## The use of generative A.I.

ChatGPT 3.5 was used for suggesting alternative wordings, grammar suggestions, and proofreading while writing the report. The following prompts were used during this process

```
How can I write this equation in LaTeX format? <input equation>
```
```
Provide BibTex citation format for this website <input website>
```
```
What is the rule of 5 in C++ and how important it is in class definitions?
```
```
What is another word for <input word>?
```
```
Is this sentence grammatically correct? <input sentence>
```
```
Is this paragraph clear for a reader? <input paragraph>
```
```
How to rephrase this sentence to make it more clear? <input sentence>
```
For the report writing, the outputs were partially adapted in a way that only alternative wordings were used and not the whole output while rephrasing the Summary and Introduction parts of the report. The LaTex code for equations was adapted from the suggestion. The suggestions for BibTeX citations are also adapted. 

Furthermore, the suggestions from the autocomplete feature of `GitHub Copilot` were utilized during the documentation of the software, and code development of the project in below areas:


```
Test functions for the Grid class methods contained in the `test_grid.cpp` were developed with the help of Copilot suggestions, mainly the example grids were adapted.
```
```
Python scripts `mpi_openmp.py` and `run_ranks.py` to generate run-time analysis data used in the parallelization section was developed using the suggestions from Copilot when implementing the subprocess call.
```

```
Indexing and logic used in the `reorganizeGrid` and `inverseReorganizeGrid` functions were developed by adapting suggestions from Copilot.
```

## Authors
Sabahattin Mert Daloglu

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) for details.
