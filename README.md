# Reachability Analysis for Neural Agent-Environment Systems

This repository contains all the code necessary to replicate the findings
described in the paper "Reachability Analysis for Neural Agent-Environment Systems" (Submission #2560),
forked from the code used in [Piecewise Linear Neural Networks Verification: A
comparative study](https://arxiv.org/abs/1711.00455).


Note that all tools have already been compiled in this archive, with results in the `./results` directory,
so users may skip to section "Analyse the results", unless generating own datasets/results.


In addition, this also contains conversion scripts to operate other solvers,
included as submodules. If you make use of them, please cite the corresponding paper.
* Reluplex, in `./ReluplexCav2017`
* Planet, in `./Planet`


## Structure of the repository
* `./planet/` is a git submodule, linking
  to [the official Planet repository](https://github.com/progirep/planet)
* `./ReluplexCav2017/` is a git submodule, linking to a fork
of
[the official Reluplex repository](https://github.com/guykatzz/ReluplexCav2017).
The fork was made to include some additional code to support additional
experiments that the originally included ones.
* `./plnn/` contains the code for the MIP solver and the BaB solver.
* `./tools/` is a set of python tools used to go from one solver's format to
  another, run a solver on some property, compare experimental results, or
  generate datasets.
* `./scripts/` is a set of bash scripts, instrumenting the tools of `./tools` to
  reproduce the results of the paper.
  
## Running the code
### Dependencies
The code was implemented assuming to be run under `python3.6`.
We have a dependency on:
* [The Gurobi solver](http://www.gurobi.com/) to solve the LP arising from the
Network linear approximation and the Integer programs for the MIP formulation.
Gurobi can be obtained
from [here](http://www.gurobi.com/downloads/gurobi-optimizer) and academic
licenses are available
from [here](http://www.gurobi.com/academia/for-universities).
* [Pytorch](http://pytorch.org/) to represent the Neural networks and to use as
  a Tensor library. 
* The python packages `psutil` to measure memory usage in our benchmarks and
`sh` to instrument other solvers. 
* **Reluplex** and **Planet** have their own dependency, described in their
  Readme page.
  
### Installing everything
We recommend installing everything into a python virtual environment.

```bash
git clone --recursive https://github.com/oval-group/PLNN-verification.git

cd PLNN-verification
virtualenv -p python3.6 ./venv
./venv/bin/activate

# Install gurobipy to this virtualenv
# (assuming your gurobi install is in /opt/gurobi701/linux64)
cd /opt/gurobi701/linux64/
python setup.py install
cd -

# Install pytorch to this virtualenv
# (or check updated install instructions at http://pytorch.org)
pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp36-cp36m-manylinux1_x86_64.whl 

# Install psutil
pip install psutil

# Install the code of this repository
python setup.py install

# Additionally, install the code for Planet and Reluplex
# by cd-ing into their directory and following their 
# installation instructions.
## Reluplex:
cd ReluplexCav2017/glpk-4.60
./configure_glpk.sh
make
make install
cd ../reluplex
make
cd ../check_properties
make
cd ../..

## Planet
cd planet/src
# sudo apt install valgrind qt5-qmake libglpk-dev # if necessary 
qmake
make
# if you encounter linker issues, move -lsuitesparseconfig to the end of the flag list
cd ../..
```

### Running the experiments
If you have setup everything according to the previous instructions, you should
be able to replicate the experiments of the paper. To do so, follow the
following instructions:

```bash
## Generate the datasets
# Generate the .rlv (planet/BaB/MIP inputs file from the Acas .nnet files)
./scripts/convertACAS2rlv.sh

# Generate the .nnet files (reluplex inputs) from the CollisionDetection .rlv files
./scripts/convertrlv2rlpx.sh

# Generate the .rlv and .nnet files for the TwinStream dataset
./scripts/generate_twin_ladder_benchmarks.sh

## Generate the results
./scripts/bab_runscript.sh
./scripts/mip_solver_runscript.sh
./scripts/reverify_solver_runscript.sh
./scripts/planet_runscript.sh
./scripts/reluplex_runscript.sh

## Analyse the results
# (might have to `pip install matplotlib` to generate curves)
./scripts/generate_analysis_images.sh
# TwinStream comparison
./tools/compare_benchmarks.py results/twinLadder/reluplex/ results/twinLadder/planet/ results/twinLadder/MIP/ results/twinLadder/BaB results/twinLadder/reverify --all_unsat
```


