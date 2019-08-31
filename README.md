# diss

## INSTALLATION GUIDE

To run the notebooks and/or libraries in this repository it is necessary to install some
packages. To make it easier, two scripts are provided to create a Conda environment 
installs said dependencies.

Firstly, it necessary to have Conda installed (see [HERE](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
for a tutorial on how to install Conda). Once Conda is available run:
```
conda env create -f conda_environment.yml
conda activate diss
pip install ./libraries
```
that will first create an environment named "diss", then it will activate it and install in it the
libraries in the repository ("isaac", "simulator" and "toddler"). Once this is done, the environment
should be ready to run the notebooks. In case you want to install the libraries in developer mode
substitute the last line by `pip install -e ./libraries`

To remove the environment and all its installed libraries execute:
```
conda deactivate
conda remove -y --name diss --all
```
