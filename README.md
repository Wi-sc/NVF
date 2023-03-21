# NVF
Official implementation of neural vector fields

# Requirements
The requirements are PyTotch and Pytorch3D with cuda support:
A linux system with cuda 10 is required for the project.

Please clone the repository and navigate into it in your terminal, its location is assumed for all subsequent commands.

# Installation
The nvf.yml file contains all necessary python dependencies for the project. To conveniently install them automatically with anaconda you can use:

conda env create -f NDF_env.yml
conda activate NDF
pip install git+'https://github.com/otaheri/chamfer_distance'
cd external/custom_mc
python setup.py build_ext --inplace
cd ../../