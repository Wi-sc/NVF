conda env create -f NDF_env.yml
conda activate NDF
pip install git+'https://github.com/otaheri/chamfer_distance'
cd external/custom_mc
python setup.py build_ext --inplace
cd ../PyTorchEMD
python setup.py install
cd ../../models/lib/pointops
python setup.py install