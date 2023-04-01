conda env create -f nvf.yml
conda activate nvf
pip install git+'https://github.com/otaheri/chamfer_distance'
cd external/custom_mc
python setup.py build_ext --inplace
cd ../PyTorchEMD
python setup.py install
cd ../../models/lib/pointops
python setup.py install