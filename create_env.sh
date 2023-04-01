# install chamfer distance
pip install git+'https://github.com/otaheri/chamfer_distance'
# install meshudf
cd external/custom_mc
python setup.py build_ext --inplace
# install earth mover distance
cd ../PyTorchEMD
python setup.py install
# install point-transformer
cd ../../models/lib/pointops
python setup.py install