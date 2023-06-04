# Neural Vector Fields (NVF):
<img src="./vis/reconstruction_black.gif" data-canonical-src="./vis/reconstruction_black.gif" width="1200" height="700" />
<img src="./vis/deform.gif" data-canonical-src="./vis/reconstruction_black.gif"/>

Official implementation of Neural Vector Fields (NVF). Feel free to use this code for academic work, but please cite the following:
```
@misc{yang2023neural,
      title={Neural Vector Fields: Implicit Representation by Explicit Learning}, 
      author={Xianghui Yang and Guosheng Lin and Zhenghao Chen and Luping Zhou},
      year={2023},
      eprint={2303.04341},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
[CVF](https://openaccess.thecvf.com/content/CVPR2023/html/Yang_Neural_Vector_Fields_Implicit_Representation_by_Explicit_Learning_CVPR_2023_paper.html)
[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Neural_Vector_Fields_Implicit_Representation_by_Explicit_Learning_CVPR_2023_paper.pdf) -
[Supplementaty](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Yang_Neural_Vector_Fields_CVPR_2023_supplemental.pdf) -
[Arxiv](https://arxiv.org/abs/2303.04341) -
Published in CVPR 2023.

## Requirements
The codes have been tested on the linux Ubuntu 20.04 system with NVIDIA RTX3090ti. The enviroment inlcude:
* Python=3.9
* Pytorch=1.12.1
* Torchvision=0.13.1
* CUDA=11.6
  
Please clone the repository and navigate into it in your terminal, its location is assumed for all subsequent commands.

## Installation
The `nvf.yml` file contains all necessary python dependencies for the project. You can create the anaconda environment using: 
```
conda env create -f nvf.yml
conda activate nvf
```
There are other dependencies to be installed.

1. To train the model, relavattive operations in [point-transfomrer](https://github.com/POSTECH-CVLab/point-transformer) are needed. 
2. To extract meshes, [meshUDF](https://github.com/cvlab-epfl/MeshUDF) is needed. 
3. To test the model, libaries for [chamfer distance](https://github.com/otaheri/chamfer_distance), [earth mover distance](https://github.com/daerduoCarey/PyTorchEMD) are needed. 

The commands have been incorperated by `create_env.sh`. You can install them via runing the script:
```
bash create_env.sh
```
Or you can install step by step by yourself.

## Data Preparation
First, create a configuration file in folder `configs/`, use `configs/shapenet_cube_offset_generalization_pt_vq_k16.txt` as reference and see configs/config_loader.py for detailed explanation of all configuration options. Change the desired data directory with variable `SAVE_DIR` in `create_split_generalization.py`, `convert_to_scaled_off.py ` and `boundary_sampling.py`. Please also make sure to move the split files `split_generalization_*.npz` from `dataprocessing` to your desired `SAVE_DIR`.

Next, prepare the data for NVF using
```
python dataprocessing/preprocess.py --config configs/shapenet_cube_offset_generalization_pt_vq_k16.txt
```
You can generate a random test/training/validation split of the data using
```
python dataprocessing/create_split.py --config configs/shapenet_cube_offset_generalization_pt_vq_k16.txt
```
but replacing `configs/shapenet_cube_offset_generalization_pt_vq_k16.txt` in the commands with the desired configuration.

## Training
To train your NVF, you can change the parameters in the configs and run:
```
python train_generalization.py --config ./configs/${exp_name}.txt 2>&1|tee ${save_dir}/log.txt
```
In the `experiments/ `folder you can find an experiment folder containing the model checkpoints, the checkpoint of validation minimum, and a folder containing a tensorboard summary, which can be started at with
```
tensorboard --logdir experiments/${exp_name}/summary/ --host 0.0.0.0
```
## Generation
To generate meshes after training:
```
python generation.py
```
Please specify the desired model before running.
## Test
To test results after generation:
```
python test.py
```
Please specify the desired experiment name before running.

## Contact
For questions and comments please leave your questions in the issue or contact Xianghui Yang via email xianghui.yang@sydney.edu.au.

## Acknowledge
The code is modified from the [NDF](https://github.com/jchibane/ndf). Thanks for open-sourcing.

