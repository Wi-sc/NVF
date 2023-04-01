import trimesh
import igl
import numpy as np
import glob
import multiprocessing as mp
from multiprocessing import Pool
import os
import traceback
from functools import partial
import random
import gc
import configs.config_loader as cfg_loader

# number of distance field samples generated per object
sample_num = 100000
SAVE_DIR = "/home/xianghui_yang/data/NVF_data/"
def boundary_sampling(path, sigma):
    try:
        cat_idx = path.split("/")[-3]
        file_name = path.split("/")[-2]
        out_path = SAVE_DIR + cat_idx
        
        input_file = os.path.join(out_path, file_name + '.obj')
        if not os.path.exists(os.path.join(out_path, file_name)):
            os.mkdir(os.path.join(out_path, file_name))
        out_file = os.path.join(out_path, file_name, 'boundary_{}_samples.npz'.format(sigma))

        if os.path.exists(out_file):
            print('Exists: {}'.format(out_file))
            return

        mesh = trimesh.load(input_file)
        points = mesh.sample(sample_num)

        if sigma == 0:
            boundary_points = points
        else:
            boundary_points = points + sigma * np.random.randn(sample_num, 3)

        target_points = igl.signed_distance(boundary_points, mesh.vertices, mesh.faces)[2]


        # np.savez(out_file, points=boundary_points, df = df, grid_coords= grid_coords)
        np.savez(out_file, query_points=boundary_points, target_points=target_points)
        print('Finished: {}'.format(out_file))

    except:
        print('Error with {}: {}'.format(out_file, traceback.format_exc()))


    # del mesh, df, boundary_points, grid_coords, points
    del mesh, boundary_points, target_points, points
    gc.collect()
