import sys
sys.path.append("../NVF")
from dataprocessing.convert_to_scaled_off import to_off
from dataprocessing.boundary_sampling import boundary_sampling
from glob import glob
import configs.config_loader as cfg_loader
import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import os
from functools import partial

cfg = cfg_loader.get_config()
cls_idx_list = ["02958343", "03001627", "02691156", "03691459", "02828884", "04530566", "03636649", "04379243"] # [car, chair, plane, speaker, bench, watercraft, lamp, table]
print('Finding raw files for preprocessing.')

paths = []
for cls_idx in cls_idx_list:
	data_np = np.load(os.path.join(cfg.data_dir, "split_generalization_%s.npz"%cls_idx))
	for mode in ["train", "val", "test"]:
		for fn in data_np[mode]:
			paths.append(os.path.join(cfg.data_dir, fn, "model.obj"))
	print("Total number:", len(paths))


num_cpus = 4
def multiprocess(func):
	p = Pool(num_cpus)
	p.map(func, paths)
	p.close()
	p.join()

print('Start scaling.')
# multiprocess(to_off)
for cur_p in paths:
	to_off(cur_p)

print('Start distance field sampling.')
for sigma in cfg.sample_std_dev:
	print(f'Start distance field sampling with sigma: {sigma}.')
	multiprocess(partial(boundary_sampling, sigma = sigma))
	# this process is multi-processed for each path: IGL parallelizes the distance field computation of multiple points.
	# for path in paths:
	# 	boundary_sampling(path, sigma)

