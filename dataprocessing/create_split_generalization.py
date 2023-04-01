import os
import random
from glob import glob
import numpy as np
from collections import defaultdict

shapenetv1_dir = "/home/xianghui_yang/data/ShapeNet/ShapeNetCore.v1"
save_dir = "/home/xianghui_yang/data/NVF_data"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
cls_idx_list = ["02958343", "03001627", "02691156", "03691459", "02828884", "04530566", "03636649", "04379243"] # [car, chair, plane, speaker, bench, watercraft, lamp, table]
for cls_idx in cls_idx_list:
    train_all = []
    test_all = []
    val_all = []

    print('Finding raw files for preprocessing.')
    paths = glob(shapenetv1_dir+"/%s/*/model.obj"%cls_idx)
    paths = [os.path.dirname(p) for p in paths]
    

    all_samples = paths

    random.shuffle(all_samples)

    # Number of examples
    n_total = len(all_samples)

    n_val = 200
    n_test = 200

    if n_total < n_val + n_test:
        print('Error: too few training samples.')
        exit()

    n_train = n_total - n_val - n_test
    n_train_min = min(n_train, 3000)
    assert(n_train >= 0)

    # Select elements
    train_all.extend(all_samples[:n_train_min])
    val_all.extend(all_samples[n_train:n_train+n_val])
    test_all.extend(all_samples[n_train+n_val:])


    np.savez('%s/split_generalization_%s.npz'%(save_dir, cls_idx), train = train_all, test = test_all, val = val_all)
    print(cls_idx, 'processed.')