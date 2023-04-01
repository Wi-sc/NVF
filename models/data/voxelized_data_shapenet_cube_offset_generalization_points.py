from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import traceback
import json

class VoxelizedDataset(Dataset):


    def __init__(self, mode, res, pointcloud_samples, data_path, split_file ,
                 batch_size, num_sample_points, num_workers, sample_distribution, sample_sigmas):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)

        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)
        with open('/home/xianghui_yang/NVF/dataprocessing/id2cat.json', 'r') as file:
            idx2cat = json.load(file)
        self.path = data_path
        self.data = []
        self.mode = mode
        for cls_idx in split_file:
            split = np.load(os.path.join(data_path, "split_generalization_%s.npz"%cls_idx))
            self.data.extend([(cls_idx, fn) for fn in split[mode]])
            print(mode, cls_idx, idx2cat[cls_idx], len(split[mode]))
        self.res = res

        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pointcloud_samples = pointcloud_samples

        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            
            cls, fn = self.data[idx]
            samples_path = os.path.join(self.path, cls, fn)
            
            voxel_path = os.path.join(samples_path, 'voxelized_point_cloud_{}res_{}points.npz'.format(self.res, self.pointcloud_samples))
            input = np.load(voxel_path)['point_cloud']

            if self.mode == 'test':
                return {'inputs': np.array(input, dtype=np.float32), 'fn' : fn, 'class': cls}

            coords = []
            offsets = []

            for i, num in enumerate(self.num_samples):
                boundary_samples_path = samples_path + '/boundary_{}_samples.npz'.format(self.sample_sigmas[i])
                boundary_samples_npz = np.load(boundary_samples_path)
                boundary_sample_points = boundary_samples_npz['query_points']
                boundary_sample_targets = boundary_samples_npz['target_points']
                subsample_indices = np.random.randint(0, len(boundary_sample_points), num)
                boundary_sample_points = boundary_sample_points[subsample_indices]
                boundary_sample_targets = boundary_sample_targets[subsample_indices]
                
                # boundary_sample_coords = boundary_sample_points.copy()
                # boundary_sample_coords = np.flip(boundary_sample_coords, axis=1)
                # boundary_sample_coords *= 2
                # points.extend(boundary_sample_points)
                coords.extend(boundary_sample_points)
                offsets.extend(boundary_sample_targets-boundary_sample_points)
                
            # assert len(points) == self.num_sample_points
            assert len(offsets) == self.num_sample_points
            assert len(coords) == self.num_sample_points
        except:
            print('Error with {}: {}'.format(fn, traceback.format_exc()))
            raise

        return {'grid_coords':np.array(coords, dtype=np.float32),'offsets': np.array(offsets, dtype=np.float32), 'inputs': np.array(input, dtype=np.float32), 'fn' : fn, 'class' : cls}

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
