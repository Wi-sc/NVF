import trimesh
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import models.local_model_offset_point_transformer_vq as model
import models.data.voxelized_data_shapenet_cube_offset_generalization_points as voxelized_data
import sys
sys.path.append("./external/custom_mc")
from _marching_cubes_lewiner import udf_mc_lewiner

class Generator():
    def __init__(self, model, cuda_device, bouding_box=0.55, step_num=256, points_batch_size=200000):
        self.model = model
        self.device = cuda_device
        self.points_batch_size = points_batch_size
        self.step_num = step_num
        
        p = torch.linspace(-bouding_box, bouding_box, step_num)
        px, py, pz  = torch.meshgrid([p, p, p])
        points = torch.stack([px, py, pz], 3)
        points = points.view(-1, 3)
        self.points = points
    
    def generate_mesh(self, inputs):
        # feature_planes = self.model.encode_inputs(inputs)
        encoding = self.model.encoder(inputs)
        p_split = torch.split(self.points, self.points_batch_size)
        perd_dist_all = []
        perd_offset_all = []
        for pi in p_split:
            samples = pi.unsqueeze(0).clone().to(self.device)
            with torch.no_grad():
                perd_offset, _, _ = self.model.decoder(samples, inputs, encoding)
                perd_dist = torch.sqrt(torch.sum(torch.pow(perd_offset, 2), 2))
                perd_offset = perd_offset/perd_dist.unsqueeze(-1)
            perd_dist_all.append(perd_dist.squeeze(0).squeeze(-1).detach().cpu())
            perd_offset_all.append(perd_offset.squeeze(0).detach().cpu())

        perd_dist_all = torch.cat(perd_dist_all, dim=0)
        perd_offset_all = torch.cat(perd_offset_all, dim=0)
        
        perd_dist_all = perd_dist_all.reshape(self.step_num, self.step_num, self.step_num)
        perd_offset_all = perd_offset_all.reshape(self.step_num, self.step_num, self.step_num, 3)
        print(perd_dist_all.shape, perd_offset_all.shape)
        N = perd_dist_all.shape[0]
        voxel_size = 1.1 / (N - 1)
        verts, faces, _, _ = udf_mc_lewiner(perd_dist_all.cpu().numpy(), perd_offset_all.detach().cpu().numpy(), spacing=[voxel_size] * 3)
        print(faces.min(), faces.max())
        verts = verts - 0.55 # since voxel_origin = [-1, -1, -1]
        if(verts.shape[0]<10 or faces.shape[0]<10):
            print('no sur---------------------------------------------')
            return

        verts_torch = torch.from_numpy(verts).float().cuda()
        print(verts_torch.shape, torch.any(torch.isnan(verts_torch)))
        with torch.no_grad():
            p_split = torch.split(verts_torch, self.points_batch_size)
            pred_df_verts = []
            for pi in p_split:
                # perd_offset = self.model.decoder(pi.unsqueeze(0), *encoding)
                perd_offset, _, _ = self.model.decoder(pi.unsqueeze(0), inputs, encoding)
                perd_dist = torch.sqrt(torch.sum(torch.pow(perd_offset, 2), 2))
                pred_df_verts.append(perd_dist.squeeze(0))
                # df_pred = torch.clamp(self.model.decoder(pi.unsqueeze(0), *encoding), max=0.1)
                # pred_df_verts.append(df_pred.squeeze(0))

        pred_df_verts = torch.cat(pred_df_verts, dim=0).unsqueeze(1)
        pred_df_verts = pred_df_verts.cpu().numpy()
        print("pred_df_verts:", pred_df_verts.shape)
        # Remove faces that have vertices far from the surface
        filtered_faces = faces[np.max(pred_df_verts[faces], axis=1)[:,0] < voxel_size / 3]
        print(verts.shape)
        filtered_mesh = trimesh.Trimesh(verts, filtered_faces)
        # return filtered_mesh
        ### 4: clean the mesh a bit
        # Remove NaNs, flat triangles, duplicate faces
        filtered_mesh = filtered_mesh.process(validate=False) # DO NOT try to consistently align winding directions: too slow and poor results
        filtered_mesh.remove_duplicate_faces()
        filtered_mesh.remove_degenerate_faces()
        # Fill single triangle holes
        filtered_mesh.fill_holes()

        filtered_mesh_2 = trimesh.Trimesh(filtered_mesh.vertices, filtered_mesh.faces)
        # Re-process the mesh until it is stable:
        n_verts, n_faces, n_iter = 0, 0, 0
        while (n_verts, n_faces) != (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces)) and n_iter<10:
            filtered_mesh_2 = filtered_mesh_2.process(validate=False)
            filtered_mesh_2.remove_duplicate_faces()
            filtered_mesh_2.remove_degenerate_faces()
            (n_verts, n_faces) = (len(filtered_mesh_2.vertices), len(filtered_mesh_2.faces))
            n_iter += 1
            filtered_mesh_2 = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)

        filtered_mesh = trimesh.Trimesh(filtered_mesh_2.vertices, filtered_mesh_2.faces)
        return filtered_mesh


MODEL_DIR = '/home/xianghui_yang/ndf/experiments/shapenet_cube_offset_generalization_pt_k16/'
best_model = "checkpoint_279_113h:0m:32s_406832.6937339306.tar"
MODEL_NAME = 'checkpoints/%s'%best_model
OUTPUT_DIR = os.path.join(MODEL_DIR, 'mesh')
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

net = model.NDF(k=16, pos_dim=128, out_dim=128)
device = torch.device("cuda:0")
net = net.to(device)
model_path = os.path.join(MODEL_DIR, MODEL_NAME)
checkpoint = torch.load(model_path)
epoch = checkpoint['epoch']
training_time = checkpoint['training_time']
print("Load model:", model_path)
print("Epoch:", epoch)
net.load_state_dict(checkpoint['model_state_dict'], strict=True)
net.eval()
for param in net.parameters():
    param.requires_grad = False
generator = Generator(net, device)

dataset = voxelized_data.VoxelizedDataset('test',
                                          res=256,
                                          pointcloud_samples=10000,
                                          data_path="/home/xianghui_yang/data/NVF_data/",
                                          split_file=["02958343", "03001627", "02691156", "04379243", "02828884", "04530566", "03636649", "03691459"],
                                          batch_size=1,
                                          num_sample_points=50000,
                                          num_workers=8,
                                          sample_sigmas=[0.08, 0.02, 0.003],
                                          sample_distribution=[0.01, 0.49, 0.5])

loader = dataset.get_loader(shuffle=True)

for i, data in enumerate(loader):
    fn = data['fn'][0]
    cat = data["class"][0]
    export_path = os.path.join(OUTPUT_DIR, '{}/{}'.format(cat, fn))
    if not os.path.exists(os.path.join(OUTPUT_DIR, cat)):
        os.mkdir(os.path.join(OUTPUT_DIR, cat))
    print(i, i%200, cat, fn)
    if os.path.exists(export_path+".obj"):
        continue
    inputs = data['inputs'].to(device)
    mesh = generator.generate_mesh(inputs)
    mesh.export(export_path+".obj")

    

    