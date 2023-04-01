import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import sys
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from utils import AverageValueMeter, get_f_score
from external.PyTorchEMD.emd import earth_mover_distance
from chamfer_distance import ChamferDistance as distChamfer

CD_NUM = 100000
EMD_NUM = 2048
REMOVE_OUTLIERS = True
REMOVE_OUTLIERS_THRESHOLD = 0.01
WORK_DIR = "/home/xianghui_yang/ndf/experiments/shapenet_cube_offset_generalization_pt_k16_mean_loss_2_curl/"
OBJECTS_DIR_GT = "/home/xianghui_yang/data/ShapeNet/ShapeNetV1NDF_CUBE_OFFSET/"
OBJECTS_DIR = os.path.join(WORK_DIR, "mesh")
sys.stdout = open(os.path.join(WORK_DIR, "test_mesh_median.txt"), "w")
BASE_CLASSES = ["car","chair","plane", "table"]
NOVEL_CLASSES = ["speaker", "bench", "lamp", "watercraft"]
# BASE_CLASSES = ["car"]
# NOVEL_CLASSES = []
device = torch.device("cuda:0")

data_dir = '/home/xianghui_yang/data/ShapeNet/'
with open(os.path.join(data_dir, 'id2cat.json'), 'r') as f:
    idx2cls = json.load(f)
cls2idx = {}
for idx in idx2cls.keys():
    cls2idx[idx2cls[idx]]=idx

class TestDataset(data.Dataset):
    def __init__(self, categories=["car"], object_dir="/home/xianghui_yang/data/ShapeNet/ShapeNetV1NDF_CUBE_OFFSET/", object_dir_gt="/home/xianghui_yang/data/ShapeNet/ShapeNetV1NDF_CUBE_OFFSET/"):
        self.object_dir = object_dir
        self.object_dir_gt = object_dir_gt
        with open('/home/xianghui_yang/data/ShapeNet/id2cat.json', 'r') as f:
            self.idx2cls = json.load(f)
        self.cls2idx = {}
        for idx in idx2cls.keys():
            self.cls2idx[idx2cls[idx]]=idx
        
        self.file_list = []
        if len(categories)==1:
            data_np = np.load("/home/xianghui_yang/data/ShapeNet/ShapeNetV1NDF_CUBE_OFFSET/split_02958343.npz")['test']
            for fn in data_np:
                self.file_list.append((fn, "car", "02958343"))
        else:
            for cat in categories:
                idx = self.cls2idx[cat]
                data_np = np.load("/home/xianghui_yang/data/ShapeNet/ShapeNetV1NDF_CUBE_OFFSET/split_generalization_%s.npz"%(idx))['test']
                for fn in data_np:
                    self.file_list.append((fn, cat, idx))

        print(categories)
        print(len(self.file_list))

    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        fn, cat, idx = self.file_list[idx]
        verts_pred, faces_pred, _  = load_obj(os.path.join(self.object_dir, idx, fn+".obj"), load_textures=False)
        verts_gt, faces_gt, _  = load_obj(os.path.join(self.object_dir_gt, idx, fn+".obj"), load_textures=False)

        voxel_path = os.path.join(self.object_dir_gt, idx, fn, 'voxelized_point_cloud_256res_10000points.npz')
        input_points = torch.from_numpy(np.load(voxel_path)['point_cloud'].astype(np.float32))
        return verts_pred, verts_gt, faces_pred.verts_idx, faces_gt.verts_idx, input_points, cat, fn

val_distance = {}
val_cd_median = {}
val_normal = {}
val_emd = {}
val_fscore_tau = {}
val_fscore_2tau = {}
val_fscore_3tau = {}
val_fscore_4tau = {}
for cat in BASE_CLASSES+NOVEL_CLASSES:
    val_distance[cat] = AverageValueMeter()
    val_cd_median[cat] = AverageValueMeter()
    val_normal[cat] = AverageValueMeter()
    val_emd[cat] = AverageValueMeter()
    val_fscore_tau[cat] = AverageValueMeter()
    val_fscore_2tau[cat] = AverageValueMeter()
    val_fscore_3tau[cat] = AverageValueMeter()
    val_fscore_4tau[cat] = AverageValueMeter()

dataset = TestDataset(categories=BASE_CLASSES+NOVEL_CLASSES, object_dir=OBJECTS_DIR, object_dir_gt=OBJECTS_DIR_GT)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

for i, data in enumerate(dataloader):
    verts_pred, verts_gt, faces_pred_idx, faces_gt_idx, input_points, cat, fn = data
    if verts_pred.shape[1]==0 or faces_pred_idx.shape[1]==0:
        print(cat[0], fn[0])
        exit()
    if REMOVE_OUTLIERS:
        # print(cat[0], fn[0])
        dist_vp, _, _, _= distChamfer(verts_pred.to(device=device), input_points.to(device=device))
        dist_vp = dist_vp.squeeze(0).cpu()
        # print(dist_vp.shape, faces_pred_idx.shape)
        faces_pred_idx = faces_pred_idx[0]
        dist_fp = dist_vp[faces_pred_idx]
        # print(dist_fp.shape)
        dist_fp = torch.max(dist_fp, dim=1)[0]
        faces_pred_idx[dist_fp>REMOVE_OUTLIERS_THRESHOLD] = -1
        faces_pred_idx = faces_pred_idx.unsqueeze(0)
        # print(verts_pred.shape, faces_pred_idx.shape, verts_pred.device, faces_pred_idx.device)

    verts_pred, verts_gt, faces_pred_idx, faces_gt_idx, cat, fn = \
        verts_pred[0], verts_gt[0], faces_pred_idx[0], faces_gt_idx[0], cat[0], fn[0]

    meshes = Meshes(verts=[verts_pred, verts_gt], faces=[faces_pred_idx, faces_gt_idx]).to(device=device)
    pc, normals = sample_points_from_meshes(meshes, CD_NUM, return_normals=True)
    pred_pc = pc[0].unsqueeze(0)
    gt_pc = pc[1].unsqueeze(0)
    pred_normals = normals[0].unsqueeze(0)
    gt_normals = normals[1].unsqueeze(0)
    # meshes = Meshes(verts=[verts_gt], faces=[faces_gt_idx]).to(device=device)
    # gt_pc = sample_points_from_meshes(meshes, CD_NUM, return_normals=False)
    # pred_pc = input_points.to(device=device)
    # print(pred_pc.shape, gt_pc.shape)

    d1, d2, _, _= distChamfer(pred_pc, gt_pc)
    val_fscore = get_f_score(d1, d2, [1e-5, 2e-5, 2.5e-5, 1e-4])
    val_pc_chamfer_loss = (torch.mean(d1) + torch.mean(d2))/2
    val_pc_chamfer_loss, val_normal_loss = chamfer_distance(pred_pc, gt_pc, x_normals=pred_normals, y_normals=gt_normals)
    val_pc_chamfer_loss /= 2
    val_pc_chamfer_median = (torch.median(d1) + torch.median(d2)) /2 
    indices = np.random.choice(CD_NUM, size=EMD_NUM, replace=False)
    val_emd_loss = earth_mover_distance(pred_pc[:, indices, :], gt_pc[:, indices, :], transpose=False)
    
    
    val_distance[cat].update(val_pc_chamfer_loss.item())
    val_cd_median[cat].update(val_pc_chamfer_median.item())
    val_normal[cat].update(val_normal_loss.item())
    val_emd[cat].update(val_emd_loss[0].item()/EMD_NUM)
    val_fscore_tau[cat].update(val_fscore[0, 0].item()) 
    val_fscore_2tau[cat].update(val_fscore[0, 1].item())
    val_fscore_3tau[cat].update(val_fscore[0, 2].item())
    val_fscore_4tau[cat].update(val_fscore[0, 3].item())

for cat in BASE_CLASSES+NOVEL_CLASSES:
    print(cat)
    print(val_distance[cat].count, val_normal[cat].count, val_emd[cat].count, val_fscore_tau[cat].count, val_fscore_2tau[cat].count)

def calculate_metrics(eval_classes, class_split="mean"):
    if len(eval_classes)==0:
        return
    mean_chamfer = []
    mean_median = []
    mean_normal = []
    mean_emd = []
    mean_f_score_tau = []
    mean_f_score_2tau = []
    mean_f_score_3tau = []
    mean_f_score_4tau = []
    print("Class", "CD Mean", "CD Median", "Normal",  "EMD", "F-score tau", "F-score 2tau")
    for cat in eval_classes:
        mean_chamfer.append(val_distance[cat].avg*1e4)
        mean_median.append(val_cd_median[cat].avg*1e4)
        mean_normal.append(val_normal[cat].avg)
        mean_emd.append(val_emd[cat].avg*100)
        mean_f_score_tau.append(val_fscore_tau[cat].avg)
        mean_f_score_2tau.append(val_fscore_2tau[cat].avg)
        mean_f_score_3tau.append(val_fscore_3tau[cat].avg)
        mean_f_score_4tau.append(val_fscore_4tau[cat].avg)
        print(cat, "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f"%(val_distance[cat].avg*1e4, val_cd_median[cat].avg*1e4, val_normal[cat].avg, val_emd[cat].avg*100, 
            val_fscore_tau[cat].avg, val_fscore_2tau[cat].avg, val_fscore_3tau[cat].avg, val_fscore_4tau[cat].avg))

    print(class_split, 'cd mean: %.4f'%(np.mean(mean_chamfer)))
    print(class_split, 'cd median: %.4f'%(np.mean(mean_median)))
    print(class_split, 'normal: %.4f'%(np.mean(mean_normal)))
    print(class_split, 'EMD: %.4f'%(np.mean(mean_emd)))
    print(class_split, 'f-score 1e-5: %.4f'%(np.mean(mean_f_score_tau)))
    print(class_split, 'f-score 2e-5: %.4f'%(np.mean(mean_f_score_2tau)))
    print(class_split, 'f-score 2.5e-5: %.4f'%(np.mean(mean_f_score_3tau)))
    print(class_split, 'f-score 1e-4: %.4f'%(np.mean(mean_f_score_4tau)))
    return

print("=========Mean=========")
calculate_metrics(BASE_CLASSES+NOVEL_CLASSES, class_split="mean")
print("=========BASE=========")
calculate_metrics(BASE_CLASSES, class_split="base")
print("=========NOVEL=========")
calculate_metrics(NOVEL_CLASSES, class_split="novel")