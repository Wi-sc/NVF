import sys
sys.path.append("./")
import models.local_model_offset_point_transformer_vq as model
import models.data.voxelized_data_shapenet_cube_offset_generalization_points as voxelized_data
from models import training_vq as training
import torch
import configs.config_loader as cfg_loader


cfg = cfg_loader.get_config()
net = model.NVF(k=cfg.query_available_points)
print(cfg)
print(net)

train_dataset = voxelized_data.VoxelizedDataset('train',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=["02958343", "03001627", "02691156", "04379243"],
                                          batch_size=cfg.batch_size,
                                          num_sample_points=cfg.num_sample_points_training,
                                          num_workers=12,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)
val_dataset = voxelized_data.VoxelizedDataset('val',
                                          res=cfg.input_res,
                                          pointcloud_samples=cfg.num_points,
                                          data_path=cfg.data_dir,
                                          split_file=["02958343", "03001627", "02691156", "04379243", "02828884", "04530566", "03636649", "03691459"],
                                          batch_size=cfg.batch_size,
                                          num_sample_points=50000,
                                          num_workers=4,
                                          sample_distribution=cfg.sample_ratio,
                                          sample_sigmas=cfg.sample_std_dev)



trainer = training.Trainer(net,
                           torch.device("cuda"),
                           train_dataset,
                           val_dataset,
                           cfg.exp_name,
                           optimizer=cfg.optimizer,
                           lr=cfg.lr)

trainer.train_model(cfg.num_epochs)
