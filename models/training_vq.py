from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import time
import trimesh



class Trainer(object):

    def __init__(self, model, device, train_dataset, val_dataset, exp_name, optimizer='Adam', lr = 1e-4, threshold = 0.1):
        self.model = model.to(device)
        self.device = device
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr= lr)
        if optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters())
        if optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [30, 70, 120, 200], 0.3)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_data_iterator = self.val_dataset.get_loader()

        self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None
        self.max_dist = threshold


    def train_step(self,batch):
        self.model.train()
        self.optimizer.zero_grad()
        loss, loss_dict = self.compute_loss(batch)
        loss.backward()
        self.optimizer.step()

        return loss.item(), loss_dict

    def compute_loss(self, batch, return_loss_vq=True):
        device = self.device

        p = batch.get('grid_coords').to(device)
        offset_gt = batch.get('offsets').to(device)
        inputs = batch.get('inputs').to(device)

        offset_pred, loss_vq, perplexity = self.model(p, inputs)
        assert offset_pred.shape[-1]==offset_gt.shape[-1]==3

        loss_offset = torch.nn.L1Loss(reduction='none')(offset_pred, offset_gt)
        loss_vq = loss_vq/inputs.shape[0]
        loss_offset = loss_offset.mean()
        if return_loss_vq==False:
            return loss_offset
        loss = loss_offset + loss_vq
        return loss, {"offset":loss_offset.item(), "vq":loss_vq.item(), "usage":perplexity}

    def train_model(self, epochs):
        loss = 0
        train_data_loader = self.train_dataset.get_loader()
        start, training_time = self.load_checkpoint()
        iteration_start_time = time.time()

        for epoch in range(start, epochs):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))
            print('Current Lr: ', self.lr_scheduler.get_last_lr()[0])
            print("Current mean", self.model.vector_quantizer._codebook.embed[0, 0, :])
            print("Current sigma", self.model.vector_quantizer._codebook.sigma_codebook[0, 0, :])
            for i, batch in enumerate(train_data_loader):
                #save model
                iteration_duration = time.time() - iteration_start_time

                if iteration_duration > 60 * 60:  # eve model every X min and at start
                    training_time += iteration_duration
                    iteration_start_time = time.time()

                    self.save_checkpoint(epoch, training_time)
                    val_loss = self.compute_val_loss()

                    if self.val_min is None:
                        self.val_min = val_loss

                    if val_loss < self.val_min:
                        self.val_min = val_loss
                        for path in glob(self.exp_path + 'val_min=*'):
                            os.remove(path)
                        np.save(self.exp_path + 'val_min={}-time={}h'.format(epoch, convertSecs(training_time)[0]), [epoch, val_loss])

                    self.writer.add_scalar('val loss batch avg', val_loss, epoch)

                #optimize model
                loss, loss_dict = self.train_step(batch)
                if i%50==0:
                    print("{:04}: Current loss: {:.4}".format(i, loss / self.train_dataset.num_sample_points), end=" - ")
                    # print("{:04}: Current loss: {}".format(i, loss), end=" - ")
                    print(" - ".join("{}: {:.8}".format(k, v) for k, v in loss_dict.items()))
                sum_loss += loss

            self.lr_scheduler.step()


            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', sum_loss / len(train_data_loader), epoch)




    def save_checkpoint(self, epoch, training_time):
        path = self.checkpoint_path + 'checkpoint_{}_{}h:{}m:{}s_{}.tar'.format(epoch, *[*convertSecs(training_time), training_time])
        if not os.path.exists(path):
            torch.save({ #'state': torch.cuda.get_rng_state_all(),
                        'training_time': training_time ,'epoch':epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'lr_scheduler_state_dict': self.lr_scheduler.state_dict()}, path)



    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0,0

        checkpoints = [(os.path.splitext(os.path.basename(path))[0].split('_')[-1], os.path.splitext(os.path.basename(path))[0].split('_')[1]) for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints, axis=0)
        path = self.checkpoint_path + 'checkpoint_{}_{}h:{}m:{}s_{}.tar'.format(int(checkpoints[-1][1]), *[*convertSecs(checkpoints[-1][0]), checkpoints[-1][0]])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        training_time = checkpoint['training_time']
        # torch.cuda.set_rng_state_all(checkpoint['state']) # batch order is restored. unfortunately doesn't work like that.
        return epoch, training_time

    def compute_val_loss(self):
        self.model.eval()

        sum_val_loss = 0
        num_batches = 0
        for i, val_batch in enumerate(self.val_data_iterator):
            val_loss = self.compute_loss(val_batch, return_loss_vq=False).item()
            sum_val_loss += val_loss
            num_batches += 1
            if i%50==0:
                print("Val {:04}: Current loss: {:.4}".format(i, val_loss))
        
        self.visualize(val_batch)

        return sum_val_loss / num_batches
    
    def visualize(self, data):
        device = self.device
        p = data.get('grid_coords').to(device)
        inputs = data.get('inputs').to(device)
        offset_pred, _, _ = self.model(p, inputs)
        point_cloud = p + offset_pred
        point_cloud = point_cloud[0].detach().cpu().numpy()
        trimesh.Trimesh(vertices=point_cloud, faces=[]).export(self.exp_path + 'vis_pred.ply')
        trimesh.Trimesh(vertices=data.get('grid_coords')[0].cpu().numpy()+data.get('offsets')[0].cpu().numpy(), faces=[]).export(self.exp_path + 'vis_gt.ply')
        return 


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds

def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds