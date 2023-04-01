import torch
import torch.nn as nn
import torch.nn.functional as F
from models.point_transformer import pointtransformer_seg_repro as encoder
from pytorch3d.ops import knn_points, knn_gather
from models.vector_quantizer import VectorQuantize

class NVF(nn.Module):
    def __init__(self, hidden_dim=256, k=8, pos_dim=128, out_dim=128, codebook_size=128, heads=4):
        super(NVF, self).__init__()
        self.k_nearest  = k
        self.pc_encoder = encoder(c=3, k=out_dim)
        self.codebook_size = codebook_size
        self.heads = heads
        self.vector_quantizer = VectorQuantize(dim=hidden_dim, codebook_size=codebook_size, threshold_ema_dead_code=4, codebook_dim=64, heads = heads, 
                                    separate_codebook_per_head=True, channel_last=False, commitment_weight=1e-3, orthogonal_reg_weight=1e-5,
                                    kmeans_init=True, kmeans_iters=10)
        self.fc_pos = nn.Linear(9, pos_dim)
        self.fc_0 = nn.Conv1d((out_dim+pos_dim)*k, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim*2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_3 = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.fc_4 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 3, 1)
        self.actvn = nn.ReLU()

    def encoder(self, x):
        point_featrues = self.pc_encoder(x)
        return point_featrues 

    def decoder(self, q, xyz, points):
        assert points.shape[1] == xyz.shape[1]
        B, Q, _ = q.shape
        _, idx, q_nearest_k = knn_points(q, xyz, K=self.k_nearest, return_nn=True)
        q_nearest_k_feature = knn_gather(points, idx)
        qk = q.unsqueeze(2).expand(B, Q, self.k_nearest, 3)
        pos_feature = self.actvn(self.fc_pos(torch.cat([qk, q_nearest_k, q_nearest_k-qk], dim=-1)))
        features = torch.cat([pos_feature, q_nearest_k_feature], dim=-1).view(B, Q, -1).transpose(1,2).contiguous() # [B, Q, K, F] => [B, Q, K*F] => [B, K*F, Q]
        features = self.actvn(self.fc_0(features))
        features = self.actvn(self.fc_1(features))
        features = self.actvn(self.fc_2(features))
        features_vq, embed_ind, loss_vq = self.vector_quantizer(features)
        net = self.actvn(self.fc_3(torch.cat([features, features_vq], dim=1)))
        net = self.actvn(self.fc_4(net))
        net = self.fc_out(net)
        out = net.transpose(1,2)
        usage = len(torch.unique(embed_ind[:, :, 0]))/self.codebook_size
        return out, loss_vq, usage


    def forward(self, q, x):
        out, loss_vq, usage = self.decoder(q, x, self.encoder(x))
        return out, loss_vq, usage

if __name__=="__main__":
    device = torch.device("cuda:0")
    points = torch.rand(4, 10000, 3).to(device)
    query_points = torch.rand(4, 50000, 3).to(device)
    model = NVF().to(device)
    prediction, loss_vq, perplexity = model(query_points, points)
    print(prediction.shape)
    print(loss_vq.shape, loss_vq)
    print(perplexity)